#!/usr/bin/env python3
import logging
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import load_settings
from detector import AiDetector
from reddit_client import RateLimitedReddit
from storage import Storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("isthisai")


@dataclass(order=True)
class QueueJob:
    priority: int
    sequence: int
    kind: str = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)


class Bot:
    def __init__(self):
        load_dotenv()
        self.settings = load_settings()
        self.storage = Storage(self.settings.db_path)
        self.detector = AiDetector(self.settings.model_name)
        self.reddit = RateLimitedReddit(
            client_id=self.settings.reddit_client_id,
            client_secret=self.settings.reddit_client_secret,
            username=self.settings.reddit_username,
            password=self.settings.reddit_password,
            user_agent=self.settings.reddit_user_agent,
            calls_per_minute=self.settings.api_calls_per_minute,
        )

        self.jobs: queue.PriorityQueue[QueueJob] = queue.PriorityQueue()
        self.stop_event = threading.Event()
        self.poll_in_progress = threading.Event()
        self._seq = 0
        self._seq_lock = threading.Lock()

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def enqueue_poll(self) -> None:
        self.jobs.put(QueueJob(priority=1, sequence=self._next_seq(), kind="poll"))

    def enqueue_fetch_reply(self, comment_id: str) -> None:
        self.jobs.put(
            QueueJob(
                priority=2,
                sequence=self._next_seq(),
                kind="fetch_and_reply",
                payload={"comment_id": comment_id},
            )
        )

    def scheduler_loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.poll_in_progress.is_set():
                self.poll_in_progress.set()
                self.enqueue_poll()
            self.stop_event.wait(self.settings.poll_interval_seconds)

    def worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                job = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if job.kind == "poll":
                    self.handle_poll_job()
                elif job.kind == "fetch_and_reply":
                    self.handle_fetch_and_reply(job.payload["comment_id"])
            except Exception:
                LOGGER.exception("Unhandled error while processing %s", job.kind)
                if job.kind == "poll":
                    self.poll_in_progress.clear()
            finally:
                self.jobs.task_done()

    def handle_poll_job(self) -> None:
        last_seen = self.storage.get_last_seen_id()
        comments = self.reddit.get_comments(limit=100, after=last_seen)

        if not comments:
            self.poll_in_progress.clear()
            return

        # Persist newest seen comment id immediately to minimize restart gaps.
        self.storage.set_last_seen_id(comments[0].fullname)

        for comment in reversed(comments):
            if self.settings.command_trigger in comment.body.lower():
                if self.storage.has_replied(comment.id):
                    continue
                self.enqueue_fetch_reply(comment.id)

        if len(comments) == 100:
            self.enqueue_poll()
        else:
            self.poll_in_progress.clear()

    def handle_fetch_and_reply(self, comment_id: str) -> None:
        if self.storage.has_replied(comment_id):
            return

        comment = self.reddit.fetch_comment(comment_id)
        submission = self.reddit.fetch_parent_submission(comment)
        text = (submission.selftext or "").strip()

        if not text:
            body = (
                "🤖 **AI Analysis:** I can only analyze text posts with body content.\n\n"
                "*Note: AI detection is never definitive.*"
            )
            self.reddit.post_reply(comment, body)
            self.storage.mark_replied(comment_id)
            return

        words = len(text.split())
        result = self.detector.detect(text)
        pct = int(round(result.probability_ai * 100))
        confidence = self._confidence_band(result.probability_ai)

        warning = ""
        if words < self.settings.min_words_warning:
            warning = (
                f"\n\n⚠️ Short text warning: this post is only {words} words, "
                "so detector accuracy may be lower."
            )

        body = (
            f"🤖 **AI Analysis:** {pct}% likely AI-generated "
            f"*(confidence: {confidence} - post is {words} words)*\n"
            f"Signal: classifier score {result.probability_ai:.2f}\n\n"
            f"*Note: AI detection is never definitive.*"
            f"{warning}"
        )

        self.reddit.post_reply(comment, body)
        self.storage.mark_replied(comment_id)

    @staticmethod
    def _confidence_band(score: float) -> str:
        if score >= 0.8 or score <= 0.2:
            return "high"
        if score >= 0.65 or score <= 0.35:
            return "medium"
        return "low"

    def run(self) -> None:
        scheduler = threading.Thread(target=self.scheduler_loop, name="scheduler", daemon=True)
        worker = threading.Thread(target=self.worker_loop, name="worker", daemon=True)
        scheduler.start()
        worker.start()

        def _shutdown(signum, _frame):
            LOGGER.info("Received signal %s, shutting down", signum)
            self.stop_event.set()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        while not self.stop_event.is_set():
            time.sleep(0.5)

        scheduler.join(timeout=2)
        worker.join(timeout=2)


if __name__ == "__main__":
    Bot().run()
