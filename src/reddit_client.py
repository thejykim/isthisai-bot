import logging
import threading
import time
from typing import Optional

import praw
from praw.models import Comment, Submission

LOGGER = logging.getLogger(__name__)


class TokenBucket:
    def __init__(self, capacity: int, refill_per_second: float):
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._refill_per_second = refill_per_second
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, amount: float = 1.0) -> None:
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                missing = amount - self._tokens
                wait_seconds = missing / self._refill_per_second
            time.sleep(max(wait_seconds, 0.01))

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_second)
        self._last_refill = now


class RateLimitedReddit:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        username: str,
        password: str,
        user_agent: str,
        calls_per_minute: int,
    ):
        self._reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            username=username,
            password=password,
            user_agent=user_agent,
        )
        self._bucket = TokenBucket(capacity=calls_per_minute, refill_per_second=calls_per_minute / 60.0)

    def get_comments(self, *, limit: int = 100, after: Optional[str] = None) -> list[Comment]:
        self._bucket.consume()
        params = {"after": after} if after else None
        comments = list(self._reddit.subreddit("all").comments(limit=limit, params=params))
        LOGGER.debug("Fetched %s comments (after=%s)", len(comments), after)
        return comments

    def fetch_parent_submission(self, comment: Comment) -> Submission:
        self._bucket.consume()
        comment.refresh()
        return comment.submission

    def fetch_comment(self, comment_id: str) -> Comment:
        self._bucket.consume()
        return self._reddit.comment(comment_id)

    def post_reply(self, comment: Comment, body: str) -> None:
        self._bucket.consume()
        comment.reply(body)
