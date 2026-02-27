#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from detector import AiDetector


DEFAULT_MODEL_NAME = "ShantanuT01/BERT-tiny-RAID"


def confidence_band(score: float) -> str:
    if score >= 0.8 or score <= 0.2:
        return "high"
    if score >= 0.65 or score <= 0.35:
        return "medium"
    return "low"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AI text detection locally without Reddit API access."
    )
    parser.add_argument("text", nargs="?", help="Text to analyze.")
    parser.add_argument("--file", "-f", help="Read text from file.")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read text from stdin (for piping).",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Prompt for text in an interactive loop.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model name (default: {DEFAULT_MODEL_NAME} or MODEL_NAME env var).",
    )
    return parser.parse_args()


def _read_text(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8").strip()
    if args.stdin:
        return sys.stdin.read().strip()
    if args.text:
        return args.text.strip()
    return ""


def _print_result(text: str, detector: AiDetector) -> None:
    words = len(text.split())
    result = detector.detect(text)
    pct = int(round(result.probability_ai * 100))
    confidence = confidence_band(result.probability_ai)

    print(f"AI likelihood: {pct}%")
    print(f"Confidence: {confidence}")
    print(f"Score: {result.probability_ai:.4f}")
    print(f"Model label: {result.label}")
    print(f"Words: {words}")


def _run_interactive(detector: AiDetector) -> int:
    print("Interactive mode. Enter text to analyze. Press Ctrl-D to exit.")
    while True:
        try:
            text = input("\nText> ").strip()
        except EOFError:
            print()
            return 0
        if not text:
            print("No input provided.")
            continue
        _print_result(text, detector)


def main() -> int:
    load_dotenv()
    args = parse_args()
    selected_inputs = sum(bool(v) for v in [args.text, args.file, args.stdin, args.interactive])
    if selected_inputs > 1:
        print("Choose only one input mode: positional text, --file, --stdin, or --interactive.")
        return 2

    model_name = args.model or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    detector = AiDetector(model_name)

    if args.interactive:
        return _run_interactive(detector)

    text = _read_text(args)
    if not text:
        print("No input text provided. Use positional text, --file, --stdin, or --interactive.")
        return 2

    _print_result(text, detector)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
