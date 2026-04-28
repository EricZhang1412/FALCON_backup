"""Sample evaluation texts from WikiText and save as one-line-per-sample file.

Examples:
  python sample_wikitext_texts.py --num-samples 500
  python sample_wikitext_texts.py --subset wikitext-2-raw-v1 --split test --num-samples 200
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample clean lines from WikiText.")
    parser.add_argument(
        "--subset",
        type=str,
        default="wikitext-103-raw-v1",
        choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
        help="WikiText subset on HuggingFace datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split",
    )
    parser.add_argument("--num-samples", type=int, default=500, help="Number of output lines")
    parser.add_argument("--min-chars", type=int, default=40, help="Minimum non-space chars per line")
    parser.add_argument("--max-chars", type=int, default=280, help="Maximum chars per line before truncation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval_texts_wikitext.txt",
        help="Output text file path (one line per sample)",
    )
    return parser


def _clean_line(text: str, min_chars: int, max_chars: int) -> str | None:
    s = " ".join(text.strip().split())
    if not s:
        return None
    # Skip article titles/section markers like "= Foo ="
    if s.startswith("=") and s.endswith("="):
        return None
    if len(s.replace(" ", "")) < min_chars:
        return None
    if len(s) > max_chars:
        s = s[:max_chars].rstrip()
    return s


def main() -> int:
    args = _build_parser().parse_args()
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from exc

    ds = load_dataset("wikitext", args.subset, split=args.split)
    cleaned: list[str] = []
    for item in ds:
        c = _clean_line(str(item.get("text", "")), args.min_chars, args.max_chars)
        if c is not None:
            cleaned.append(c)

    if not cleaned:
        raise ValueError("No usable lines found after cleaning/filtering.")

    rng = random.Random(args.seed)
    if len(cleaned) >= args.num_samples:
        selected = rng.sample(cleaned, args.num_samples)
    else:
        # If requested samples exceed pool size, sample with replacement.
        selected = [rng.choice(cleaned) for _ in range(args.num_samples)]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(selected) + "\n", encoding="utf-8")

    print(f"Saved {len(selected)} samples to: {out_path}")
    print(f"Subset/split: wikitext/{args.subset} ({args.split})")
    print(f"Candidate pool after cleaning: {len(cleaned)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
