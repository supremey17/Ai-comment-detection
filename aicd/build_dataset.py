"""Step 3: merge human and AI rows into one labeled dataset.

    python -m aicd.build_dataset
    python -m aicd.build_dataset --extra-human-test data/raw/modern_comments.csv

Writes data/processed/dataset.csv with columns:
  id, pair_id, text, label (0 = human, 1 = AI), split, source, generator,
  mode, style, persona, sentiment

Splits:
  train / val      for training and choosing settings
  test             final check, on models also used in training
  test_holdout     final check on a model never used in training, plus the
                   human tweets those AI posts were paired with
  test_modern_human  (optional) recent human comments you collected; shows
                   how often real present-day people get flagged

Only human tweets that have an AI counterpart are kept, so the classes stay
balanced and each pair lands in the same split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import read_jsonl
from .normalize import normalize, word_count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", type=Path, default=Path("data/processed/human.jsonl"))
    ap.add_argument("--ai", type=Path, default=Path("data/processed/ai.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/dataset.csv"))
    ap.add_argument("--extra-human-test", type=Path, nargs="*", default=[],
                    help="CSV files of recent human comments to add as a test-only split")
    ap.add_argument("--text-col", default="text", help="text column name in the extra CSV files")
    ap.add_argument("--keep-emoji", action="store_true",
                    help="keep emoji (only sensible once you have human data that contains emoji)")
    args = ap.parse_args()

    humans = {r["id"]: r for r in read_jsonl(args.human)}
    ai_rows = list(read_jsonl(args.ai))
    if not ai_rows:
        raise SystemExit(f"{args.ai} is empty; run `python -m aicd.generate_ai` first")

    strip_emoji = not args.keep_emoji
    rows = []
    for a in ai_rows:
        h = humans.get(a["pair_id"])
        if h is None:
            continue
        split = a["split"]  # "test_holdout" overrides the human's "test"
        rows.append({"id": h["id"], "pair_id": h["id"], "text": normalize(h["raw_text"], strip_emoji=strip_emoji),
                     "label": 0, "split": split, "source": "tweets_2009", "generator": "",
                     "mode": "", "style": "", "persona": "", "sentiment": h.get("sentiment")})
        rows.append({"id": a["id"], "pair_id": h["id"], "text": normalize(a["text"], strip_emoji=strip_emoji),
                     "label": 1, "split": split, "source": "generated", "generator": a["generator"],
                     "mode": a["mode"], "style": a["style"], "persona": a["persona"],
                     "sentiment": a.get("sentiment")})

    for path in args.extra_human_test:
        ext = pd.read_csv(path)
        if args.text_col not in ext.columns:
            raise SystemExit(f"{path} has no column {args.text_col!r}")
        for i, t in enumerate(ext[args.text_col].dropna().astype(str)):
            rid = f"m_{path.stem}_{i:06d}"
            rows.append({"id": rid, "pair_id": rid, "text": normalize(t, strip_emoji=strip_emoji), "label": 0,
                         "split": "test_modern_human", "source": path.stem, "generator": "",
                         "mode": "", "style": "", "persona": "", "sentiment": None})

    df = pd.DataFrame(rows)
    df = df[df["text"].map(word_count) >= 2]
    # Same text in two places would leak between splits; keep the first.
    dup = df["text"].str.lower().duplicated()
    if dup.any():
        print(f"dropping {int(dup.sum())} duplicate texts")
        df = df[~dup]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows to {args.output}")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns={0: "human", 1: "ai"}))
    gen = df[df.label == 1].groupby(["generator", "split"]).size().unstack(fill_value=0)
    print("\nAI rows by generator:\n", gen)


if __name__ == "__main__":
    main()
