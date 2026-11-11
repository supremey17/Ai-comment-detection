"""Step 1: clean the human tweets and assign train/val/test splits.

    python -m aicd.prepare_human --input data/raw/train.xlsx

Only the tweet text and sentiment are kept. The dataset's "Time of Tweet",
"Age of User" and "Country" columns are fabricated (they cycle with the row
number), so they are dropped and must never be used as features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import stable_rng, write_jsonl
from .normalize import normalize, word_count

SPLITS = (("train", 0.8), ("val", 0.1), ("test", 0.1))


def assign_split(key: str, seed: int) -> str:
    r = stable_rng("split", seed, key).random()
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if r < acc:
            return name
    return SPLITS[-1][0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data/raw/train.xlsx"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/human.jsonl"))
    ap.add_argument("--min-words", type=int, default=3, help="drop tweets shorter than this after cleanup")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if args.input.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)
    if "text" not in df.columns:
        raise SystemExit(f"{args.input} has no 'text' column (columns: {list(df.columns)})")

    n_in = len(df)
    df = df.dropna(subset=["text"]).copy()
    df["raw_text"] = df["text"].astype(str)
    df["text"] = df["raw_text"].map(normalize)
    df["n_words"] = df["text"].map(word_count)
    short = df["n_words"] < args.min_words
    df = df[~short]
    before_dedupe = len(df)
    df["_key"] = df["text"].str.lower()
    df = df.drop_duplicates("_key")

    id_col = "textID" if "textID" in df.columns else None
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        hid = f"h_{getattr(r, id_col)}" if id_col else f"h_{i:06d}"
        rows.append(
            {
                "id": hid,
                "text": r.text,
                "raw_text": r.raw_text,
                "sentiment": getattr(r, "sentiment", None),
                "n_words": int(r.n_words),
                "split": assign_split(hid, args.seed),
            }
        )

    n = write_jsonl(args.output, rows)
    split_counts = pd.Series([r["split"] for r in rows]).value_counts().to_dict()
    print(f"read {n_in} rows; dropped {int(short.sum())} under {args.min_words} words; "
          f"dropped {before_dedupe - len(df)} duplicates")
    print(f"wrote {n} human tweets to {args.output}  splits={split_counts}")


if __name__ == "__main__":
    main()
