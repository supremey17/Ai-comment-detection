"""Step 4: sanity-check the dataset before training a real model.

    python -m aicd.check_leakage

Trains two quick baselines on the train split and reports them on val and
the test splits:

  1. surface features only (length, capitals, punctuation, digits...).
     If these alone separate the classes almost perfectly, the dataset has a
     formatting shortcut and a real model will learn it instead of AI style.
  2. TF-IDF + logistic regression (the baseline a fine-tuned transformer has
     to beat). Its most telling words/characters are printed so you can spot
     artifacts (for example a token that only one side ever uses).

Metrics: AUROC (0.5 = guessing, 1.0 = perfect) and TPR at 1% FPR (share of
AI posts caught when only 1 in 100 human posts is wrongly flagged).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


def surface_features(texts: pd.Series) -> np.ndarray:
    def f(t: str) -> list[float]:
        n = max(len(t), 1)
        words = t.split()
        return [
            len(t), len(words),
            sum(c.isupper() for c in t) / n,
            float(t[:1].isupper()),
            float(t.rstrip()[-1:] in ".!?"),
            t.count("!") / n, t.count("?") / n, t.count(",") / n, t.count(".") / n,
            t.count("'") / n, t.count('"') / n, t.count("-") / n,
            sum(c.isdigit() for c in t) / n,
            float("URL" in t), float("****" in t), float("..." in t),
            len(re.findall(r"(.)\1\1", t)) / n,
            np.mean([len(w) for w in words]) if words else 0.0,
        ]
    return np.array([f(t) for t in texts], dtype=float)


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def report(name: str, y: np.ndarray, s: np.ndarray) -> None:
    if len(set(y)) < 2:
        print(f"  {name:20s} n={len(y):6d}  (one class only) mean score={s.mean():.3f}")
        return
    print(f"  {name:20s} n={len(y):6d}  AUROC={roc_auc_score(y, s):.3f}  "
          f"TPR@1%FPR={tpr_at_fpr(y, s, 0.01):.3f}  TPR@5%FPR={tpr_at_fpr(y, s, 0.05):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/processed/dataset.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False)
    train = df[df.split == "train"]
    if train.label.nunique() < 2:
        raise SystemExit("train split needs both classes")
    evals = [s for s in ("val", "test", "test_holdout", "test_modern_human") if (df.split == s).any()]

    print("\n[1] surface features only (should NOT be near-perfect)")
    scaler = StandardScaler().fit(surface_features(train.text))
    lr = LogisticRegression(max_iter=2000).fit(scaler.transform(surface_features(train.text)), train.label)
    surface_auc = None
    for s in evals:
        part = df[df.split == s]
        scores = lr.predict_proba(scaler.transform(surface_features(part.text)))[:, 1]
        report(s, part.label.values, scores)
        if s == "val":
            surface_auc = roc_auc_score(part.label, scores)

    print("\n[2] TF-IDF word + character n-grams + logistic regression")
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, lowercase=False)
    Xtr = hstack([wv.fit_transform(train.text), cv.fit_transform(train.text)]).tocsr()
    clf = LogisticRegression(max_iter=4000, C=4.0).fit(Xtr, train.label)
    for s in evals:
        part = df[df.split == s]
        X = hstack([wv.transform(part.text), cv.transform(part.text)]).tocsr()
        report(s, part.label.values, clf.predict_proba(X)[:, 1])

    # Per-generator catch rate at the threshold that flags 1% of val humans.
    val = df[df.split == "val"]
    if (val.label == 0).any():
        Xv = hstack([wv.transform(val.text), cv.transform(val.text)]).tocsr()
        sv = clf.predict_proba(Xv)[:, 1]
        thr = float(np.quantile(sv[val.label.values == 0], 0.99))
        print(f"\n    threshold flagging 1% of val humans: {thr:.3f}; share of AI posts caught, by generator:")
        ai = df[(df.label == 1) & df.split.isin(["test", "test_holdout"])]
        Xa = hstack([wv.transform(ai.text), cv.transform(ai.text)]).tocsr()
        ai = ai.assign(flag=clf.predict_proba(Xa)[:, 1] >= thr)
        print(ai.groupby(["split", "generator"]).flag.agg(["mean", "size"]).round(3).to_string())
        mh = df[df.split == "test_modern_human"]
        if len(mh):
            Xm = hstack([wv.transform(mh.text), cv.transform(mh.text)]).tocsr()
            print(f"    modern human comments wrongly flagged: {(clf.predict_proba(Xm)[:, 1] >= thr).mean():.1%}")

    names = np.array(list(wv.get_feature_names_out()) + [f"chr:{c!r}" for c in cv.get_feature_names_out()])
    coef = clf.coef_[0]
    order = np.argsort(coef)
    print("\n    features pointing most to HUMAN:", ", ".join(names[order[:25]]))
    print("    features pointing most to AI:   ", ", ".join(names[order[-25:][::-1]]))

    if surface_auc is not None and surface_auc > 0.9:
        print(f"\nWARNING: surface features alone reach AUROC {surface_auc:.3f} on val. Look for a formatting "
              "difference between the classes (length, capitals, punctuation) and fix it in the prompts "
              "or in aicd/normalize.py before training.")


if __name__ == "__main__":
    main()
