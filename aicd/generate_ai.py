"""Step 2: write one AI post for each human tweet.

    python -m aicd.generate_ai --dry-run            # show the plan and sample prompts, no API calls
    python -m aicd.generate_ai --limit 200          # small trial run
    python -m aicd.generate_ai                      # everything

Output goes to data/processed/ai.jsonl. The run is resumable: rows already
in the output file are skipped, so you can stop it (Ctrl+C) and start it
again. Rows that failed are logged to ai_failures.jsonl and retried on the
next run.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional; plain environment variables work too
    load_dotenv = None

from .common import read_jsonl, stable_rng, weighted_choice
from .normalize import normalize, word_count
from .prompts import SYSTEM_PROMPT, build_prompt, clean_output, too_similar
from .providers import Generator, from_config

BANNED_PHRASES = ("as an ai", "language model", "i can't help", "i cannot help", "<post>")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def enabled_generators(cfg: dict, only: Optional[list[str]] = None) -> list[Generator]:
    gens = [from_config(e) for e in cfg.get("generators", []) if e.get("enabled")]
    if only:
        gens = [g for g in gens if g.name in only]
    return gens


def plan_row(row: dict, cfg: dict, gens: list[Generator], seed: int) -> dict:
    """Decide generator, mode, style, persona and temperature for one row.

    Deterministic in (row id, seed), so re-runs and resumes make the same choices.
    """
    rng = stable_rng("plan", seed, row["id"])
    train_gens = [g for g in gens if not g.holdout] or gens
    holdout = [g for g in gens if g.holdout]
    split = row["split"]
    if split == "test" and holdout and rng.random() < float(cfg.get("holdout_share_of_test", 0.5)):
        gen = holdout[0]
        split = "test_holdout"
    else:
        gen = weighted_choice(rng, [{"g": g, "weight": g.weight} for g in train_gens])["g"]
    mode = weighted_choice(rng, cfg["modes"])["name"]
    style = weighted_choice(rng, cfg["styles"])["name"]
    persona = rng.choice(cfg["personas"])
    if gen.temperature is not None:
        temp = float(gen.temperature)
    else:
        lo, hi = cfg.get("temperature_range", [0.7, 1.1])
        temp = round(rng.uniform(lo, hi), 2)
    if gen.provider == "anthropic":
        temp = None  # current Claude models reject sampling parameters
    return {"gen": gen, "mode": mode, "style": style, "persona": persona,
            "temperature": temp, "split": split}


def validate(text: Optional[str], seed: str, n_words: int) -> tuple[bool, str]:
    if not text:
        return False, "empty"
    low = text.lower()
    if any(p in low for p in BANNED_PHRASES):
        return False, "meta_text"
    norm = normalize(text)
    n = word_count(norm)
    if n < 2:
        return False, "too_short"
    if n > 3 * n_words + 15 or len(text) > 400:
        return False, "too_long"
    if too_similar(norm, seed):
        return False, "copied_seed"
    return True, "ok"


def generate_one(row: dict, plan: dict, max_attempts: int) -> dict:
    gen: Generator = plan["gen"]
    prompt = build_prompt(seed=row["text"], mode=plan["mode"], persona=plan["persona"],
                          style=plan["style"], sentiment=row.get("sentiment"), n_words=row["n_words"])
    reasons = []
    for attempt in range(1, max_attempts + 1):
        raw = gen.generate(SYSTEM_PROMPT, prompt, plan["temperature"])
        if raw is None:
            reasons.append("declined")
            continue
        text = clean_output(raw)
        ok, why = validate(text, row["text"], row["n_words"])
        if ok:
            return {
                "id": "a_" + row["id"][2:] if row["id"].startswith("h_") else "a_" + row["id"],
                "pair_id": row["id"],
                "text": text,
                "label": 1,
                "split": plan["split"],
                "generator": gen.name,
                "provider": gen.provider,
                "model": gen.model,
                "mode": plan["mode"],
                "style": plan["style"],
                "persona": plan["persona"],
                "temperature": plan["temperature"],
                "sentiment": row.get("sentiment"),
                "attempts": attempt,
            }
        reasons.append(why)
    raise RuntimeError(f"no valid output after {max_attempts} attempts: {reasons}")


def preflight(gens: list[Generator]) -> None:
    """Call each generator once so a missing key or bad model name fails fast."""
    print("checking each generator with one test call...")
    bad = []
    for g in gens:
        try:
            out = g.generate(SYSTEM_PROMPT, "Write a short post about your morning coffee. About 10 words.", None)
            print(f"  ok   {g.name}: {clean_output(out)!r}")
        except Exception as e:  # noqa: BLE001 - report every kind of setup failure
            bad.append(g.name)
            print(f"  FAIL {g.name}: {type(e).__name__}: {e}")
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                print("       -> no API key found. Put it in a .env file (see .env.example).")
    if bad:
        sys.exit(f"fix or disable these generators in the config first: {', '.join(bad)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/generation.yaml"))
    ap.add_argument("--human", type=Path, default=Path("data/processed/human.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/ai.jsonl"))
    ap.add_argument("--failures", type=Path, default=Path("data/processed/ai_failures.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="only generate for this many tweets (a fixed random subset)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests")
    ap.add_argument("--only", nargs="*", help="only run these generator names (others' rows are skipped for now)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and sample prompts without calling any model")
    ap.add_argument("--no-preflight", action="store_true")
    args = ap.parse_args()

    if load_dotenv is not None:
        load_dotenv()  # reads API keys from a local .env file (which git ignores)
    cfg = load_config(args.config)
    all_gens = enabled_generators(cfg)
    if not all_gens:
        sys.exit(f"no generators enabled in {args.config}; set `enabled: true` on at least two")
    holdouts = [g.name for g in all_gens if g.holdout]
    if len(holdouts) > 1:
        sys.exit(f"only one enabled generator may have holdout: true (found {holdouts})")
    if not holdouts:
        print("warning: no holdout generator enabled; you won't be able to test on an unseen model")
    if len(all_gens) - len(holdouts) < 2:
        print("warning: fewer than two training generators; the detector will overfit to one model's style")

    humans = list(read_jsonl(args.human))
    if not humans:
        sys.exit(f"{args.human} is empty or missing; run `python -m aicd.prepare_human` first")
    humans.sort(key=lambda r: stable_rng("order", args.seed, r["id"]).random())
    full_counts = Counter(plan_row(r, cfg, all_gens, args.seed)["gen"].name for r in humans)
    if args.limit:
        humans = humans[: args.limit]

    plans = {r["id"]: plan_row(r, cfg, all_gens, args.seed) for r in humans}
    if args.only:
        humans = [r for r in humans if plans[r["id"]]["gen"].name in args.only]

    done = {r["pair_id"] for r in read_jsonl(args.output)}
    todo = [r for r in humans if r["id"] not in done]

    counts = Counter((plans[r["id"]]["gen"].name, plans[r["id"]]["split"]) for r in todo)
    print(f"{len(humans)} tweets selected, {len(humans) - len(todo)} already done, {len(todo)} to generate")
    for (g, s), c in sorted(counts.items()):
        print(f"  {g:24s} {s:13s} {c}")

    if args.dry_run:
        for r in todo[:5]:
            p = plans[r["id"]]
            print("\n" + "=" * 70)
            print(f"[{p['gen'].name} | mode={p['mode']} style={p['style']} persona={p['persona']!r} temp={p['temperature']}]")
            print(build_prompt(seed=r["text"], mode=p["mode"], persona=p["persona"], style=p["style"],
                               sentiment=r.get("sentiment"), n_words=r["n_words"]))
        return

    if not todo:
        print("nothing to do")
        return
    gens_used = {plans[r["id"]]["gen"].name: plans[r["id"]]["gen"] for r in todo}
    if not args.no_preflight:
        preflight(list(gens_used.values()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    max_attempts = int(cfg.get("max_attempts", 3))
    ok = fail = 0
    start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        with args.output.open("a", encoding="utf-8") as out, args.failures.open("a", encoding="utf-8") as ferr:
            futures = {pool.submit(generate_one, r, plans[r["id"]], max_attempts): r for r in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                r = futures[fut]
                try:
                    rec = fut.result()
                    with lock:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out.flush()
                    ok += 1
                except Exception as e:  # noqa: BLE001 - log and keep going
                    with lock:
                        ferr.write(json.dumps({"pair_id": r["id"], "generator": plans[r["id"]]["gen"].name,
                                               "error": f"{type(e).__name__}: {e}"}) + "\n")
                        ferr.flush()
                    fail += 1
                if i % 100 == 0 or i == len(todo):
                    rate = i / max(time.time() - start, 1e-9)
                    print(f"  {i}/{len(todo)}  ok={ok} failed={fail}  {rate:.1f}/s")
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        print("\nstopped; finished rows are saved. Run the same command again to continue.")
        raise SystemExit(130)
    pool.shutdown()
    print(f"done: {ok} written to {args.output}, {fail} failed (see {args.failures})")
    print_usage(list(gens_used.values()), plans, todo, full_counts)


def print_usage(gens: list[Generator], plans: dict, todo: list[dict], full_counts: Counter) -> None:
    print("\ntoken usage this run (includes the preflight test call):")
    for g in gens:
        u = g.usage
        line = f"  {g.name:24s} calls={u.calls:6d}  in={u.input_tokens:9d}  out={u.output_tokens:9d}"
        cost = g.cost_usd()
        if cost is not None:
            rows = sum(1 for r in todo if plans[r["id"]]["gen"].name == g.name)
            line += f"  ~${cost:.2f}"
            if rows:
                line += f"  (~${cost / rows * full_counts[g.name]:.2f} for all {full_counts[g.name]} rows)"
        print(line)


if __name__ == "__main__":
    main()
