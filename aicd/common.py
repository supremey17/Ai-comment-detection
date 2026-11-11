"""Small shared helpers: deterministic randomness and JSONL I/O."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Iterator


def stable_rng(*parts: object) -> random.Random:
    """A Random seeded from the given parts, identical on every run and machine.

    Used so that split assignment, generator choice, persona, etc. never change
    between runs, which keeps an interrupted generation run resumable.
    """
    key = "|".join(str(p) for p in parts).encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return random.Random(seed)


def weighted_choice(rng: random.Random, items: list[dict], weight_key: str = "weight") -> dict:
    weights = [float(it.get(weight_key, 1.0)) for it in items]
    return rng.choices(items, weights=weights, k=1)[0]


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
