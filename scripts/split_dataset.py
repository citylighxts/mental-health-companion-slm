"""Stratified train/valid/test split of dataset/processed/conversations.jsonl.

Stratifies by (label, turn-type) so multi-turn coverage is even across splits,
and refuses to run if the same opening user message would land in two splits.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
SRC = Path("dataset/processed/conversations.jsonl")
OUT = Path("dataset/processed")
RATIOS = (0.8, 0.1, 0.1)


def turn_type(row: dict) -> str:
    n_user = sum(1 for m in row["messages"] if m["role"] == "user")
    return "single" if n_user == 1 else "multi"


def _opening(row: dict) -> str:
    return row["messages"][0]["content"]


def split_conversations(rows, *, seed: int = SEED, ratios=RATIOS):
    by_stratum = defaultdict(list)
    for r in rows:
        by_stratum[(r["label"], turn_type(r))].append(r)

    train, valid, test = [], [], []
    for _, items in sorted(by_stratum.items()):
        rng = random.Random(seed)
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * ratios[0])
        n_valid = round(n * ratios[1])
        train += items[:n_train]
        valid += items[n_train:n_train + n_valid]
        test += items[n_train + n_valid:]

    for a, b, name in [(train, valid, "train/valid"), (train, test, "train/test"), (valid, test, "valid/test")]:
        overlap = {_opening(r) for r in a} & {_opening(r) for r in b}
        if overlap:
            raise ValueError(f"opening-message leak across {name}: {list(overlap)[:3]}")

    for split in (train, valid, test):
        random.Random(seed).shuffle(split)
    return train, valid, test


def main() -> None:
    rows = [json.loads(line) for line in open(SRC)]
    train, valid, test = split_conversations(rows)

    for path, split in [(OUT / "train.jsonl", train), (OUT / "valid.jsonl", valid), (OUT / "test.jsonl", test)]:
        with open(path, "w") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dist = defaultdict(int)
        for r in split:
            dist[(r["label"], turn_type(r))] += 1
        print(f"{path.name}: {len(split)} rows  {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
