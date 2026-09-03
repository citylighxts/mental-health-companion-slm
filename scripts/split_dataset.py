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
    """Stratified split that assigns identical openings as a unit.

    Rows sharing an opening user message form one group; a group is assigned to a
    single split, so a duplicated opening can never leak across splits. The stratum
    key is the (label, turn_type) of the group's first row.
    """
    by_opening = defaultdict(list)
    for r in rows:
        by_opening[_opening(r)].append(r)

    by_stratum = defaultdict(list)
    for _, group in sorted(by_opening.items()):  # sorted -> input-order independent
        by_stratum[(group[0]["label"], turn_type(group[0]))].append(group)

    train, valid, test = [], [], []
    for _, groups in sorted(by_stratum.items()):
        rng = random.Random(seed)
        rng.shuffle(groups)
        n = len(groups)
        n_train = round(n * ratios[0])
        n_valid = round(n * ratios[1])
        for g in groups[:n_train]:
            train += g
        for g in groups[n_train:n_train + n_valid]:
            valid += g
        for g in groups[n_train + n_valid:]:
            test += g

    # now an invariant, not a guard: grouping above makes a leak impossible
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
