import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
SRC = Path("dataset/processed/narrative_dataset.jsonl")
OUT = Path("dataset/processed")

# Load and group by label
by_label = defaultdict(list)
with open(SRC) as f:
    for line in f:
        row = json.loads(line)
        by_label[row["label"]].append(row)

# Each class: 250 total → 200 train / 25 valid / 25 test
train, valid, test = [], [], []
for label, rows in sorted(by_label.items()):
    rng = random.Random(SEED)
    rng.shuffle(rows)
    train += rows[:200]
    valid += rows[200:225]
    test  += rows[225:250]

# Shuffle splits
random.Random(SEED).shuffle(train)
random.Random(SEED).shuffle(valid)
random.Random(SEED).shuffle(test)

def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            # mlx_lm needs only the "messages" key (label is extra, harmless but clean to keep)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows → {path}")

write(OUT / "train.jsonl", train)
write(OUT / "valid.jsonl", valid)
write(OUT / "test.jsonl", test)

# Sanity check label distribution
from collections import Counter
for split_name, rows in [("train", train), ("valid", valid), ("test", test)]:
    dist = Counter(r["label"] for r in rows)
    print(f"  {split_name}: {dict(sorted(dist.items()))}")
