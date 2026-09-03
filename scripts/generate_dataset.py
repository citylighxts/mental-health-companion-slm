"""Generate the companion fine-tuning dataset.

One Claude call per sampled CSV row -> a whole 1-6 turn Indonesian Gen Z
conversation, validated against scripts/dataset_validators.py before it is kept.

Run a small check first:
    python scripts/generate_dataset.py --limit 12 --out dataset/processed/_smoke.jsonl
Then the full run:
    python scripts/generate_dataset.py --n-per-class 300
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import anthropic

from companion_prompt import SYSTEM_PROMPT, build_user_prompt
from dataset_validators import VALID_LABELS, validate_conversation

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "../dataset/raw/mental_heath_unbanlanced.csv"
DEFAULT_OUT = SCRIPT_DIR / "../dataset/processed/conversations.jsonl"
DEFAULT_MODEL = "claude-sonnet-5"  # --model claude-opus-5 for higher quality


def load_seeds(csv_path, n_per_class: int, *, seed: int, min_words: int = 3) -> list[dict]:
    by_label: dict[str, list[str]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            label = (row.get("status") or "").strip()
            if label in VALID_LABELS and len(text.split()) >= min_words:
                by_label[label].append(text)

    rng = random.Random(seed)
    seeds: list[dict] = []
    for label in sorted(VALID_LABELS):
        pool = sorted(set(by_label[label]))
        rng.shuffle(pool)
        if len(pool) < n_per_class:
            raise ValueError(f"only {len(pool)} usable seeds for {label}, need {n_per_class}")
        seeds.extend({"text": t, "label": label} for t in pool[:n_per_class])
    rng.shuffle(seeds)
    return seeds


def assign_modes(seeds, *, multi_turn_ratio: float, turns_min: int, turns_max: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for s in seeds:
        single = rng.random() >= multi_turn_ratio
        out.append({
            **s,
            "single_turn": single,
            "target_turns": 1 if single else rng.randint(turns_min, turns_max),
        })
    return out


def parse_response(text: str) -> list[dict]:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as e:
        raise ValueError(f"not JSON: {e}") from e
    msgs = obj.get("messages") if isinstance(obj, dict) else None
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("no non-empty 'messages' list")
    return msgs


def _call(client, model: str, prompt: str) -> str:
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        except anthropic.RateLimitError:
            time.sleep(5 * (attempt + 1))
        except anthropic.APIStatusError as e:
            if getattr(e, "status_code", 0) >= 500 and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("rate limited past retry budget")


def generate_one(client, model: str, seed_row: dict, *, turns_min: int, turns_max: int,
                 max_attempts: int = 3) -> dict | None:
    prompt = build_user_prompt(
        seed_row["text"], seed_row["label"],
        single_turn=seed_row["single_turn"], target_turns=seed_row["target_turns"],
    )
    for _ in range(max_attempts):
        raw = _call(client, model, prompt)
        try:
            messages = parse_response(raw)
        except ValueError:
            continue
        candidate = {"label": seed_row["label"], "messages": messages}
        reasons = validate_conversation(
            candidate, single_turn=seed_row["single_turn"],
            turns_min=turns_min, turns_max=turns_max,
        )
        if not reasons:
            return candidate
        generate_one.last_reasons = reasons  # for stats
    return None


generate_one.last_reasons = []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--multi-turn-ratio", type=float, default=0.7)
    ap.add_argument("--turns-min", type=int, default=3)
    ap.add_argument("--turns-max", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="cap total seeds (smoke runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="sample + assign modes, no API calls")
    args = ap.parse_args()

    seeds = load_seeds(args.csv, args.n_per_class, seed=args.seed)
    seeds = assign_modes(
        seeds, multi_turn_ratio=args.multi_turn_ratio,
        turns_min=args.turns_min, turns_max=args.turns_max, seed=args.seed,
    )
    if args.limit:
        seeds = seeds[: args.limit]

    print(f"{len(seeds)} seeds | multi-turn ratio "
          f"{sum(not s['single_turn'] for s in seeds) / len(seeds):.2f}")
    print(Counter(s["label"] for s in seeds))
    if args.dry_run:
        for s in seeds[:3]:
            print(s)
        return

    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    client = anthropic.Anthropic(
        default_headers={"anthropic-workspace-id": workspace_id} if workspace_id else {}
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept, dropped = 0, 0
    drop_reasons: Counter = Counter()
    with open(out_path, "w") as f:
        for i, seed_row in enumerate(seeds, 1):
            result = generate_one(
                client, args.model, seed_row,
                turns_min=args.turns_min, turns_max=args.turns_max,
            )
            if result is None:
                dropped += 1
                drop_reasons.update(generate_one.last_reasons or ["parse_or_unknown"])
            else:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
            if i % 25 == 0 or i == len(seeds):
                print(f"  [{i}/{len(seeds)}] kept={kept} dropped={dropped}")

    print(f"\nWrote {kept} conversations -> {out_path}  (dropped {dropped})")
    if drop_reasons:
        print("drop reasons:", dict(drop_reasons))


if __name__ == "__main__":
    main()
