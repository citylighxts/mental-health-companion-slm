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

from companion_prompt import SYSTEM_PROMPT_CACHED, build_user_prompt
from dataset_validators import (
    VALID_LABELS, _normalize, assistant_turns, validate_conversation,
)

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
                # One explicit breakpoint on the stable system text (prompt + few-shot
                # block). It is byte-identical every call, so after the first write the
                # whole ~2k-token prefix is a cache read for the rest of the run. The
                # per-seed user message stays uncached (it varies) — which is correct.
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT_CACHED,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.APITimeoutError):
            # transient: a network blip must not abort a 1200-seed batch
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
        except anthropic.APIStatusError as e:
            if getattr(e, "status_code", 0) >= 500 and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    # unreachable: attempt 3 always returns or re-raises above. Kept as a
    # belt-and-braces guard against future edits to the retry logic.
    raise RuntimeError("retry budget exhausted")


def generate_one(client, model: str, seed_row: dict, *, turns_min: int, turns_max: int,
                 max_attempts: int = 3) -> tuple[dict | None, list[str]]:
    """Return `(conversation, [])` on success, `(None, reasons)` once attempts run out.

    The reasons come from the final attempt — returning them (rather than stashing
    them on a function attribute) keeps the caller's tally from carrying stale
    reasons over from a previous seed.
    """
    prompt = build_user_prompt(
        seed_row["text"], seed_row["label"],
        single_turn=seed_row["single_turn"], target_turns=seed_row["target_turns"],
    )
    last_reasons: list[str] = []
    for _ in range(max_attempts):
        raw = _call(client, model, prompt)
        try:
            messages = parse_response(raw)
        except ValueError:
            last_reasons = ["parse_error"]
            continue
        candidate = {"label": seed_row["label"], "messages": messages}
        reasons = validate_conversation(
            candidate, single_turn=seed_row["single_turn"],
            turns_min=turns_min, turns_max=turns_max,
        )
        if not reasons:
            return candidate, []
        last_reasons = reasons
    return None, last_reasons or ["parse_error"]


def sidecar_path(out_path) -> Path:
    """Progress file next to the output: one JSON-encoded seed text per processed seed."""
    return Path(str(out_path) + ".seeds")


def already_done(sidecar) -> set[str]:
    """Seed texts a previous (interrupted) run already processed — kept or dropped."""
    p = Path(sidecar)
    if not p.exists():
        return set()
    done: set[str] = set()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line))
            except json.JSONDecodeError:
                done.add(line)  # tolerate a hand-edited / half-written line
    return done


def report_repeated_replies(out_path, *, top: int = 10) -> None:
    """Corpus-level duplicate check — a safe-closer collapse hides from per-convo checks."""
    counts: Counter = Counter()
    total = 0
    with open(out_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                turns = assistant_turns(row["messages"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue  # a hard kill mid-write can leave a truncated final line
            total += 1
            counts.update(_normalize(t) for t in turns)
    if not counts:
        return
    print("\ntop repeated assistant turns (watch for a safe-closer collapse):")
    for text, n in counts.most_common(top):
        share = f"{n / total:.1%}" if total else "-"
        print(f"  {n:4d}  ({share} of convos)  {text[:90]}")


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
    sidecar = sidecar_path(out_path)
    done = already_done(sidecar)
    # Resume only when a sidecar survives from an interrupted run; a fresh run starts
    # the output file clean so a re-run never silently doubles the dataset.
    mode = "a" if done else "w"
    if done:
        print(f"resuming: {len(done)} seeds already processed (from {sidecar.name})")

    kept, dropped, skipped = 0, 0, 0
    drop_reasons: Counter = Counter()
    with open(out_path, mode) as f, open(sidecar, "a") as sf:
        for i, seed_row in enumerate(seeds, 1):
            if seed_row["text"] in done:
                skipped += 1
                continue
            try:
                result, reasons = generate_one(
                    client, args.model, seed_row,
                    turns_min=args.turns_min, turns_max=args.turns_max,
                )
            except Exception as e:  # one bad seed must never kill the batch
                print(f"  ! seed {i} failed: {type(e).__name__}: {e}")
                dropped += 1
                drop_reasons[f"exception:{type(e).__name__}"] += 1
                result = None
            else:
                if result is None:
                    dropped += 1
                    drop_reasons.update(reasons or ["parse_or_unknown"])
                else:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    kept += 1
            sf.write(json.dumps(seed_row["text"], ensure_ascii=False) + "\n")
            sf.flush()
            if i % 25 == 0 or i == len(seeds):
                print(f"  [{i}/{len(seeds)}] kept={kept} dropped={dropped} skipped={skipped}")

    # clean completion: the loop finished, so the progress file has done its job
    if sidecar.exists():
        os.remove(sidecar)

    print(f"\nWrote {kept} conversations -> {out_path}  (dropped {dropped}, skipped {skipped})")
    if drop_reasons:
        print("drop reasons:", dict(drop_reasons))
    report_repeated_replies(out_path)


if __name__ == "__main__":
    main()
