# Design — Companion Dataset Rework

**Date:** 2026-09-04
**Status:** Approved (brainstorm), pending spec review
**Topic:** Rebuild the LoRA fine-tuning dataset so the on-device companion stops repeating
itself, stops inventing context, stops interrogating stressed users, and understands native
Gen Z Indonesian slang.

---

## 1. Problem

The shipped adapter (`training/adapters`, iter 1200) produces low-quality chat:

- **Degenerate repetition** — emits a "safe closer" verbatim across turns
  (`"Ada yang bisa dibantu sekarang?"` ×3 in one session).
- **Invents context** — every checkpoint hallucinates details the user never gave
  (`"Hari Senin yang panjang"` when nobody mentioned Monday).
- **Interrogates** — trained to end replies with reflective questions
  (`"Mau cerita lebih?"`); ~11% of train assistant turns end in `?`.
- **Fumbles native slang** — `asu`, `jir`, `mboh`, `lebay`, `cape bat` confuse it.

### Root cause (verified by probing checkpoints iter 200/400/600/1200)

All checkpoints fail the same way → it is the **dataset**, not just hyperparameters:

1. **100% single-turn.** Every one of the 800 train samples is exactly one user turn +
   one assistant turn. The model never saw turn 3, so in a longer chat it repeats or
   invents.
2. **All user inputs are English social-media posts** from
   `dataset/raw/mental_heath_unbanlanced.csv`. The model never sees Indonesian chat
   register as *input*, so it cannot parse it.
3. **Reflective-question closers are trained in.** The old `SYSTEM_PROMPT` asked for
   "validasi perasaan + saran kecil"; many generated replies close with a soft question.
4. **Overfit config** (secondary): `scale: 10.0` (mlx-lm does not divide by rank →
   ~5× a typical `alpha/rank = 2.0` setup) × 1200 iters (~3 epochs) on 800 samples.

Config is a follow-up (see §9). This spec covers the dataset only.

---

## 2. Goals / Non-goals

### Goals

- A regenerated conversation dataset that teaches: multi-turn flow, native Gen Z ID input
  comprehension, validation-without-interrogation, no invented context.
- A rewritten generation script with an enforced style guide + automated validators.
- Updated docs/README reflecting the product framing (companion, not label-explainer).

### Non-goals

- Retraining / hyperparameter changes (separate plan — §9).
- Core ML / iOS export work (unrelated branch).
- Changing the 4-label taxonomy or the source CSV.
- A live classifier — labels are internal steering metadata only.

---

## 3. Product framing

This is a **peer companion**, not a clinical tool and no longer bound to the thesis
"narasi menjelaskan label" methodology. The 4 labels (`Suicidal`, `Depression`, `Anxiety`,
`Normal`) come from the source CSV ground truth and are kept **only** as:

- internal steering during generation (crisis vs. non-crisis reply rules), and
- dataset metadata (`label` key, kept for stratified splitting; harmless to mlx-lm).

Replies never name a label or cite research to the user. The journal grounding in
`docs/references.md` is retained as documentation of how severity categories are defined,
not as content that appears in replies.

---

## 4. Generation architecture

**Approach A — one Claude call per CSV seed, produces a whole conversation.**

```
CSV row (english post + label)
        │
        ▼
  one messages.create call  ──►  {"label", "messages": [user, asst, user, asst, ...]}
        │
        ▼
  validators (§6)  ──►  pass → conversations.jsonl
                        fail → regen (≤3×) → drop
```

Per seed, Claude is instructed to:

1. **Transcreate** the English post into an opening first-person Indonesian chat message
   in Gen Z register (slang, lowercase, typos, contractions, sometimes one word) — NOT a
   translation, a re-creation of what a real user would type.
2. Continue into a **3–6 turn** conversation (or stop at 1 turn for the single-turn
   portion), playing **both** sides.
3. Follow the style guide (§5) on every assistant turn.

Output: strict JSON, `messages` array, roles alternating `user`/`assistant` starting with
`user`.

### Rejected alternatives

- **B — two-phase iterative** (transcreate, then alternate one call per turn): more
  realistic user turns but N× cost, orchestration complexity, topic drift.
- **C — assistant-only, template user turns**: realistic user voice but manual work and
  low topic variety.

### Known risk + mitigation

Claude playing "user" tends to write too coherently. Mitigation: explicit messy-user
instructions + one few-shot conversation per label embedded in the system prompt showing
the target user register (short, messy, unpunctuated).

---

## 5. Style guide (the core change — new `SYSTEM_PROMPT`)

Persona: a peer who is also tired sometimes — not a counselor.

### Every assistant turn MUST

- Validate + normalize the feeling; acknowledge it is heavy / real.
- Be 1–4 sentences.
- Use heavy Gen Z ID↔EN code-switch (same register as the old dataset —
  `"genuinely exhausting"`, `"valid banget"`, `"overwhelmed"`).
- Engage with the specific detail the user gave, not a template.

### Every assistant turn MUST NOT

- **Contain a question mark. At all.** No `"mau cerita lebih?"`, no reflective questions.
  Conversation continues only if the user continues.
- Use toxic positivity / empty filler: `"semoga besok lebih ringan"`, `"semangat ya"`,
  `"pasti bisa"`, `"semoga cepat membaik"`.
- Diagnose or use clinical terms toward the user (`"itu tanda depresi"`, `"kayaknya GAD"`).
- Give unsolicited advice/solutions (`"coba journaling"`, `"olahraga aja"`) unless the
  user explicitly asks for suggestions.
- Invent context the user did not state (no `"hari Senin ya"` unless the user said so).

### Crisis rules (label `Suicidal`)

- Mention the hotline **Into The Light Indonesia — 119 ext 8** only when the user names
  **explicit intent, plan, or method**.
- Vague passive ideation (`"pengen ilang"`, `"capek hidup"`) → validate first, name it
  gently, do **not** immediately refer.
- **Safety net:** if a `Suicidal` conversation runs 3+ turns without improvement, exactly
  one gentle hotline mention still appears. "Validate first" never means "never refer."
- Clear risk signals always get the hotline in the same turn.

### User-turn generation rules

- Short, lowercase, slang, contractions, typos; sometimes a single word (`"iya"`,
  `"mboh"`, `"gatau"`).
- Emotionally messy, not articulate. May not directly answer the previous assistant turn.
- Mix of Jakartan and lightly regional slang (`asu`, `jir`, `anjir`, `mager`, `gabut`,
  `cape bat`, `mboh gaada`).

---

## 6. Validators (in-script, before writing each sample)

A sample is regenerated (≤3 attempts) then dropped if any check fails:

| Check | Rule |
|---|---|
| No interrogation | zero `?` characters in any `assistant` turn |
| Turn count | matches requested (1 for single-turn; 3–6 for multi-turn) |
| Role order | strictly alternating, starts `user`, ends `assistant` |
| Reply length | each assistant turn ≤ 4 sentences and ≥ 15 chars |
| Language | assistant turns majority-Indonesian (heuristic: ID stopword ratio) |
| No dupes | near-exact duplicate assistant turns within a conversation rejected |
| Crisis gate | if `label == Suicidal` and any assistant turn names a method/plan cue,
  a hotline mention must be present |
| JSON | parses; only `role`/`content` keys; roles in {user, assistant} |

Counts of drops/regens per reason are printed at the end for a quality read.

---

## 7. Dataset composition

| Item | Value |
|---|---|
| Total conversations | 1200 (300 per label, stratified from the 49,612-row CSV) |
| Multi-turn (3–6 turns) | ~70% (840) |
| Single-turn (opener only) | ~30% (360) |
| Split | 960 train / 120 valid / 120 test |
| Split stratification | by `label` **and** turn-type (single/multi) |
| Schema | `{"label": str, "messages": [{"role","content"}, ...]}` (mlx-lm compatible) |
| Seed filtering | drop CSV rows with < 3 words before sampling |
| Sampling seed | fixed (reproducible) |

Cost: ~1200 Sonnet calls, ~800 output tokens each — negligible.

---

## 8. File changes

| File | Change |
|---|---|
| `scripts/generate_narrative.py` → `scripts/generate_dataset.py` | Rename. Rewrite `SYSTEM_PROMPT` (§5). Emit a full conversation per call. Add few-shot (one good multi-turn example per label). New flags: `--multi-turn-ratio`, `--turns-min`, `--turns-max`, `--n-per-class`, `--limit`, `--dry-run`. Keep retry/backoff + `ANTHROPIC_WORKSPACE_ID` handling. Add validators (§6). |
| `scripts/split_dataset.py` | Add turn-type stratification. Point at `conversations.jsonl`. Otherwise unchanged. |
| `dataset/processed/narrative_dataset.jsonl` | Regenerate as `dataset/processed/conversations.jsonl`. Archive old file as `narrative_dataset.jsonl.bak`. |
| `dataset/processed/{train,valid,test}.jsonl` | Regenerated by `split_dataset.py`. Old ones archived `.bak`. |
| `docs/references.md` | Reframe: journals document severity-category grounding, not reply content. Remove "narasi menjelaskan label" language. |
| `README.md` | Update "Pendekatan" + "Struktur": companion framing, label = internal steering, new script name, multi-turn + native-input notes. |
| `training/lora_config.yaml` | `data:` path stays `dataset/processed`. No hyperparameter change in this plan (see §9). |

---

## 9. Follow-up (separate plan, after dataset lands)

Retrain tuning — not in scope here, tracked for next:

- `scale: 10.0 → 2.0`
- `iters: 1200 → ~500`, rely on `steps_per_eval` + pick the checkpoint at min valid loss
  (stop-early by hand; mlx-lm has no auto early-stop)
- try `rank: 8 → 16`
- keep `mask_prompt: true`, `max_seq_length: 1024` (bump only if multi-turn p99 exceeds it)
- re-probe with the multi-turn script used during this brainstorm before shipping

---

## 10. Success criteria

- `conversations.jsonl` has 1200 rows, ~70% multi-turn, balanced 4-label.
- **Zero** `?` in any assistant turn across the whole dataset.
- Manual read of 30 random conversations: opening messages read like real Gen Z ID chat;
  assistant replies validate without advising/interrogating/inventing; crisis rules
  visibly followed.
- `train/valid/test.jsonl` regenerated, stratified, no leakage (dedupe seeds across
  splits).
- `docs/references.md` + `README.md` no longer describe a label-explainer dataset.
- A retrain on the new data (follow-up) re-probed with the multi-turn harness shows no
  verbatim repetition and no invented context across a 4-turn venting conversation.
