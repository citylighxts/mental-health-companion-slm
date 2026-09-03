# Companion Dataset Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the LoRA fine-tuning dataset generator so the on-device companion learns multi-turn flow, understands native Gen Z Indonesian chat input, validates without interrogating, and never invents context.

**Architecture:** A rewritten `scripts/generate_dataset.py` makes one Claude call per CSV seed row and gets back a whole 1–6 turn conversation as JSON. Pure-function modules `scripts/companion_prompt.py` (system prompt + few-shot) and `scripts/dataset_validators.py` (structural + content checks) are unit-tested with pytest and have no API dependency. `scripts/split_dataset.py` gains turn-type stratification. The 1200-conversation generation run itself is a documented manual step (costs API credits).

**Tech Stack:** Python 3.14 (`.venv`), `anthropic` SDK 1.2.0, `pytest` (new dev dependency), `mlx-lm` (downstream, unchanged here).

## Global Constraints

- Target model default: `claude-sonnet-5` (override via `--model`; `claude-opus-5` documented as the higher-quality option). Use the exact ID string, no date suffix.
- Anthropic SDK call shape: `client.messages.create(model=, max_tokens=, system=, messages=[...])`. No `thinking`, no prefill, no `output_config`.
- Output schema per conversation: `{"label": str, "messages": [{"role": "user"|"assistant", "content": str}, ...]}` — roles strictly alternating, first `user`, last `assistant`. mlx-lm reads only `messages`; `label` is kept for stratified splitting.
- Every assistant turn in every generated conversation: **zero `?` characters**, 1–4 sentences, ≥ 15 chars, majority Indonesian, heavy ID↔EN code-switch.
- Labels are internal steering + split metadata only. Replies never name a label or cite research.
- Crisis (`label == "Suicidal"`): hotline **"Into The Light Indonesia — 119 ext 8"** appears only in a turn responding to explicit intent/plan/method; vague passive ideation is validated without referral; a Suicidal conversation of 3+ turns with no improvement gets exactly one gentle hotline mention.
- Python env: all commands run after `source .venv/bin/activate` from the repo root.
- Commit after every task. Do NOT add a Claude co-author trailer (repo rule in `CLAUDE.md`).
- Reproducibility: every sampling / mode-assignment step takes an explicit integer seed.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `pytest.ini` | pytest config: `pythonpath = scripts`, `testpaths = tests` | Create |
| `tests/conftest.py` | shared test fixtures / fake Anthropic client | Create |
| `tests/test_dataset_validators.py` | unit tests for every validator function | Create |
| `tests/test_companion_prompt.py` | assert prompt invariants; few-shot examples pass validators | Create |
| `tests/test_generate_dataset.py` | unit tests for seed loading, mode assignment, response parsing, retry loop (mocked client) | Create |
| `tests/test_split_dataset.py` | unit tests for turn-type stratified split + leakage guard | Create |
| `scripts/dataset_validators.py` | pure validation functions + `validate_conversation` composer | Create |
| `scripts/companion_prompt.py` | `SYSTEM_PROMPT`, `FEWSHOT`, `build_user_prompt()` | Create |
| `scripts/generate_dataset.py` | CLI: sample seeds → assign modes → call Claude → validate/retry → write `conversations.jsonl` | Rename from `generate_narrative.py` + rewrite |
| `scripts/split_dataset.py` | stratified train/valid/test split by (label, turn-type) + leakage assert | Modify |
| `dataset/processed/conversations.jsonl` | generated dataset (replaces `narrative_dataset.jsonl`) | Generated artifact |
| `dataset/processed/{train,valid,test}.jsonl` | regenerated splits | Generated artifact |
| `dataset/processed/*.jsonl.bak` | archived old single-turn dataset + splits | Create (git mv) |
| `docs/references.md` | reframe: journals = severity grounding, not reply content | Modify |
| `README.md` | companion framing, new script name, multi-turn + native-input notes | Modify |
| `requirements.txt` | add `pytest` to the `.venv` section | Modify |

---

## Task 1: pytest scaffold + structural validators

**Files:**
- Create: `pytest.ini`
- Create: `tests/conftest.py`
- Create: `scripts/dataset_validators.py`
- Create: `tests/test_dataset_validators.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `dataset_validators.assistant_turns(messages: list[dict]) -> list[str]`
  - `dataset_validators.user_turns(messages: list[dict]) -> list[str]`
  - `dataset_validators.schema_errors(obj) -> list[str]`
  - `dataset_validators.roles_alternate(messages: list[dict]) -> bool`
  - `dataset_validators.turn_count_ok(messages: list[dict], *, single_turn: bool, turns_min: int, turns_max: int) -> bool`
  - `dataset_validators.no_questions(messages: list[dict]) -> bool`
  - `dataset_validators.reply_length_ok(messages: list[dict], *, max_sentences: int = 4, min_chars: int = 15) -> bool`
  - `dataset_validators.no_repeated_replies(messages: list[dict], *, similarity: float = 0.9) -> bool`

- [ ] **Step 1: Install pytest and pin it**

Run:
```bash
source .venv/bin/activate && pip install pytest && pip show pytest | grep -i version
```
Expected: a version prints (pytest ≥ 8).

Then edit `requirements.txt` — in the `# --- training / general (.venv, Python 3.14) ---` block, add a line after `huggingface_hub`:
```
pytest            # dev: dataset validator + generator unit tests
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
pythonpath = scripts
testpaths = tests
python_files = test_*.py
```

- [ ] **Step 3: Create `tests/conftest.py` with the fake Anthropic client (used in Task 4 too)**

```python
"""Shared test helpers."""
from types import SimpleNamespace


class FakeMessages:
    """Stand-in for client.messages — pops canned text responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeMessages ran out of canned responses")
        text = self._responses.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)
```

- [ ] **Step 4: Write the failing structural-validator tests**

Create `tests/test_dataset_validators.py`:

```python
import dataset_validators as dv

GOOD_MULTI = [
    {"role": "user", "content": "anjir capek bat hari ini"},
    {"role": "assistant", "content": "Hari yang nguras abis itu berat banget. Wajar kalau kamu ngerasa gitu."},
    {"role": "user", "content": "iya pengen rebahan aja"},
    {"role": "assistant", "content": "Rebahan dulu aja, badan kamu jelas minta istirahat. Nggak harus produktif sekarang."},
]
GOOD_SINGLE = [
    {"role": "user", "content": "halo lagi gabut"},
    {"role": "assistant", "content": "Halo, gabut sore-sore itu relatable banget. Santai aja dulu."},
]


def test_assistant_and_user_turns_split():
    assert dv.assistant_turns(GOOD_MULTI) == [
        "Hari yang nguras abis itu berat banget. Wajar kalau kamu ngerasa gitu.",
        "Rebahan dulu aja, badan kamu jelas minta istirahat. Nggak harus produktif sekarang.",
    ]
    assert dv.user_turns(GOOD_MULTI)[0] == "anjir capek bat hari ini"


def test_schema_errors_accepts_good():
    assert dv.schema_errors({"label": "Normal", "messages": GOOD_SINGLE}) == []


def test_schema_errors_flags_bad_role_and_keys():
    errs = dv.schema_errors({"label": "Normal", "messages": [
        {"role": "system", "content": "x"},
        {"role": "assistant", "content": "y", "extra": 1},
    ]})
    assert errs  # at least one error string


def test_roles_alternate():
    assert dv.roles_alternate(GOOD_MULTI) is True
    assert dv.roles_alternate([
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]) is False
    assert dv.roles_alternate([
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
    ]) is False  # must start with user


def test_turn_count_ok():
    assert dv.turn_count_ok(GOOD_SINGLE, single_turn=True, turns_min=3, turns_max=6) is True
    assert dv.turn_count_ok(GOOD_MULTI, single_turn=True, turns_min=3, turns_max=6) is False
    assert dv.turn_count_ok(GOOD_MULTI, single_turn=False, turns_min=3, turns_max=6) is False  # only 2 asst turns
    big = GOOD_MULTI + [
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "Makasih udah cerita ya. Aku di sini."},
    ]
    assert dv.turn_count_ok(big, single_turn=False, turns_min=3, turns_max=6) is True


def test_no_questions():
    assert dv.no_questions(GOOD_MULTI) is True
    assert dv.no_questions([
        {"role": "user", "content": "capek"},
        {"role": "assistant", "content": "Capek ya. Mau cerita lebih?"},
    ]) is False


def test_reply_length_ok():
    assert dv.reply_length_ok(GOOD_MULTI) is True
    assert dv.reply_length_ok([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]) is False  # too short
    five = {"role": "assistant", "content": "Satu. Dua. Tiga. Empat. Lima."}
    assert dv.reply_length_ok([{"role": "user", "content": "x"}, five]) is False


def test_no_repeated_replies():
    assert dv.no_repeated_replies(GOOD_MULTI) is True
    dup = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "Capek itu valid banget, aku dengerin."},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "capek itu valid banget aku dengerin"},
    ]
    assert dv.no_repeated_replies(dup) is False
```

- [ ] **Step 5: Run the tests, verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_dataset_validators.py -q`
Expected: FAIL / errors — `ModuleNotFoundError: No module named 'dataset_validators'`.

- [ ] **Step 6: Implement `scripts/dataset_validators.py` (structural half)**

```python
"""Pure validators for generated companion conversations. No API / network."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

VALID_ROLES = {"user", "assistant"}
VALID_MSG_KEYS = {"role", "content"}
VALID_LABELS = {"Suicidal", "Depression", "Anxiety", "Normal"}


def assistant_turns(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m.get("role") == "assistant"]


def user_turns(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m.get("role") == "user"]


def schema_errors(obj) -> list[str]:
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["not a dict"]
    if obj.get("label") not in VALID_LABELS:
        errs.append(f"bad label: {obj.get('label')!r}")
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return errs + ["messages missing or empty"]
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            errs.append(f"msg {i} not a dict")
            continue
        if set(m.keys()) != VALID_MSG_KEYS:
            errs.append(f"msg {i} keys {sorted(m.keys())}")
        if m.get("role") not in VALID_ROLES:
            errs.append(f"msg {i} role {m.get('role')!r}")
        if not isinstance(m.get("content"), str) or not m.get("content", "").strip():
            errs.append(f"msg {i} empty content")
    return errs


def roles_alternate(messages: list[dict]) -> bool:
    if not messages or messages[0].get("role") != "user":
        return False
    if messages[-1].get("role") != "assistant":
        return False
    expected = "user"
    for m in messages:
        if m.get("role") != expected:
            return False
        expected = "assistant" if expected == "user" else "user"
    return True


def turn_count_ok(messages: list[dict], *, single_turn: bool, turns_min: int, turns_max: int) -> bool:
    n_asst = len(assistant_turns(messages))
    n_user = len(user_turns(messages))
    if n_asst != n_user:
        return False
    if single_turn:
        return n_asst == 1
    return turns_min <= n_asst <= turns_max


_SENT_SPLIT = re.compile(r"[.!]+")


def no_questions(messages: list[dict]) -> bool:
    return all("?" not in t for t in assistant_turns(messages))


def reply_length_ok(messages: list[dict], *, max_sentences: int = 4, min_chars: int = 15) -> bool:
    for t in assistant_turns(messages):
        if len(t.strip()) < min_chars:
            return False
        sentences = [s for s in _SENT_SPLIT.split(t) if s.strip()]
        if len(sentences) > max_sentences:
            return False
    return True


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def no_repeated_replies(messages: list[dict], *, similarity: float = 0.9) -> bool:
    norms = [_normalize(t) for t in assistant_turns(messages)]
    for i in range(len(norms)):
        for j in range(i + 1, len(norms)):
            if not norms[i] or not norms[j]:
                continue
            if norms[i] == norms[j]:
                return False
            if SequenceMatcher(None, norms[i], norms[j]).ratio() >= similarity:
                return False
    return True
```

- [ ] **Step 7: Run the tests, verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_dataset_validators.py -q`
Expected: PASS (9 tests).

- [ ] **Step 8: Commit**

```bash
git add pytest.ini tests/conftest.py tests/test_dataset_validators.py scripts/dataset_validators.py requirements.txt
git commit -m "test: pytest scaffold + structural conversation validators"
```

---

## Task 2: Content validators + `validate_conversation` composer

**Files:**
- Modify: `scripts/dataset_validators.py`
- Modify: `tests/test_dataset_validators.py`

**Interfaces:**
- Consumes: everything from Task 1.
- Produces:
  - `dataset_validators.is_majority_indonesian(text: str, *, threshold: float = 0.12) -> bool`
  - `dataset_validators.mentions_crisis_cue(text: str) -> bool`
  - `dataset_validators.mentions_hotline(text: str) -> bool`
  - `dataset_validators.crisis_gate_ok(label: str, messages: list[dict]) -> bool`
  - `dataset_validators.validate_conversation(obj: dict, *, single_turn: bool, turns_min: int = 3, turns_max: int = 6) -> list[str]` — returns failure-reason strings; empty list means valid.

- [ ] **Step 1: Write failing content-validator tests (append to `tests/test_dataset_validators.py`)**

```python
def test_is_majority_indonesian():
    assert dv.is_majority_indonesian("Hari ini berat banget ya, tapi kamu udah ngejalanin sejauh ini.") is True
    assert dv.is_majority_indonesian("This is a fully english sentence with no indonesian words at all.") is False


def test_crisis_cue_and_hotline_detection():
    assert dv.mentions_crisis_cue("tadi malem mikirin caranya buat ngilangin diri") is True
    assert dv.mentions_crisis_cue("capek banget hari ini") is False
    assert dv.mentions_hotline("hubungi Into The Light Indonesia 119 ext 8 ya") is True
    assert dv.mentions_hotline("aku di sini kok") is False


def test_crisis_gate_requires_hotline_after_explicit_cue():
    convo_ok = [
        {"role": "user", "content": "pengen berhenti aja rasanya"},
        {"role": "assistant", "content": "Pengen berhenti itu sinyal kamu udah nahan beban berat kelamaan. Aku dengerin."},
        {"role": "user", "content": "tadi malem sempet mikirin caranya buat ngilang"},
        {"role": "assistant", "content": "Aku khawatir banget sama kamu. Tolong ngomong sama Into The Light Indonesia di 119 ext 8, mereka ada 24 jam."},
    ]
    assert dv.crisis_gate_ok("Suicidal", convo_ok) is True

    convo_bad = [
        {"role": "user", "content": "tadi malem sempet mikirin caranya buat bunuh diri"},
        {"role": "assistant", "content": "Berat banget ya. Aku dengerin kok, nggak ke mana-mana."},
    ]
    assert dv.crisis_gate_ok("Suicidal", convo_bad) is False

    # vague ideation, no cue -> hotline NOT required
    convo_vague = [
        {"role": "user", "content": "kadang pengen ilang aja"},
        {"role": "assistant", "content": "Rasa pengen ilang itu berat banget dibawa sendirian. Kamu nggak sendirian ngadepin ini."},
    ]
    assert dv.crisis_gate_ok("Suicidal", convo_vague) is True
    assert dv.crisis_gate_ok("Normal", convo_vague) is True  # gate only applies to Suicidal


def test_validate_conversation_composes_all_checks():
    good = {"label": "Anxiety", "messages": [
        {"role": "user", "content": "besok interview gua panik parah"},
        {"role": "assistant", "content": "Malam sebelum interview emang bikin kepala muter terus. Wajar kamu susah tenang."},
        {"role": "user", "content": "takut blank"},
        {"role": "assistant", "content": "Takut blank itu manusiawi. Kamu udah sampe tahap ini bukan karena kebetulan."},
        {"role": "user", "content": "makasih"},
        {"role": "assistant", "content": "Sama-sama. Kamu nggak harus nyelesain semua kekhawatiran itu malam ini."},
    ]}
    assert dv.validate_conversation(good, single_turn=False) == []

    bad = {"label": "Anxiety", "messages": [
        {"role": "user", "content": "besok interview"},
        {"role": "assistant", "content": "Gimana perasaan kamu sekarang?"},
    ]}
    reasons = dv.validate_conversation(bad, single_turn=False)
    assert any("question" in r for r in reasons)
    assert any("turn_count" in r for r in reasons)
```

- [ ] **Step 2: Run, verify fail**

Run: `source .venv/bin/activate && pytest tests/test_dataset_validators.py -q`
Expected: FAIL — `AttributeError: module 'dataset_validators' has no attribute 'is_majority_indonesian'`.

- [ ] **Step 3: Implement the content half (append to `scripts/dataset_validators.py`)**

```python
# --- content checks ---

_ID_MARKERS = {
    "yang", "nggak", "ga", "gak", "aku", "kamu", "banget", "aja", "udah", "lagi",
    "tuh", "sih", "kok", "deh", "ya", "di", "ke", "itu", "ini", "dari", "buat",
    "sama", "juga", "kalau", "kalo", "biar", "emang", "masih", "terus", "gitu",
    "nggak", "harus", "bukan", "kayak", "lah", "dong", "nih", "kan",
}

_CRISIS_CUE_PATTERNS = [
    r"bunuh diri", r"pengen mati", r"pengin mati", r"mau mati", r"pengen mati aja",
    r"akhiri hidup", r"ngakhirin hidup", r"ngilangin diri", r"ngilangin nyawa",
    r"gantung diri", r"loncat dari", r"lompat dari", r"nyilet", r"nyayat",
    r"overdosis", r"minum obat banyak", r"nenggak obat", r"cara(nya)?\s+(buat\s+)?(mati|bunuh diri|ngilang)",
    r"kill myself", r"end my life", r"end it all", r"\bsuicide\b", r"take my own life",
]
_HOTLINE_MARKERS = ["into the light", "119 ext 8", "119 ekstensi 8", "119 ext.8", "119 ext8"]


def is_majority_indonesian(text: str, *, threshold: float = 0.12) -> bool:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in _ID_MARKERS)
    return hits >= 2 or (hits / len(tokens)) >= threshold


def mentions_crisis_cue(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in _CRISIS_CUE_PATTERNS)


def mentions_hotline(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _HOTLINE_MARKERS)


def crisis_gate_ok(label: str, messages: list[dict]) -> bool:
    if label != "Suicidal":
        return True
    convo_has_hotline = any(mentions_hotline(t) for t in assistant_turns(messages))
    for i, m in enumerate(messages):
        if m.get("role") != "user" or not mentions_crisis_cue(m["content"]):
            continue
        # the assistant reply to this user turn (next message) must carry the hotline,
        # OR some later assistant turn must (safety net); simplest correct rule:
        later_assistant = [
            mm["content"] for mm in messages[i + 1:] if mm.get("role") == "assistant"
        ]
        if not any(mentions_hotline(t) for t in later_assistant):
            return False
    return convo_has_hotline or not _any_user_cue(messages)


def _any_user_cue(messages: list[dict]) -> bool:
    return any(
        m.get("role") == "user" and mentions_crisis_cue(m["content"]) for m in messages
    )


def validate_conversation(
    obj: dict, *, single_turn: bool, turns_min: int = 3, turns_max: int = 6
) -> list[str]:
    reasons: list[str] = []
    se = schema_errors(obj)
    if se:
        return [f"schema: {e}" for e in se]  # can't run structural checks on bad schema

    messages = obj["messages"]
    label = obj["label"]

    if not roles_alternate(messages):
        reasons.append("roles_do_not_alternate")
    if not turn_count_ok(messages, single_turn=single_turn, turns_min=turns_min, turns_max=turns_max):
        reasons.append(f"turn_count (single_turn={single_turn})")
    if not no_questions(messages):
        reasons.append("assistant_turn_contains_question")
    if not reply_length_ok(messages):
        reasons.append("reply_length")
    if not no_repeated_replies(messages):
        reasons.append("repeated_assistant_reply")
    for t in assistant_turns(messages):
        if not is_majority_indonesian(t):
            reasons.append("assistant_turn_not_majority_indonesian")
            break
    if not crisis_gate_ok(label, messages):
        reasons.append("crisis_gate_missing_hotline")
    return reasons
```

- [ ] **Step 4: Run, verify pass**

Run: `source .venv/bin/activate && pytest tests/test_dataset_validators.py -q`
Expected: PASS (all tests, ~13).

- [ ] **Step 5: Commit**

```bash
git add scripts/dataset_validators.py tests/test_dataset_validators.py
git commit -m "test: content validators + validate_conversation composer"
```

---

## Task 3: Companion prompt module

**Files:**
- Create: `scripts/companion_prompt.py`
- Create: `tests/test_companion_prompt.py`

**Interfaces:**
- Consumes: `dataset_validators.validate_conversation`.
- Produces:
  - `companion_prompt.SYSTEM_PROMPT: str`
  - `companion_prompt.FEWSHOT: list[dict]` — each `{"label", "single_turn": bool, "messages": [...]}`
  - `companion_prompt.build_user_prompt(post: str, label: str, *, single_turn: bool, target_turns: int) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_companion_prompt.py`:

```python
import dataset_validators as dv
import companion_prompt as cp


def test_system_prompt_states_the_hard_rules():
    s = cp.SYSTEM_PROMPT.lower()
    assert "tanda tanya" in s or "tanpa pertanyaan" in s
    assert "into the light" in s
    assert "119 ext 8" in s
    assert "json" in s


def test_fewshot_examples_all_pass_validators():
    assert len(cp.FEWSHOT) == 4
    labels = {ex["label"] for ex in cp.FEWSHOT}
    assert labels == {"Suicidal", "Depression", "Anxiety", "Normal"}
    for ex in cp.FEWSHOT:
        reasons = dv.validate_conversation(
            {"label": ex["label"], "messages": ex["messages"]},
            single_turn=ex["single_turn"],
        )
        assert reasons == [], f"{ex['label']} few-shot invalid: {reasons}"


def test_build_user_prompt_includes_seed_and_mode():
    p = cp.build_user_prompt("i feel so empty lately", "Depression", single_turn=False, target_turns=4)
    assert "i feel so empty lately" in p
    assert "Depression" in p
    assert "4" in p
    single = cp.build_user_prompt("hi", "Normal", single_turn=True, target_turns=1)
    assert "1" in single
```

- [ ] **Step 2: Run, verify fail**

Run: `source .venv/bin/activate && pytest tests/test_companion_prompt.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'companion_prompt'`.

- [ ] **Step 3: Implement `scripts/companion_prompt.py`**

```python
"""System prompt, few-shot examples, and per-seed user prompt for dataset generation."""
from __future__ import annotations

SYSTEM_PROMPT = """\
Kamu bikin data latih untuk chatbot teman curhat berbahasa Indonesia gaya Gen Z.
Persona bot: teman sebaya yang empatik dan kadang capek juga — BUKAN konselor, BUKAN dokter.

Kamu dikasih satu postingan media sosial berbahasa Inggris + satu kategori kondisi
(ground truth, JANGAN diubah, JANGAN disebut ke user). Tugasmu:

1. TRANSCREATE postingan itu jadi SATU pesan chat pembuka, sudut pandang orang pertama,
   gaya anak muda Indonesia: huruf kecil, singkatan, typo wajar, slang campur
   ("anjir", "capek bat", "jir", "mager", "gabut", "mboh", "gaada abisnya"). Bukan
   terjemahan harfiah — tulis ulang jadi kayak orang beneran ngetik ke temennya.
2. Lanjutin jadi percakapan utuh. Kamu meranin DUA sisi (user dan asisten).

ATURAN TIAP BALASAN ASISTEN — WAJIB:
- Validasi + normalisasi perasaan user. Akui itu berat / nyata / masuk akal.
- 1–4 kalimat. Code-switch Indonesia–Inggris deras ala Gen Z
  ("genuinely exhausting", "valid banget", "overwhelmed", "it makes sense").
- Nyambung ke detail spesifik yang user sebut. Bukan template.

ATURAN TIAP BALASAN ASISTEN — DILARANG KERAS:
- TANDA TANYA. Sama sekali nggak boleh ada "?" di balasan asisten. Jangan nanya balik,
  termasuk "mau cerita lebih?" / "gimana perasaan kamu?". Percakapan lanjut kalau
  user-nya sendiri yang lanjut.
- Toxic positivity / basa-basi kosong: "semoga besok lebih ringan", "semangat ya",
  "pasti bisa", "semoga cepet membaik".
- Nge-diagnosa atau istilah klinis ke user ("itu tanda depresi", "kayaknya kamu GAD").
- Saran / solusi yang nggak diminta ("coba journaling", "olahraga aja"). Kalau user
  eksplisit minta saran, boleh, secukupnya.
- Ngarang konteks yang user nggak sebut (jangan bilang "hari Senin ya" kalau user
  nggak nyebut hari).

ATURAN KRISIS (kategori "Suicidal"):
- Sebut hotline "Into The Light Indonesia — 119 ext 8" HANYA di balasan yang menanggapi
  niat / rencana / metode yang eksplisit ("tadi mikirin caranya", "udah nyiapin ...").
- Kalau cuma pikiran pasif & samar ("pengen ilang", "capek hidup") → validasi dulu,
  JANGAN langsung rujuk hotline.
- Safety net: kalau percakapan "Suicidal" jalan 3+ giliran dan user nggak membaik,
  selipin SATU ajakan lembut ke hotline itu.

ATURAN GILIRAN USER (kamu yang nulis juga):
- Pendek, huruf kecil, slang, typo, kadang satu kata ("iya", "mboh", "gatau").
- Emosional & berantakan, bukan artikulatif. Boleh nggak langsung jawab balasan asisten.

OUTPUT: HANYA JSON valid, tanpa markdown fence, bentuknya:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
Mulai dari "user", selang-seling, diakhiri "assistant".
"""


FEWSHOT = [
    {
        "label": "Anxiety",
        "single_turn": False,
        "messages": [
            {"role": "user", "content": "anjir besok interview gua deg2an parah gabisa tidur"},
            {"role": "assistant", "content": "Malam sebelum interview emang paling brutal — kepala muter terus mikirin besok. Wajar banget badan kamu nolak buat rileks."},
            {"role": "user", "content": "takut blank pas ditanya"},
            {"role": "assistant", "content": "Takut blank itu manusiawi banget, dan biasanya pas momennya dateng kamu lebih siap dari yang kamu kira sekarang. Kamu udah sampe tahap ini bukan karena kebetulan."},
            {"role": "user", "content": "iya sih. makasih ya"},
            {"role": "assistant", "content": "Sama-sama. Kamu nggak harus nyelesain semua kekhawatiran itu malam ini."},
        ],
    },
    {
        "label": "Depression",
        "single_turn": False,
        "messages": [
            {"role": "user", "content": "udah seminggu ini berat bat buat bangun pagi"},
            {"role": "assistant", "content": "Kayak tiap pagi harus manjat tembok cuma buat mulai hari — itu exhausting banget, dan kamu udah ngejalanin itu seminggu penuh."},
            {"role": "user", "content": "dan gaada yang notice"},
            {"role": "assistant", "content": "Berat banget pas kamu lagi struggling tapi ngerasa invisible di saat yang sama. Yang kamu rasain tetep valid walaupun orang sekitar nggak lihat."},
            {"role": "user", "content": "makasih udah dengerin"},
            {"role": "assistant", "content": "Aku di sini. Nggak harus buru-buru ngerasa mendingan."},
        ],
    },
    {
        "label": "Normal",
        "single_turn": True,
        "messages": [
            {"role": "user", "content": "hari ini lumayan sih kelas kelar cepet"},
            {"role": "assistant", "content": "Nice, hari yang ngasih kamu ruang napas dikit. Enjoy sisanya ya."},
        ],
    },
    {
        "label": "Suicidal",
        "single_turn": False,
        "messages": [
            {"role": "user", "content": "capek bat rasanya pengen berhenti aja"},
            {"role": "assistant", "content": "\"Pengen berhenti\" itu sinyal kamu udah nahan beban yang berat banget, kelamaan. Aku dengerin, dan aku nggak ke mana-mana."},
            {"role": "user", "content": "kadang kepikiran mending gaada aja"},
            {"role": "assistant", "content": "Makasih udah jujur soal itu — pikiran kayak gitu nakut-nakutin dan berat buat dibawa sendirian. Kamu nggak sendirian ngadepin ini."},
            {"role": "user", "content": "tadi malem sempet mikirin caranya buat ngilang beneran"},
            {"role": "assistant", "content": "Aku beneran khawatir sama kamu sekarang. Tolong ngomong sama Into The Light Indonesia di 119 ext 8, mereka ada 24 jam buat nemenin kamu lewat ini. Kamu berhak dapet itu."},
        ],
    },
]


def _fewshot_block() -> str:
    import json
    lines = []
    for ex in FEWSHOT:
        mode = "single-turn" if ex["single_turn"] else f"{len(ex['messages']) // 2}-turn"
        lines.append(f"# contoh ({ex['label']}, {mode})")
        lines.append(json.dumps({"messages": ex["messages"]}, ensure_ascii=False))
    return "\n".join(lines)


def build_user_prompt(post: str, label: str, *, single_turn: bool, target_turns: int) -> str:
    turns_line = (
        "Buat SATU giliran saja (user lalu assistant)."
        if single_turn
        else f"Buat percakapan {target_turns} giliran (jadi {target_turns} pesan user + {target_turns} pesan assistant, selang-seling)."
    )
    return (
        f"{_fewshot_block()}\n\n"
        f"---\n"
        f"Postingan (Inggris): \"{post}\"\n"
        f"Kategori (ground truth, jangan disebut ke user): {label}\n"
        f"{turns_line}\n"
        f"Balas HANYA JSON."
    )
```

- [ ] **Step 4: Run, verify pass**

Run: `source .venv/bin/activate && pytest tests/test_companion_prompt.py -q`
Expected: PASS (3 tests). If a few-shot example fails validation, fix the example text (not the validator) until it passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/companion_prompt.py tests/test_companion_prompt.py
git commit -m "feat: companion system prompt + validated few-shot examples"
```

---

## Task 4: Rewrite the generation script

**Files:**
- Rename: `scripts/generate_narrative.py` → `scripts/generate_dataset.py` (via `git mv`), then rewrite contents
- Create: `tests/test_generate_dataset.py`

**Interfaces:**
- Consumes: `companion_prompt.SYSTEM_PROMPT`, `companion_prompt.build_user_prompt`, `dataset_validators.validate_conversation`, `tests/conftest.py::FakeClient`.
- Produces (importable from `generate_dataset`):
  - `load_seeds(csv_path: str, n_per_class: int, *, seed: int, min_words: int = 3) -> list[dict]` — each `{"text": str, "label": str}`, balanced per label.
  - `assign_modes(seeds: list[dict], *, multi_turn_ratio: float, turns_min: int, turns_max: int, seed: int) -> list[dict]` — adds `single_turn: bool`, `target_turns: int`.
  - `parse_response(text: str) -> list[dict]` — returns the `messages` list; raises `ValueError` on malformed output.
  - `generate_one(client, model: str, seed_row: dict, *, turns_min: int, turns_max: int, max_attempts: int = 3) -> dict | None` — returns `{"label", "messages"}` or `None` if every attempt fails validation.
  - `main()` — CLI entry point.

- [ ] **Step 1: `git mv` and stub the module**

```bash
git mv scripts/generate_narrative.py scripts/generate_dataset.py
```

Replace the file contents entirely (implementation lands in Step 5; for now leave the old code so the repo still imports — the rewrite is one commit).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_generate_dataset.py`:

```python
import json
import pytest
import generate_dataset as gd
from conftest import FakeClient

SEED_CSV_ROWS = "Unique_ID,text,status\n" + "\n".join(
    f'{i}.0,"post number {i} about feeling things",{label}'
    for i, label in enumerate(
        ["Anxiety", "Depression", "Suicidal", "Normal"] * 5
    )
) + '\n99.0,"hi",Normal\n'  # last row has < 3 words -> must be filtered


@pytest.fixture
def seed_csv(tmp_path):
    p = tmp_path / "raw.csv"
    p.write_text(SEED_CSV_ROWS)
    return str(p)


def test_load_seeds_filters_short_and_balances(seed_csv):
    seeds = gd.load_seeds(seed_csv, n_per_class=3, seed=0)
    assert len(seeds) == 12  # 3 per label x 4 labels
    from collections import Counter
    assert Counter(s["label"] for s in seeds) == {
        "Anxiety": 3, "Depression": 3, "Suicidal": 3, "Normal": 3,
    }
    assert all(len(s["text"].split()) >= 3 for s in seeds)


def test_load_seeds_is_deterministic(seed_csv):
    assert gd.load_seeds(seed_csv, n_per_class=3, seed=0) == gd.load_seeds(seed_csv, n_per_class=3, seed=0)


def test_assign_modes_respects_ratio():
    seeds = [{"text": f"t{i}", "label": "Normal"} for i in range(100)]
    tagged = gd.assign_modes(seeds, multi_turn_ratio=0.7, turns_min=3, turns_max=6, seed=1)
    multi = [t for t in tagged if not t["single_turn"]]
    assert 60 <= len(multi) <= 80
    assert all(3 <= t["target_turns"] <= 6 for t in multi)
    assert all(t["target_turns"] == 1 for t in tagged if t["single_turn"])


def test_parse_response_handles_fenced_and_plain():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    plain = json.dumps({"messages": msgs})
    fenced = f"```json\n{plain}\n```"
    assert gd.parse_response(plain) == msgs
    assert gd.parse_response(fenced) == msgs
    with pytest.raises(ValueError):
        gd.parse_response("not json at all")
    with pytest.raises(ValueError):
        gd.parse_response('{"foo": 1}')


def _good_convo_json(single=False):
    msgs = [
        {"role": "user", "content": "anjir capek bat hari ini gaada abisnya"},
        {"role": "assistant", "content": "Hari yang nguras abis kayak gitu berat banget. Wajar kamu ngerasa kosong sekarang."},
    ]
    if not single:
        msgs += [
            {"role": "user", "content": "iya pengen rebahan aja"},
            {"role": "assistant", "content": "Rebahan dulu aja, badan kamu jelas minta jeda. Nggak harus produktif hari ini."},
            {"role": "user", "content": "makasih ya"},
            {"role": "assistant", "content": "Aku di sini kok. Pelan-pelan aja."},
        ]
    return json.dumps({"messages": msgs}, ensure_ascii=False)


def test_generate_one_returns_validated_dict():
    client = FakeClient([_good_convo_json(single=False)])
    row = {"text": "i am so tired all the time", "label": "Depression",
           "single_turn": False, "target_turns": 3}
    out = gd.generate_one(client, "fake-model", row, turns_min=3, turns_max=6)
    assert out["label"] == "Depression"
    assert len(out["messages"]) == 6


def test_generate_one_retries_then_gives_up():
    client = FakeClient(["garbage", "still garbage", '{"messages": []}'])
    row = {"text": "i am so tired all the time", "label": "Depression",
           "single_turn": False, "target_turns": 3}
    out = gd.generate_one(client, "fake-model", row, turns_min=3, turns_max=6, max_attempts=3)
    assert out is None
    assert len(client.messages.calls) == 3


def test_generate_one_succeeds_on_second_attempt():
    client = FakeClient(["garbage", _good_convo_json(single=False)])
    row = {"text": "i am so tired all the time", "label": "Anxiety",
           "single_turn": False, "target_turns": 3}
    out = gd.generate_one(client, "fake-model", row, turns_min=3, turns_max=6)
    assert out is not None
    assert len(client.messages.calls) == 2
```

- [ ] **Step 3: Run, verify fail**

Run: `source .venv/bin/activate && pytest tests/test_generate_dataset.py -q`
Expected: FAIL — old module has no `load_seeds` / `assign_modes` / `parse_response` / `generate_one`.

- [ ] **Step 4: Run the existing validator + prompt tests to confirm no regression baseline**

Run: `source .venv/bin/activate && pytest -q`
Expected: Tasks 1–3 tests PASS; only `test_generate_dataset.py` fails.

- [ ] **Step 5: Write `scripts/generate_dataset.py`**

```python
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
```

- [ ] **Step 6: Run the new tests, verify pass**

Run: `source .venv/bin/activate && pytest tests/test_generate_dataset.py -q`
Expected: PASS (8 tests).

- [ ] **Step 7: Run the full suite + a no-API dry run**

Run:
```bash
source .venv/bin/activate && pytest -q && python scripts/generate_dataset.py --dry-run --n-per-class 300
```
Expected: all tests PASS; dry run prints `1200 seeds | multi-turn ratio 0.7x` and a balanced `Counter`.

- [ ] **Step 8: Commit**

```bash
git add scripts/generate_dataset.py tests/test_generate_dataset.py
git commit -m "feat: rewrite generator — one call per seed, full conversation, validated + retried"
```

---

## Task 5: Turn-type stratified split

**Files:**
- Modify: `scripts/split_dataset.py`
- Create: `tests/test_split_dataset.py`

**Interfaces:**
- Consumes: `conversations.jsonl` rows shaped `{"label", "messages"}`.
- Produces (importable from `split_dataset`):
  - `split_dataset.turn_type(row: dict) -> str` — `"single"` if exactly one user turn else `"multi"`.
  - `split_dataset.split_conversations(rows: list[dict], *, seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> tuple[list, list, list]` — stratified by `(label, turn_type)`; raises `ValueError` if the same opening user message lands in more than one split.

- [ ] **Step 1: Write the failing test**

Create `tests/test_split_dataset.py`:

```python
from collections import Counter
import pytest
import split_dataset as sd


def _row(label, n_user_turns, uid):
    msgs = [{"role": "user", "content": f"opening {uid}"},
            {"role": "assistant", "content": "Valid banget rasanya. Aku dengerin ya."}]
    for k in range(n_user_turns - 1):
        msgs += [{"role": "user", "content": f"more {uid}-{k}"},
                 {"role": "assistant", "content": "Berat ya. Aku di sini kok."}]
    return {"label": label, "messages": msgs}


def test_turn_type():
    assert sd.turn_type(_row("Normal", 1, 0)) == "single"
    assert sd.turn_type(_row("Normal", 4, 0)) == "multi"


def test_split_is_stratified_and_leak_free():
    rows = []
    uid = 0
    for label in ["Suicidal", "Depression", "Anxiety", "Normal"]:
        for _ in range(70):
            rows.append(_row(label, 4, uid)); uid += 1
        for _ in range(30):
            rows.append(_row(label, 1, uid)); uid += 1

    train, valid, test = sd.split_conversations(rows, seed=42)
    assert len(train) + len(valid) + len(test) == len(rows)
    assert abs(len(train) - 0.8 * len(rows)) <= 4

    # every (label, turn_type) stratum present in train
    strata = Counter((r["label"], sd.turn_type(r)) for r in train)
    assert len(strata) == 8

    openings = [Counter(m["messages"][0]["content"] for m in split) for split in (train, valid, test)]
    assert (openings[0] & openings[1]) == Counter()
    assert (openings[0] & openings[2]) == Counter()


def test_split_rejects_leaked_duplicate_opening():
    dup = _row("Normal", 1, 0)
    with pytest.raises(ValueError):
        sd.split_conversations([dup, dict(dup)] * 10, seed=1)
```

- [ ] **Step 2: Run, verify fail**

Run: `source .venv/bin/activate && pytest tests/test_split_dataset.py -q`
Expected: FAIL — `split_dataset` has no `turn_type` / `split_conversations`.

- [ ] **Step 3: Rewrite `scripts/split_dataset.py`**

```python
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
```

- [ ] **Step 4: Run, verify pass**

Run: `source .venv/bin/activate && pytest tests/test_split_dataset.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Full suite**

Run: `source .venv/bin/activate && pytest -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/split_dataset.py tests/test_split_dataset.py
git commit -m "feat: stratify split by label + turn-type, add opening-message leak guard"
```

---

## Task 6: Docs, README, archive old data

**Files:**
- Modify: `docs/references.md`
- Modify: `README.md`
- Rename: `dataset/processed/narrative_dataset.jsonl` → `.jsonl.bak`; `train.jsonl` → `train.jsonl.bak`; `valid.jsonl` → `valid.jsonl.bak`; `test.jsonl` → `test.jsonl.bak` (via `git mv`)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing importable — documentation + housekeeping only.

- [ ] **Step 1: Archive the old single-turn dataset**

```bash
cd dataset/processed
git mv narrative_dataset.jsonl narrative_dataset.jsonl.bak
git mv train.jsonl train.jsonl.bak
git mv valid.jsonl valid.jsonl.bak
git mv test.jsonl test.jsonl.bak
cd ../..
```

- [ ] **Step 2: Rewrite the "Pendekatan" + "Struktur" sections of `README.md`**

Replace the `## Pendekatan` bullet list with:

```markdown
## Pendekatan

- **Framing**: teman curhat (companion), bukan alat klinis. 4 label
  (`Suicidal`/`Depression`/`Anxiety`/`Normal`) dari `ourafla/Mental-Health_Text-Classification_Dataset`
  dipakai hanya sebagai steering internal saat generate (aturan krisis vs non-krisis)
  dan metadata buat stratified split — model tidak pernah menyebut label ke user.
- **Dataset**: `scripts/generate_dataset.py` — tiap baris CSV di-transcreate Claude jadi
  pesan chat orang pertama gaya Gen Z Indonesia (slang, typo, singkatan), lalu dilanjut
  jadi percakapan 1–6 giliran. ~70% multi-turn. Tiap balasan asisten wajib lolos
  validator di `scripts/dataset_validators.py` (nol tanda tanya, 1–4 kalimat, mayoritas
  Bahasa Indonesia, aturan hotline krisis) — yang gagal di-regenerate lalu di-drop.
- **Fine-tuning**: LoRA (rank 8, 16 layer terakhir) pakai [mlx-lm](https://github.com/ml-explore/mlx-lm)
  di Apple Silicon, `mask_prompt=true`.
- **Target deployment**: on-device iOS (Core ML, kuantisasi INT4).
```

In `## Struktur`, update the `scripts/` block:

```
scripts/
  generate_dataset.py     # sampling + generate percakapan pakai Claude (+ validator)
  dataset_validators.py   # cek struktur & gaya tiap percakapan (unit-tested)
  companion_prompt.py     # system prompt + few-shot buat generate
  split_dataset.py        # stratified split train/valid/test (label + turn-type)
  merge_lora_to_hf.py     # merge adapter LoRA (mlx) -> checkpoint HF PyTorch
  load_model.py           # utilitas load model
```

And change the `dataset/processed/` line to:
```
dataset/processed/        # conversations.jsonl + train/valid/test.jsonl
```

- [ ] **Step 3: Reframe `docs/references.md`**

Change the opening paragraph (the "Dipakai sebagai ground truth ... generate narasi" sentence) to:

```markdown
Dipakai sebagai acuan definisi tingkat keparahan tiap kategori saat sampling seed di
`scripts/generate_dataset.py`. Penanda linguistik di bawah **tidak** muncul di dalam
balasan chatbot — model diposisikan sebagai teman curhat, bukan alat klasifikasi. Sitasi
ini murni dokumentasi kenapa 4 kategori itu dipisah begitu.
```

In `## Catatan batasan`, replace the last bullet ("Dataset sumber ... bukan menentukan ulang labelnya") with:

```markdown
- Label `status` dari dataset sumber dipakai apa adanya sebagai steering internal generate
  (krisis vs non-krisis) + kunci stratifikasi split. Balasan yang digenerate adalah respons
  companion — tidak menyebut kategori, tidak mengklaim diagnosis, tidak mengutip riset.
```

- [ ] **Step 4: Verify no stale references remain**

Run:
```bash
grep -rn "generate_narrative\|narrative_dataset\.jsonl\b\|narasi" README.md docs/ scripts/ || echo "clean"
```
Expected: only hits are in `docs/references.md` explaining the reframe and any `.jsonl.bak` mentions — no live reference to `generate_narrative.py` or a `narrative_dataset.jsonl` input. Fix any that remain.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/references.md dataset/processed/
git commit -m "docs: reframe as companion dataset; archive old single-turn data as .bak"
```

---

## Task 7: Generation runbook (manual — spends API credits)

**Files:**
- Create: `dataset/processed/conversations.jsonl` (generated)
- Modify: `dataset/processed/{train,valid,test}.jsonl` (generated)

**Interfaces:**
- Consumes: `generate_dataset.main`, `split_dataset.main`.
- Produces: the committed dataset artifacts.

> This task runs the real generator. Cost ≈ 1200 `claude-sonnet-5` calls (plus regens),
> on the order of a few USD. Requires a working `ANTHROPIC_API_KEY` (or `ant auth login`
> profile) in `.venv`. Do these steps by hand and eyeball the output — do not automate.

- [ ] **Step 1: Smoke run (12 conversations, ~$0.05)**

```bash
source .venv/bin/activate
python scripts/generate_dataset.py --limit 12 --out dataset/processed/_smoke.jsonl
```
Expected: `Wrote N conversations` with N ≥ 9 (a few drops are fine). Print of drop reasons if any.

- [ ] **Step 2: Read all 12 by hand**

```bash
python -c "import json;[print(f'--- {d[\"label\"]}',*[f'{m[\"role\"][:1]}: {m[\"content\"]}' for m in d['messages']],sep='\n') for d in map(json.loads,open('dataset/processed/_smoke.jsonl'))]"
```
Check every conversation:
- opening message reads like a real Gen Z ID chat (not translated English, not a counselor)
- assistant replies validate feelings, never ask a question, never advise unprompted, never invent context
- Suicidal ones: hotline only after an explicit cue; vague ideation validated without referral

If a systemic problem shows up (e.g. replies too long, register too formal, invents context), fix `scripts/companion_prompt.py` `SYSTEM_PROMPT`, re-run Steps 1–2. Commit prompt fixes separately:
```bash
git add scripts/companion_prompt.py && git commit -m "fix: tune companion prompt from smoke-run read"
```

- [ ] **Step 3: Delete the smoke file and do the full run**

```bash
rm dataset/processed/_smoke.jsonl
python scripts/generate_dataset.py --n-per-class 300 2>&1 | tee /tmp/gen.log
```
Expected: `Wrote ~1150-1200 conversations -> dataset/processed/conversations.jsonl`. Inspect the printed `drop reasons` — if any single reason is > 10% of attempts, fix the prompt and re-run.

- [ ] **Step 4: Sanity-check the dataset**

```bash
python -c "
import json
from collections import Counter
rows=[json.loads(l) for l in open('dataset/processed/conversations.jsonl')]
print('total', len(rows))
print('label', Counter(r['label'] for r in rows))
multi=sum(1 for r in rows if sum(m['role']=='user' for m in r['messages'])>1)
print('multi-turn frac', round(multi/len(rows),2))
q=sum(1 for r in rows for m in r['messages'] if m['role']=='assistant' and '?' in m['content'])
print('assistant turns with ?:', q)
"
```
Expected: total ≈ 1150–1200; labels roughly balanced; multi-turn frac ≈ 0.65–0.72; **assistant turns with `?`: 0**.

- [ ] **Step 5: Split and check**

```bash
python scripts/split_dataset.py
```
Expected: three files written; per-file `(label, turn_type)` distribution printed, all 8 strata present in `train.jsonl`; no `ValueError`.

- [ ] **Step 6: Read 20 random conversations from the full set**

```bash
python -c "
import json,random
rows=[json.loads(l) for l in open('dataset/processed/conversations.jsonl')]
random.seed(0)
for d in random.sample(rows,20):
    print('===',d['label'])
    for m in d['messages']: print(f'  {m[\"role\"][:1]}: {m[\"content\"]}')
"
```
Confirm the §10 success criteria from the spec hold. If quality is off, fix the prompt and regenerate (back to Step 3).

- [ ] **Step 7: Commit the dataset**

```bash
git add dataset/processed/conversations.jsonl dataset/processed/train.jsonl dataset/processed/valid.jsonl dataset/processed/test.jsonl
git commit -m "data: regenerate companion dataset (1200 convos, ~70% multi-turn, native ID input)"
```

- [ ] **Step 8: Full test suite one more time**

Run: `source .venv/bin/activate && pytest -q`
Expected: all PASS.

---

## Follow-up (NOT in this plan)

Retrain tuning is a separate plan once this dataset is committed:
- `training/lora_config.yaml`: `scale: 10.0 → 2.0`, `iters: 1200 → ~500`, hand-pick the min-valid-loss checkpoint, try `rank: 8 → 16`, keep `mask_prompt: true`.
- Re-probe with a 4-turn venting conversation (the harness used during brainstorming) before shipping — expect no verbatim repetition and no invented context.
- Then Core ML re-export from the new adapter (existing `fix-ios-coreml-load` pipeline).

---

## Self-Review

**Spec coverage:**
- §1 root cause → addressed by Tasks 3–4 (multi-turn, native input, no-question prompt) + Task 7 (regeneration).
- §3 product framing → Task 3 (`SYSTEM_PROMPT` never names label), Task 6 (docs).
- §4 generation architecture (Approach A) → Task 4 `generate_one` (one call per seed, whole conversation).
- §5 style guide → Task 3 `SYSTEM_PROMPT` + Task 2 validators enforcing each rule.
- §5 messy user turns → Task 3 `SYSTEM_PROMPT` "ATURAN GILIRAN USER" + few-shot.
- §5 crisis rules + safety net → Task 2 `crisis_gate_ok` + Task 3 prompt.
- §6 validators (8 checks) → Tasks 1–2 (schema, roles, turn count, no-questions, length, dupes, language, crisis gate); Task 4 wires regen/drop + stats.
- §7 composition (1200, 300/label, 70% multi, 960/120/120, stratified) → Task 4 defaults + Task 5 split + Task 7 run.
- §8 file changes → Tasks 3–6 one-to-one with the table.
- §9 follow-up → "Follow-up" section, explicitly out of scope.
- §10 success criteria → Task 7 Steps 4–6 check each one.

**Placeholder scan:** No TBD/TODO. Every code step has real code. Test bodies are complete. Manual steps (Task 7) list exact commands + expected output.

**Type consistency:** `validate_conversation(obj, *, single_turn, turns_min=3, turns_max=6)` — same signature in Task 2 definition, Task 2 tests, Task 4 `generate_one`. `load_seeds`/`assign_modes`/`parse_response`/`generate_one` signatures match between Task 4 interface block, Task 4 tests, and Task 4 implementation. `turn_type` / `split_conversations` match between Task 5 interface, tests, implementation. `FakeClient(responses)` defined in Task 1 `conftest.py`, used in Task 4 tests with `.messages.calls`.
