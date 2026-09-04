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


# --- content checks ---

_ID_MARKERS = {
    # function words / discourse particles (assistant + user register)
    "yang", "nggak", "nggk", "ga", "gak", "gk", "aku", "kamu", "banget", "bgt",
    "aja", "udah", "lagi", "tuh", "sih", "kok", "deh", "ya", "di", "ke", "itu",
    "ini", "dari", "buat", "sama", "juga", "kalau", "kalo", "biar", "emang",
    "masih", "terus", "trus", "gitu", "harus", "bukan", "kayak", "kayaknya",
    "kek", "lah", "dong", "nih", "kan", "doang", "soalnya", "tapi", "mending",
    "gaada", "mau", "pengen", "pengin", "pengennya", "kepikiran", "rasanya",
    # Gen Z / Jakartan slang + pronouns
    "anjir", "anjay", "njir", "jir", "gua", "gue", "gw", "lu", "lo", "elu",
    "gabut", "mager", "bat", "parah",
    # everyday content words that are not also English
    "capek", "cape", "capai", "males", "malas", "lelah", "pusing", "bingung",
    "takut", "sedih", "marah", "nangis", "semangat", "hari", "orang", "badan",
    "tidur", "bangun", "kerja", "kuliah", "sekolah", "temen", "teman",
}

_CRISIS_CUE_PATTERNS = [
    r"bunuh diri", r"pengen mati", r"pengin mati", r"mau mati", r"pengen mati aja",
    r"akhiri hidup", r"ngakhirin hidup", r"ngilangin diri", r"ngilangin nyawa",
    r"gantung diri", r"loncat dari", r"lompat dari", r"nyilet", r"nyayat",
    r"overdosis", r"minum obat banyak", r"nenggak obat", r"cara(nya)?\s+(buat\s+)?(mati|bunuh diri|ngilang)",
    r"kill myself", r"end my life", r"end it all", r"\bsuicide\b", r"take my own life",
]
_HOTLINE_MARKERS = ["into the light", "119 ext 8", "119 ekstensi 8", "119 ext.8", "119 ext8"]

# below this many word-tokens a user turn is too short to language-check meaningfully
_USER_LANG_MIN_TOKENS = 5


def is_majority_indonesian(text: str, *, threshold: float = 0.12) -> bool:
    """Heuristic: does this read as Indonesian rather than English?

    Long texts must clear BOTH bars (>= 2 markers AND the density threshold) — an
    English paragraph that happens to sprinkle in "ya" and "kok" is not Indonesian.
    Short texts keep the `hits >= 2` escape: a legitimately short reply can carry
    two markers without reaching the density bar.
    """
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in _ID_MARKERS)
    ratio = hits / len(tokens)
    if len(tokens) > 12:
        return hits >= 2 and ratio >= threshold
    return hits >= 2 or ratio >= threshold


def mentions_crisis_cue(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in _CRISIS_CUE_PATTERNS)


def mentions_hotline(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _HOTLINE_MARKERS)


def crisis_gate_ok(label: str, messages: list[dict]) -> bool:
    """Crisis referral rules for `Suicidal` conversations (spec §5).

    The gate keys off the USER turns — the model cannot be trusted to self-flag.
    Three ways to fail:

    1. a user turn names an explicit method/plan cue and neither that turn's reply
       nor any later assistant turn mentions the hotline;
    2. the conversation runs 3+ assistant turns with no hotline mention anywhere
       (the safety net: prolonged crisis talk must land somewhere);
    3. the hotline is mentioned in 2+ assistant turns — referral spam is exactly the
       verbatim-repetition failure mode this dataset rework exists to kill. The spec
       asks for *one* gentle mention.

    Everything else passes: a 1-2 turn conversation with only vague ideation and no
    hotline is the intended shape (validate first, do not reflex-refer).
    """
    if label != "Suicidal":
        return True

    asst = assistant_turns(messages)
    n_asst = len(asst)
    n_hotline = sum(1 for t in asst if mentions_hotline(t))

    for i, m in enumerate(messages):
        if m.get("role") != "user" or not mentions_crisis_cue(m["content"]):
            continue
        later_assistant = [
            mm["content"] for mm in messages[i + 1:] if mm.get("role") == "assistant"
        ]
        if not any(mentions_hotline(t) for t in later_assistant):
            return False  # (1) explicit cue never answered with a referral

    if n_asst >= 3 and n_hotline == 0:
        return False  # (2) safety net
    if n_hotline >= 2:
        return False  # (3) referral spam
    return True


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
    # User turns are written by the model too, so they can drift into English just as
    # easily. Lower threshold: user turns are short and slangy. Turns under
    # _USER_LANG_MIN_TOKENS are skipped — the system prompt explicitly asks for
    # one-word replies ("iya", "mboh", "gatau") and no ratio is meaningful there.
    for t in user_turns(messages):
        if len(re.findall(r"[a-zA-Z']+", t)) < _USER_LANG_MIN_TOKENS:
            continue
        if not is_majority_indonesian(t, threshold=0.08):
            reasons.append("user_turn_not_indonesian")
            break
    if not crisis_gate_ok(label, messages):
        reasons.append("crisis_gate_missing_hotline")
    return reasons
