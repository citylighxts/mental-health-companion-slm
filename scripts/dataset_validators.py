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
