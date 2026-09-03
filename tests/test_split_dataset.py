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
