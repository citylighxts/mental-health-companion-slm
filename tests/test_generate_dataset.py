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
