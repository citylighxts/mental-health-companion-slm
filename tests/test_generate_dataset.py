import json
import sys
import anthropic
import httpx2
import pytest
import generate_dataset as gd
from conftest import FakeClient


def _timeout_error():
    return anthropic.APITimeoutError(request=httpx2.Request("POST", "https://api.test"))

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
    out, reasons = gd.generate_one(client, "fake-model", row, turns_min=3, turns_max=6)
    assert out["label"] == "Depression"
    assert len(out["messages"]) == 6
    assert reasons == []


def test_generate_one_retries_then_gives_up():
    client = FakeClient(["garbage", "still garbage", '{"messages": []}'])
    row = {"text": "i am so tired all the time", "label": "Depression",
           "single_turn": False, "target_turns": 3}
    out, reasons = gd.generate_one(client, "fake-model", row, turns_min=3, turns_max=6,
                                   max_attempts=3)
    assert out is None
    assert reasons == ["parse_error"]
    assert len(client.messages.calls) == 3


def test_generate_one_returns_final_attempt_reasons():
    """Drop reasons come back with the result, so no stale tally carries over."""
    bad = json.dumps({"messages": [
        {"role": "user", "content": "besok interview nih gua panik parah"},
        {"role": "assistant", "content": "Gimana perasaan kamu sekarang soal itu semua?"},
    ]})
    client = FakeClient([bad, bad, bad])
    row = {"text": "i am anxious about tomorrow", "label": "Anxiety",
           "single_turn": True, "target_turns": 1}
    out, reasons = gd.generate_one(client, "fake-model", row, turns_min=3, turns_max=6)
    assert out is None
    assert "assistant_turn_contains_question" in reasons


def test_generate_one_succeeds_on_second_attempt():
    client = FakeClient(["garbage", _good_convo_json(single=False)])
    row = {"text": "i am so tired all the time", "label": "Anxiety",
           "single_turn": False, "target_turns": 3}
    out, _ = gd.generate_one(client, "fake-model", row, turns_min=3, turns_max=6)
    assert out is not None
    assert len(client.messages.calls) == 2


def test_call_survives_a_transient_timeout(monkeypatch):
    """A network blip retries instead of aborting the batch (FIX 1a)."""
    monkeypatch.setattr(gd.time, "sleep", lambda _s: None)
    client = FakeClient([_timeout_error(), _good_convo_json(single=False)])
    row = {"text": "i am so tired all the time", "label": "Depression",
           "single_turn": False, "target_turns": 3}
    out, _ = gd.generate_one(client, "fake-model", row, turns_min=3, turns_max=6)
    assert out is not None
    assert len(client.messages.calls) == 2


def test_call_caches_the_system_prefix_not_the_user_message():
    client = FakeClient([_good_convo_json(single=True)])
    row = {"text": "hari ini biasa aja", "label": "Normal",
           "single_turn": True, "target_turns": 1}
    gd.generate_one(client, "fake-model", row, turns_min=3, turns_max=6)
    call = client.messages.calls[0]

    # the breakpoint is on the stable system block, not a top-level kwarg
    assert "cache_control" not in call
    system = call["system"]
    assert isinstance(system, list)
    assert system[-1]["cache_control"] == {"type": "ephemeral"}

    # the few-shot examples ride along in the cached system text...
    assert "anjir besok interview nih gua deg2an parah gabisa tidur" in system[-1]["text"]
    # ...and are NOT concatenated into the varying user message
    user_msg = call["messages"][0]["content"]
    assert "# contoh" not in user_msg
    assert '"messages"' not in user_msg
    assert "hari ini biasa aja" in user_msg


def test_already_done_reads_and_tolerates_missing_sidecar(tmp_path):
    missing = tmp_path / "nope.jsonl.seeds"
    assert gd.already_done(missing) == set()

    sidecar = tmp_path / "conversations.jsonl.seeds"
    sidecar.write_text(
        json.dumps("seed one") + "\n"
        + json.dumps("seed\nwith newline") + "\n"
        + "\n"  # blank line tolerated
        + json.dumps("seed three") + "\n"
    )
    assert gd.already_done(sidecar) == {"seed one", "seed\nwith newline", "seed three"}


def test_sidecar_path_sits_next_to_the_output(tmp_path):
    assert gd.sidecar_path(tmp_path / "conversations.jsonl").name == "conversations.jsonl.seeds"


def _run_main(monkeypatch, seed_csv, out, responses):
    """Drive main() end to end against a FakeClient (4 single-turn seeds)."""
    fake = FakeClient(responses)
    monkeypatch.setattr(gd.anthropic, "Anthropic", lambda **kw: fake)
    monkeypatch.setattr(gd.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sys, "argv", [
        "generate_dataset.py", "--csv", seed_csv, "--out", str(out),
        "--n-per-class", "1", "--multi-turn-ratio", "0",
    ])
    gd.main()
    return fake


def test_main_writes_all_seeds_and_removes_sidecar(monkeypatch, seed_csv, tmp_path):
    out = tmp_path / "conversations.jsonl"
    _run_main(monkeypatch, seed_csv, out, [_good_convo_json(single=True)] * 4)
    assert len(out.read_text().strip().splitlines()) == 4
    assert not gd.sidecar_path(out).exists()  # clean completion cleans up


def test_main_resumes_from_sidecar_and_skips_done_seeds(monkeypatch, seed_csv, tmp_path):
    out = tmp_path / "conversations.jsonl"
    seeds = gd.load_seeds(seed_csv, 1, seed=0)
    sidecar = gd.sidecar_path(out)
    sidecar.write_text("".join(json.dumps(s["text"]) + "\n" for s in seeds[:2]))
    out.write_text('{"label": "Normal", "messages": []}\n')  # partial output survives

    fake = _run_main(monkeypatch, seed_csv, out, [_good_convo_json(single=True)] * 2)

    assert len(fake.messages.calls) == 2  # only the two unfinished seeds were called
    assert len(out.read_text().strip().splitlines()) == 3  # appended, not truncated
    assert not sidecar.exists()


def test_main_survives_an_exception_on_one_seed(monkeypatch, seed_csv, tmp_path):
    out = tmp_path / "conversations.jsonl"
    responses = [_timeout_error()] * 4 + [_good_convo_json(single=True)] * 3
    _run_main(monkeypatch, seed_csv, out, responses)
    # the seed that blew past the retry budget is dropped; the batch keeps going
    assert len(out.read_text().strip().splitlines()) == 3
    assert not gd.sidecar_path(out).exists()


def test_report_repeated_replies_prints_top_offenders(tmp_path, capsys):
    out = tmp_path / "conversations.jsonl"
    closer = "Aku di sini kok. Santai aja dulu ya."
    with open(out, "w") as f:
        for i in range(3):
            f.write(json.dumps({"label": "Normal", "messages": [
                {"role": "user", "content": f"halo {i}"},
                {"role": "assistant", "content": closer},
            ]}) + "\n")
    gd.report_repeated_replies(out)
    printed = capsys.readouterr().out
    assert "top repeated assistant turns" in printed
    assert "aku di sini kok santai aja dulu ya" in printed
    assert "100.0%" in printed


def test_report_repeated_replies_tolerates_a_truncated_final_line(tmp_path, capsys):
    """A hard kill mid-write leaves a half-written last line — the report must not crash."""
    out = tmp_path / "conversations.jsonl"
    good = json.dumps({"label": "Normal", "messages": [
        {"role": "user", "content": "halo"},
        {"role": "assistant", "content": "Aku di sini kok. Santai aja dulu ya."},
    ]})
    with open(out, "w") as f:
        f.write(good + "\n")
        f.write(good + "\n")
        f.write('{"label": "Normal", "mess')  # truncated, no newline

    gd.report_repeated_replies(out)  # must not raise
    printed = capsys.readouterr().out
    assert "aku di sini kok santai aja dulu ya" in printed
    assert "100.0%" in printed  # 2/2 good rows — the truncated line was skipped
