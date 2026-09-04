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


def test_system_prompt_cached_carries_all_four_fewshot_examples():
    """The few-shot block is relocated into the cached system text, not dropped."""
    assert cp.FEWSHOT_BLOCK in cp.SYSTEM_PROMPT_CACHED
    assert cp.SYSTEM_PROMPT in cp.SYSTEM_PROMPT_CACHED
    for ex in cp.FEWSHOT:
        first_user = ex["messages"][0]["content"]
        assert first_user in cp.SYSTEM_PROMPT_CACHED, ex["label"]
    assert cp.SYSTEM_PROMPT_CACHED.count("# contoh (") == 4


def test_build_user_prompt_includes_seed_and_mode():
    multi = cp.build_user_prompt("i feel so empty lately", "Depression", single_turn=False, target_turns=4)
    assert "i feel so empty lately" in multi
    assert "Depression" in multi
    assert "4 giliran" in multi
    single = cp.build_user_prompt("hi", "Normal", single_turn=True, target_turns=1)
    assert "SATU giliran" in single
    assert "hi" in single


def test_build_user_prompt_no_longer_carries_the_fewshot_json():
    """The few-shot block moved to the cached system prompt — the user message is per-seed only."""
    p = cp.build_user_prompt("i feel so empty lately", "Depression", single_turn=False, target_turns=4)
    assert "# contoh" not in p
    assert '"messages"' not in p
