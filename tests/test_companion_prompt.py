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
