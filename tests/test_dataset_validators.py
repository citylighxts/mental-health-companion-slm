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
