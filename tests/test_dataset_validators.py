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
