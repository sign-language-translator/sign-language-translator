import os

from sign_language_translator.languages.vocab import Vocab


def test_vocab_single_sign_collection():
    data_path = os.path.join(os.path.dirname(__file__), "data")

    vocab = Vocab(
        language="en",
        country="xy",
        organization="abc",
        part_number="1",
        data_root_dir=data_path,
        arg_is_regex=False,
    )

    assert vocab.word_to_labels == {
        "word-0": [["xy-abc-1_sign-1"]],
        "word-1": [["xy-abc-1_sign-1"]],
        "word-2": [["xy-abc-1_sign-2"]],
        "word-3(context)": [["xy-abc-1_sign-2"]],
    }
    assert vocab.ambiguous_to_unambiguous == {"word-3": ["word-3(context)"]}

    assert vocab.person_names == ["alice", "bob"]
    assert vocab.joint_word_to_split_words == {"icecream": "ice cream"}

    assert vocab.labels == {"xy-abc-1_sign-1", "xy-abc-1_sign-2"}


def test_vocab_all_sign_collections():
    data_path = os.path.join(os.path.dirname(__file__), "data")

    vocab = Vocab(
        language=r"^en$",
        country=r".+",
        organization=r".+",
        part_number=r"[0-9]+",
        data_root_dir=data_path,
        arg_is_regex=True,
    )

    assert {
        key: {tuple(seq) for seq in val} for key, val in vocab.word_to_labels.items()
    } == {
        "word-0": {("xy-abc-2_sign-1",), ("xy-abc-1_sign-1",)},
        "word-1": {("xy-abc-1_sign-1",)},
        "word-2": {("xy-abc-1_sign-2",)},
        "word-3(context)": {("xy-abc-1_sign-2",)},
        "word-4": {("xy-abc-1_sign-1", "xy-abc-2_sign-1"), ("xy-abc-2_sign-3",)},
        "word-5": {("xy-abc-1_sign-1", "xy-abc-2_sign-1"), ("xy-abc-2_sign-3",)},
        "word-6": {("xy-abc-2_sign-1",)},
        "word-7": {("xy-abc-2_sign-1", "xy-abc-1_sign-2")},
        "word-8": {("xy-abc-2_sign-1", "xy-abc-1_sign-2")},
    }
    assert vocab.supported_tokens == {
        "word-0",
        "word-1",
        "word-2",
        "word-3(context)",
        "word-4",
        "word-5",
        "word-6",
        "word-7",
        "word-8",
    }
