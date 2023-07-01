import os

from sign_language_translator.languages.vocab import Vocab


def test_vocab_single_sign_collection():
    data_path = os.path.join(os.path.dirname(__file__), "data")

    vocab = Vocab(
        language="english",
        sign_collections=["abc-xyz-1"],
        data_root_dir=data_path,
        arg_is_regex=False,
    )

    assert vocab.word_to_labels == {
        "word-0": [["abc-xyz-1_sign-1"]],
        "word-1": [["abc-xyz-1_sign-1"]],
        "word-2": [["abc-xyz-1_sign-2"]],
        "word-3(context)": [["abc-xyz-1_sign-2"]],
    }
    assert vocab.token_to_id == {" ": 0, ".": 1}
    assert vocab.supported_words == {
        "word-0",
        "word-1",
        "word-2",
        "word-3",
    }
    assert vocab.ambiguous_to_unambiguous == {"word-3": ["word-3(context)"]}

    assert vocab.person_names == ["alice", "bob"]
    assert vocab.joint_word_to_split_words == {"icecream": "ice cream"}


def test_vocab_all_sign_collections():
    data_path = os.path.join(os.path.dirname(__file__), "data")

    vocab = Vocab(
        language=r"^english$",
        sign_collections=[r".*"],
        data_root_dir=data_path,
        arg_is_regex=True,
    )

    assert {key: {tuple(seq) for seq in val} for key, val in vocab.word_to_labels.items()} == {
        "word-0": {("abc-xyz-2_sign-1",), ("abc-xyz-1_sign-1",)},
        "word-1": {("abc-xyz-1_sign-1",)},
        "word-2": {("abc-xyz-1_sign-2",)},
        "word-3(context)": {("abc-xyz-1_sign-2",)},
        "word-4": {("abc-xyz-1_sign-1", "abc-xyz-2_sign-1",), ("abc-xyz-2_sign-3",)},
        "word-5": {("abc-xyz-1_sign-1", "abc-xyz-2_sign-1",), ("abc-xyz-2_sign-3",)},
        "word-6": {("abc-xyz-2_sign-1",)},
        "word-7": {("abc-xyz-2_sign-1", "abc-xyz-1_sign-2",)},
        "word-8": {("abc-xyz-2_sign-1", "abc-xyz-1_sign-2",)},
    }
    assert vocab.supported_words_with_word_sense == {
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
