import os

from sign_language_translator.languages.vocab import Vocab


def test_vocab_single_sign_collection():
    data_path = os.path.join(os.path.dirname(__file__), "data")

    vocab = Vocab("english", sign_collections=["abc-xyz-1"], data_root_dir=data_path)

    assert vocab.sign_labels == ["abc-xyz-1_sign-1", "abc-xyz-1_sign-2"]
    assert vocab.label_to_words == {
        "abc-xyz-1_sign-1": ["word-0", "word-1"],
        "abc-xyz-1_sign-2": ["word-2", "word-3(context)"],
    }
    assert vocab.label_sequence_to_words == {}
    assert vocab.supported_words == {
        "alice",
        "bob",
        "word-0",
        "word-1",
        "word-2",
        "word-3",
    }
    assert vocab.ambiguous_to_context == {"word-3": ["word-3(context)"]}


def test_vocab_all_sign_collections():
    data_path = os.path.join(os.path.dirname(__file__), "data")

    vocab = Vocab("english", sign_collections=None, data_root_dir=data_path)

    assert vocab.sign_labels == [
        "abc-xyz-1_sign-1",
        "abc-xyz-1_sign-2",
        "abc-xyz-2_sign-3",
        "abc-xyz-2_sign-1",
    ]
    assert vocab.label_to_words == {
        "abc-xyz-1_sign-1": ["word-0", "word-1"],
        "abc-xyz-1_sign-2": ["word-2", "word-3(context)"],
        "abc-xyz-2_sign-3": ["word-4", "word-5"],
        "abc-xyz-2_sign-1": ["word-6", "word-0"],
    }
    assert vocab.label_sequence_to_words == {
        ("abc-xyz-2_sign-1", "abc-xyz-1_sign-2"): ["word-7", "word-8"],
        ("abc-xyz-1_sign-1", "abc-xyz-2_sign-1"): ["word-4", "word-5"],
    }
    assert vocab.supported_words_with_context == {
        "alice",
        "bob",
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
    assert vocab.ambiguous_to_context == {"word-3": ["word-3(context)"]}


# def test_sign_collection_regex_matching():
#     pass
