from sign_language_translator.text.utils import (
    ListRegex,
    concatenate_sentence_terminals,
    extract_supported_subsequences,
    make_ngrams,
)


def test_make_ngrams():
    sequence = [0, 1, 2, 3, 4, 5]

    expected_unigrams = [[0], [1], [2], [3], [4], [5]]
    unigrams = make_ngrams(sequence, 1)
    assert unigrams == expected_unigrams

    expected_bigrams = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    bigrams = make_ngrams(sequence, 2)
    assert bigrams == expected_bigrams

    expected_four_grams = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
    four_grams = make_ngrams(sequence, 4)
    assert four_grams == expected_four_grams


def test_extract_supported_subsequence():
    seqn = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tags = [4, 1, 1, 2, 2, 2, 3, 3, 2, 2]

    supported_tags = {1, 2}
    skipped = {5}
    expected_subsequences = [[2, 3, 4], [6], [9, 10]]

    subsequences = extract_supported_subsequences(
        seqn, tags, supported_tags, skipped_items=skipped
    )

    assert subsequences == expected_subsequences


def test_concatenate_sentence_terminals():
    start, end = "<s>", "</s>"

    sentence_terminal_pairs = [
        (
            ["Hello!", "How are you?", "I'm fine."],
            ["Hello!</s>", "<s>How are you?</s>", "<s>I'm fine."],
        ),
        (
            ["I can't understand my old code!"],
            ["I can't understand my old code!"],
        ),
        (
            ["How have you been?", "I've been worried."],
            ["How have you been?</s>", "<s>I've been worried."],
        )
    ]

    for sentence, expected in sentence_terminal_pairs:
        concatenated = concatenate_sentence_terminals(sentence, start, end)
        assert concatenated == expected


def test_list_regex():
    items = ["abc", "lmn", "123", "123", "123", "xyz", "def", "pqr"]
    span = ListRegex.match(
        items,
        [
            r"(abc|cba)",
            [r"jk", r"lmno?"],
            ("123", (0, 2)),  # test for max_count=0
        ],
    )
    assert span == (0, 4)

    items = [
        "hello",
        "(",
        "word",
        "-",
        "word",
        ")",
        "something",
        "somgething",
        "world",
        "(",
        "sense",
        ")",
        ".",
    ]
    matches = ListRegex.find_all(
        items, [r"\w+", r"\(", r"\w+", ([r"-", r"\w+"], (0, None)), r"\)"]
    )
    assert matches == [
        ["hello", "(", "word", "-", "word", ")"],
        ["world", "(", "sense", ")"],
    ]
