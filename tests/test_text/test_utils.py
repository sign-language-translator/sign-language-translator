from sign_language_translator.text.utils import (
    extract_supported_subsequence,
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

    subsequences = extract_supported_subsequence(
        seqn, tags, supported_tags, skipped_items=skipped
    )

    assert subsequences == expected_subsequences
