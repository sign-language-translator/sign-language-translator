from sign_language_translator.models.utils import top_p_top_k_indexes


def test_top_p_top_k_indexes():
    probs = [0.1, 0.2, 0.15, 0.05, 0.3, 0.2]
    top_p = 0.75
    top_k = 3

    assert top_p_top_k_indexes(probs, top_p, top_k) == [4, 1, 5]
    assert top_p_top_k_indexes(probs, None, None) == [4, 1, 5, 2, 0, 3]
