from sign_language_translator.text.metrics import Perplexity


def test_perplexity():
    all_tokens = set("abcdefghijklmnopqrstuvwxyz")
    perplexity = Perplexity(all_tokens=all_tokens, regularizing_constant=0)

    corpus = ["january", "february", "march", "april", "may", "june"]
    corpus += ["july", "august", "september", "october", "november", "december"]
    perplexity.update_frequencies(corpus)

    assert abs(perplexity.evaluate("january") - 14.386706553232036) <= 1e5
    assert abs(perplexity.evaluate("monday") - 22.516625029076184) <= 1e5
    assert abs(perplexity.evaluate("mudassar") - 19.001121731534607) <= 1e5
