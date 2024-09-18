import pytest

from sign_language_translator.text import SynonymFinder
from sign_language_translator.utils.utils import is_internet_available


@pytest.mark.skipif(not is_internet_available(), reason="No internet available")
def test_translation_synonyms():
    synonymizer = SynonymFinder(language="en")

    word = "hello"
    synonyms = synonymizer.synonyms_by_translation(word, lower_case=True)
    assert set(synonyms) & {"hi", "hey", "howdy", "hello", "greetings", "hola", "hiya"}

    word = "happy"
    cache = {}
    synonyms = synonymizer.synonyms_by_translation(word, lower_case=True, cache=cache)
    assert set(synonyms) & {"happy", "glad", "cheerful", "joyful", "merry", "delighted"}
    assert word in cache

    synonymizer.language = "ur"
    word = "اشارہ"
    synonyms = synonymizer.synonyms_by_similarity(word, top_k=10, min_similarity=0.5)
    assert set(synonyms) & {"اشارہ", "اشارے"}
