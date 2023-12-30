from sign_language_translator.text import SynonymFinder


def test_translation_synonyms():
    synonymizer = SynonymFinder(language="en")

    word = "hello"
    synonyms = synonymizer.synonyms_by_translation(word, lower_case=True)
    assert set(synonyms) & {"hi", "hey", "howdy", "hello", "greetings", "hola", "hiya"}

    word = "happy"
    synonyms = synonymizer.synonyms_by_translation(word, lower_case=True)
    assert set(synonyms) & {"happy", "glad", "cheerful", "joyful", "merry", "delighted"}
