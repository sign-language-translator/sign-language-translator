from sign_language_translator.languages.text.text_language import TextLanguage


def test_base_romanization():

    char_map = {ord("a"): "@", ord("e"): "3", ord("o"): "0"}
    ngram_map = {r"e\b": "£", r"ou": "º"}
    texts = [
        "Hello, how are you?",
    ]
    expected_romanized = [
        "H3ll0, h0w @r£ yº?",
    ]
    for txt, exp_rom in zip(texts, expected_romanized):
        assert exp_rom == TextLanguage.romanize(
            txt, character_translation_table=char_map, n_gram_map=ngram_map
        )
