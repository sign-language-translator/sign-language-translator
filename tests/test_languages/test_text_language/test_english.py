import re
import string

from sign_language_translator.config.enums import TextLanguages
from sign_language_translator.languages.text.english import English
from sign_language_translator.languages.utils import get_text_language
from sign_language_translator.text.tagger import Tags


def test_english_initialization():
    nlp = get_text_language("english")
    assert isinstance(nlp, English)

    nlp = English()
    assert English.name() == TextLanguages.ENGLISH.value
    assert re.compile(nlp.token_regex()).findall("abc123") == ["abc", "123"]
    assert set(string.ascii_letters + string.digits) <= nlp.allowed_characters()

    assert nlp.vocab is not None
    assert nlp.tokenizer is not None
    assert nlp.tagging_rules is not None
    assert nlp.tagger is not None
    assert nlp.omitted_tokens is not None


def test_english_static_attributes():
    assert English.UNICODE_RANGE[0] <= ord("A")
    assert English.UNICODE_RANGE[1] >= ord("z")

    assert English.NUMBER_REGEX
    assert re.compile(English.NUMBER_REGEX).match("123")
    assert not re.compile(English.NUMBER_REGEX).match("abc")

    assert English.WORD_REGEX
    assert re.compile(English.WORD_REGEX).match("abc")
    assert not re.compile(English.WORD_REGEX).match("123")

    assert English.UNALLOWED_CHARACTERS_REGEX
    assert re.compile(English.UNALLOWED_CHARACTERS_REGEX).search("hey you!\n") is None
    assert re.compile(English.UNALLOWED_CHARACTERS_REGEX).search("wot?¿⁇") is not None


def test_english_preprocessing():
    nlp = English()

    text_pairs = [
        (  # Foreign characters
            "━═══Hello, how are you?⁇ I’m fine, thank you… åøµ👍🏼۔ ═══━",
            "Hello, how are you? I'm fine, thank you...",
        ),
        (  # Spaces
            "  hello , \t   world!  \n\n\n\n",
            "hello, world!",
        ),
        (  # Acronyms
            "COVID started in 2019.",
            "COVID started in 2019.",
        ),
        (  # Punctuation
            "sorry  , I forgor.\n  :(",
            "sorry, I forgor.:(",
        ),
    ]

    for text, expected in text_pairs:
        assert nlp.preprocess(text) == expected


def test_english_word_tokenizer():
    nlp = English()

    text_pairs = [
        (  # Basic
            "Hello, how are you?",
            ["Hello", ",", "how", "are", "you", "?"],
        ),
        (  # Numbers
            "Covid19 ended in 2021-08.",
            ["Covid", "19", "ended", "in", "2021", "-", "08", "."],
        ),
        (  # Punctuation
            "sorry, I forgor. 😢",
            ["sorry", ",", "I", "forgor", ".", "😢"],
        ),
        (  # Compound words
            "I will watch a half an hour long drama serial.",
            ["I", "will", "watch", "a", "half an hour", "long", "drama serial", "."],
        ),
        (  # word sense
            "Milo(name) company(corporation) made blue rubber(eraser)",
            ["Milo(name)", "company(corporation)", "made", "blue", "rubber(eraser)"],
        ),
        # (  # todo: Contractions
        #     "I'm going to the store.",
        #     ["I", "'m", "going", "to", "the", "store", "."],
        # ),
    ]

    for text, expected in text_pairs:
        assert nlp.tokenize(text) == expected


def test_english_sentence_tokenizer():
    nlp = English()

    text_pairs = [
        (  # Basic . ? ! newline
            "sorry, it's so sad. steve died of ligma?? It cant be! RIP\n😭",
            ["sorry, it's so sad.", "steve died of ligma??", "It cant be!", "RIP\n😭"],
        ),
        (  # Acronyms
            "COVID started in 2019. U.S.S.R ended in 1991. U.K. is still here.",
            ["COVID started in 2019.", "U.S.S.R ended in 1991.", "U.K. is still here."],
        ),
        (  # Numbers
            "It's 1.78 meters tall. 02:05:30.566 hr long. and v2.1.3 of model.",
            ["It's 1.78 meters tall.", "02:05:30.566 hr long.", "and v2.1.3 of model."],
        ),
    ]

    for text, expected in text_pairs:
        assert nlp.sentence_tokenize(text) == expected


def test_english_detokenization():
    nlp = English()

    token_sentence_pairs = [
        (
            [],
            "",
        ),
        (
            ["", ""],
            "",
        ),
        (
            ["Hello", "\t", ",", "this", "is", "me", " ", "!"],
            "Hello, this is me!",
        ),
        (
            ["I", "am", "a", "robot", ".", "what", "is", "my", "purpose", "?", "\n"],
            "I am a robot. what is my purpose?",
        ),
    ]

    for tokens, sentence in token_sentence_pairs:
        assert nlp.detokenize(tokens) == sentence


def test_english_tagging():
    nlp = English()

    tokens_tags_pairs = [
        (
            ["This", " ", "line", "."],
            [Tags.SUPPORTED_WORD, Tags.SPACE, Tags.WORD, Tags.PUNCTUATION],
        ),
        (
            "Mark(name) will mark last(final) 20".split(),
            [Tags.NAME, Tags.AMBIGUOUS, Tags.WORD, Tags.SUPPORTED_WORD, Tags.NUMBER],
        ),
        (
            "Today marks 2024-09-03 03:46 A.M.".split(),
            [Tags.SUPPORTED_WORD, Tags.WORD, Tags.DATE, Tags.TIME, Tags.ACRONYM],
        ),
        (
            "door",
            [Tags.SUPPORTED_WORD],
        ),
    ]

    for tokens, tags in tokens_tags_pairs:
        assert nlp.get_tags(tokens) == tags
        assert nlp.tag(tokens) == list(
            zip(tokens if isinstance(tokens, list) else [tokens], tags)
        )


def test_english_word_sense_disambiguation():
    nlp = English()

    tokens_senses_pairs = [
        (
            "march",
            [{"march(month)"}],
        ),
        (
            ["orange", "grandfather", "close"],
            [
                {"orange(fruit)", "orange(color)"},
                {"grandfather(paternal)", "grandfather(maternal)"},
                {"close(near)"},  # "close(shut)"
            ],
        ),
        (
            "hello",
            [set()],
        ),
    ]

    for tokens, senses in tokens_senses_pairs:
        predicted_senses = nlp.get_word_senses(tokens)
        for predicted, expected in zip(predicted_senses, senses):
            assert expected <= set(predicted)


def test_english_romanization():
    nlp = English()

    texts = [
        "Hello, how are you?",
    ]
    expected_romanized = texts

    for txt, exp_rom in zip(texts, expected_romanized):
        assert exp_rom == nlp.romanize(txt)
