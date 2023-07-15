import os
import warnings

from sign_language_translator import Settings
from sign_language_translator.languages.text.urdu import Tags, Urdu


def get_urdu_processor_object():
    return Urdu() if os.path.exists(Settings.DATASET_ROOT_DIRECTORY) else None


def test_urdu_preprocessor():
    ur_nlp = get_urdu_processor_object()

    if ur_nlp is None:
        warnings.warn(
            "urdu text processor object could not be initialized (check slt.Settings.DATASET_ROOT_DIRECTORY)"
        )
        return

    raw_texts = [
        "━═════﷽════━ ✺ «آج کل یہ",
        "آؤ گئی آأؤۂیٔۓ",
        "500 USD is ١۵٠٠٠٠ PKR ",
        "ایم ۔ پی۔ اے۔",
    ]
    expected_texts = [
        "بسم اللہ الرحمن الرحیم آج کل یہ",
        "آؤ گئی آأؤۂئۓ",
        "500 USD ۱۵۰۰۰۰ PKR",
        "ایم۔ پی۔ اے۔",
    ]
    processes_texts = list(map(ur_nlp.preprocess, raw_texts))

    assert processes_texts == expected_texts


def test_urdu_tokenizer():
    ur_nlp = get_urdu_processor_object()

    if ur_nlp is None:
        warnings.warn(
            "urdu text processor object could not be initialized (check slt.Settings.DATASET_ROOT_DIRECTORY)"
        )
        return

    raw_texts = [
        "hello world!",
        "صبح ۸ بجے، اور شام ۹:۳۰پر",
        "سبحان(نام) اسلام آباد جا رہا ہے۔",  # "اسلام آباد سے وہ آئے"
        # :TODO: ["word(word-sense)", "word"]
    ]
    expected_tokenized = [
        ["hello", " ", "world", "!"],
        ["صبح", " ", "۸", " ", "بجے", "،", " ", "اور", " ", "شام", " ", "۹:۳۰", "پر"],
        ["سبحان(نام)", " ", "اسلام آباد", " ", "جا", " ", "رہا", " ", "ہے", "۔"],
    ]
    tokenized = list(map(ur_nlp.tokenize, raw_texts))
    assert tokenized == expected_tokenized


def test_urdu_sentence_tokenizer():
    ur_nlp = get_urdu_processor_object()

    if ur_nlp is None:
        warnings.warn(
            "urdu text processor object could not be initialized (check slt.Settings.DATASET_ROOT_DIRECTORY)"
        )
        return

    raw_texts = [
        "Hello world! My name is John Doe.",
        "U.S.S.R. is an acronym. So, do not break sentence there.",
        # "This is v2.0.23. right?",
        "This is version 2.0.23. right?",
        "09:30:25.333",
        "وہ شاہ کے والد اور ایم۔ پی۔ اے۔  حسین شاہ کے چچا تھے۔",
    ]
    expected_sentences = [
        ["Hello world!", "My name is John Doe."],
        ["U.S.S.R. is an acronym.", "So, do not break sentence there."],
        # ["This is v2.0.23.", "right?"],
        ["This is version 2.0.23.", "right?"],
        ["09:30:25.333"],
        ["وہ شاہ کے والد اور ایم۔ پی۔ اے۔  حسین شاہ کے چچا تھے۔"],
    ]
    sentences = list(map(ur_nlp.sentence_tokenize, raw_texts))
    assert sentences == expected_sentences


def test_urdu_tagger():
    ur_nlp = get_urdu_processor_object()

    if ur_nlp is None:
        warnings.warn(
            "urdu text processor object could not be initialized (check slt.Settings.DATASET_ROOT_DIRECTORY)"
        )
        return

    tokens = [
        ["hello", " ", "world", "!"],
        ["شام", " ", "۰۹:۳۰", "پر"],
        ["سبحان(نام)"],
    ]
    expected_tags = [
        [Tags.WORD, Tags.SPACE, Tags.WORD, Tags.PUNCTUATION],
        [Tags.SUPPORTED_WORD, Tags.SPACE, Tags.TIME, Tags.SUPPORTED_WORD],
        [Tags.NAME],
    ]
    tags = list(map(ur_nlp.get_tags, tokens))
    assert tags == expected_tags


def test_word_senses():
    ur_nlp = get_urdu_processor_object()

    if ur_nlp is None:
        warnings.warn(
            "urdu text processor object could not be initialized (check slt.Settings.DATASET_ROOT_DIRECTORY)"
        )
        return

    raw_words = [
        "میں",
    ]
    expected_word_senses = [
        {"میں(i)", "میں(in)"},
    ]
    word_senses = list(map(lambda x: set(ur_nlp.get_word_senses(x)[0]), raw_words))

    assert word_senses == expected_word_senses
