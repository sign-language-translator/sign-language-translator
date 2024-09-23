import os
import re
import warnings

from sign_language_translator.config.assets import Assets
from sign_language_translator.languages.text.urdu import Tags, Urdu


def get_urdu_processor_object():
    return Urdu() if os.path.exists(Assets.ROOT_DIR) else None


def test_urdu_preprocessor():
    ur_nlp = get_urdu_processor_object()

    if ur_nlp is None:
        warnings.warn(
            "urdu text processor object could not be initialized (check slt.Assets.ROOT_DIR)"
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

    assert ur_nlp.remove_diacritics("اُن کا۔") == "ان کا۔"
    assert (
        re.sub(
            f"[{''.join(map(re.escape, ur_nlp.allowed_characters()))}]",
            "",
            "اُن four کا۔",
        )
        == "four"
    )

    assert processes_texts == expected_texts

    # english: today my heart says: "hello world!".
    # Urdu: ہم آج کل یہ

    assert ur_nlp.poetry_preprocessor("آج مرا دل کر رہا ہے۔") == "آج میرا دل کر رہا ہے۔"
    assert ur_nlp.passage_preprocessor(" دل کر   رہا ہے۔") == "دل کر رہا ہے۔"
    assert ur_nlp.wikipedia_preprocessor(" دل کر   رہا ہے۔...\n") == "دل کر   رہا ہے۔"


def test_urdu_tokenizer():
    ur_nlp = get_urdu_processor_object()

    if ur_nlp is None:
        warnings.warn(
            "urdu text processor object could not be initialized (check slt.Assets.ROOT_DIR)"
        )
        return

    raw_texts = [
        "hello world!",
        "صبح ۸ بجے، اور شام ۹:۳۰پر",
        "سبحان(نام) اسلام آباد جا رہا ہے۔",  # "اسلام آباد سے وہ آئے"
        # TODO: ["word(word-sense)", "word"]
    ]
    expected_tokenized = [
        ["hello", " ", "world", "!"],
        ["صبح", " ", "۸", " ", "بجے", "،", " ", "اور", " ", "شام", " ", "۹:۳۰", "پر"],
        ["سبحان(نام)", " ", "اسلام آباد", " ", "جا", " ", "رہا", " ", "ہے", "۔"],
    ]
    tokenized = list(map(ur_nlp.tokenize, raw_texts))
    assert tokenized == expected_tokenized
    assert raw_texts[0] == ur_nlp.detokenize(tokenized[0])

    assert ur_nlp.tag("COVID") == [("COVID", Tags.ACRONYM)]
    assert ur_nlp.get_tags("COVID") == [Tags.ACRONYM]


def test_urdu_sentence_tokenizer():
    ur_nlp = get_urdu_processor_object()

    if ur_nlp is None:
        warnings.warn(
            "urdu text processor object could not be initialized (check slt.Assets.ROOT_DIR)"
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
            "urdu text processor object could not be initialized (check slt.Assets.ROOT_DIR)"
        )
        return

    tokens = [
        ["hello", " ", "world", "!"],
        ["شام", " ", "۰۹:۳۰", "پر(پہ)"],
        ["سبحان(نام)"],
    ]
    expected_tags = [
        [Tags.WORD, Tags.SPACE, Tags.WORD, Tags.PUNCTUATION],
        [Tags.AMBIGUOUS, Tags.SPACE, Tags.TIME, Tags.SUPPORTED_WORD],
        [Tags.NAME],
    ]
    tags = list(map(ur_nlp.get_tags, tokens))
    assert tags == expected_tags

    assert repr(Tags.ACRONYM) == "Tags.ACRONYM"


def test_word_senses():
    ur_nlp = get_urdu_processor_object()

    if ur_nlp is None:
        warnings.warn(
            "urdu text processor object could not be initialized (check slt.Assets.ROOT_DIR)"
        )
        return

    raw_words = [
        "میں",
    ]
    expected_word_senses = [
        {"میں(متکلم)", "میں(اندر)"},
    ]
    word_senses = list(map(lambda x: set(ur_nlp.get_word_senses(x)[0]), raw_words))

    assert word_senses == expected_word_senses


def test_urdu_romanization():
    nlp = Urdu()

    texts = [
        "ایک انار سو بیمار کی مثال تو ہم سب نے ہی سنی ہے۔",
        "میں نے ۴۷ کتابیں خریدی ہیں۔ ۱۹۸۷ء",
        "مکّهی کا زکریّاؒ کی قابلِ تعریف قوّت سے منہ کهٹّا ہو گیا ہے۔۔۔",
    ]
    expected_romanized = [
        ("ayk anar so bimar ki msal to hm sb ny hi sni hy.", True),
        ("mein̲ ny 47 ktabein̲ k̲h̲ridi hen̲. 1987CE", True),
        ("mkkhi ka zkryya(RH) ki qabl-e ta'rif qoot sy mnh khtta ho gya hy...", False),
    ]

    for txt, (exp_rom, diacritics) in zip(texts, expected_romanized):
        assert exp_rom == nlp.romanize(nlp.preprocess(txt), add_diacritics=diacritics)
