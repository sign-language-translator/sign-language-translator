import re

from sign_language_translator.languages.text import Hindi
from sign_language_translator.languages.utils import get_text_language
from sign_language_translator.text.tagger import Tags


def test_hindi_initialization():
    nlp_object = get_text_language("hindi")
    assert isinstance(nlp_object, Hindi)

    nlp_object = Hindi()
    assert nlp_object.name()


def test_hindi_preprocessing():
    nlp = Hindi()

    # remove foreign characters
    assert nlp.preprocess("hello आप कैसे हैं?") == "आप कैसे हैं?"
    # hello aap kese hain (hello how are you?)

    # preserve acronyms
    assert (
        nlp.preprocess("SLT का अर्थ है साइन लैंग्वेज ट्रांसलेटर")
        == "SLT का अर्थ है साइन लैंग्वेज ट्रांसलेटर"
    )  # SLT ka arth hai sign language translator (SLT means sign language translator)

    # remove extra spaces & symbols
    assert nlp.preprocess("आप   कैसे हैं ?¿⁇") == "आप कैसे हैं?"  # how are you?

    # normalize hindi characters
    text_1 = "य़ ग़ालिब ख़ानसामा बड़ी आवाज़ में वक़्त की फ़िक्र करता था।"  # this Ghalib Khansama used to worry about the time in a loud voice.
    text_2 = "य़ ग़ालिब ख़ानसामा बड़ी आवाज़ में वक़्त की फ़िक्र करता था।"
    assert nlp.preprocess(text_1) == text_2


def test_hindi_tokenization():
    nlp = Hindi()

    # word tokenization
    text = "आप कैसे हैं? "  # how are you?
    tokens = nlp.tokenize(text)
    assert tokens == ["आप", "कैसे", "हैं", "?"]

    # detokenization
    assert nlp.detokenize(tokens) == nlp.preprocess(text)


def test_hindi_sentence_tokenization():
    nlp = Hindi()

    texts = [
        # Basic . ? !
        "आप कैसे हैं? मैं ठीक हूँ। शानदार!",  # how are you? I'm fine. Amazing!
        # Acronyms
        "बी.एस. विज्ञान और एम॰एस॰ की डिग्री लेने के बाद वह अपने घर लौटी। जो भारत में है",  # after getting a B.S. and M.S. in science, she returned home. which is in India.
        "आई।सी।सी। विश्व कप लीग चीन में होगा॥जो एशिया में है.",  # ICC World Cup League will be in China.which is in Asia.
        # Numbers
        "ये पिछले वालों से २.५ गुना बेहतर हैं. जो बहुत शानदार है.",  # these are 2.5 times better than the previous ones. which is amazing.
    ]
    expected_sentences = [
        ["आप कैसे हैं?", "मैं ठीक हूँ।", "शानदार!"],
        ["बी.एस. विज्ञान और एम॰एस॰ की डिग्री लेने के बाद वह अपने घर लौटी।", "जो भारत में है"],
        ["आई।", "सी।", "सी।", "विश्व कप लीग चीन में होगा॥", "जो एशिया में है."],
        ["ये पिछले वालों से २.५ गुना बेहतर हैं.", "जो बहुत शानदार है."],
    ]

    sentences = [nlp.sentence_tokenize(t) for t in texts]
    assert sentences == expected_sentences


def test_hindi_tagging():
    nlp = Hindi()

    tokens = ["आप", "कैसे", "हैं", "?"]  # ["you", "how", "are", "?"]
    tags = nlp.get_tags(tokens)
    assert tags[3] == Tags.PUNCTUATION
    for i in range(3):
        assert tags[i] in (Tags.WORD, Tags.SUPPORTED_WORD)

    assert nlp.tag(tokens) == list(zip(tokens, tags))

    assert nlp.tag(" ") == [(" ", Tags.SPACE)]
    assert nlp.get_tags("करण(नाम)") == [Tags.NAME]


def test_hindi_word_sense_disambiguation():
    nlp = Hindi()

    # get word senses
    possible_senses = nlp.get_word_senses("सोना")  # sona (sleep/gold)
    assert set(possible_senses[0]) & {"सोना(स्वर्ण)"}


def test_hindi_static_attributes():
    for k, v in Hindi.CHARACTER_TO_DECOMPOSED.items():
        assert len(k) == 1
        assert len(v) > 1

    assert re.compile(Hindi.NUMBER_REGEX)
    assert re.compile(Hindi.WORD_REGEX)
    assert re.compile(Hindi.token_regex())
    assert re.compile(Hindi.UNALLOWED_CHARACTERS_REGEX)

    assert isinstance(Hindi.allowed_characters(), set)
    assert Hindi.UNICODE_RANGE[1] > Hindi.UNICODE_RANGE[0]
    assert all(len(c) == 1 for c in Hindi.allowed_characters())
    assert all(len(c) == 1 for c in Hindi.CHARACTERS)
    assert all(len(c) == 1 for c in Hindi.DIACRITICS)


def test_hindi_romanization():
    nlp = Hindi()

    texts = [
        "अपनी ऊंचाई के कारण उछाल पाने में भी कामयाब होते हैं.",
        "मैंने किताब खरीदी है।",
        "ईशांत को शानदार गेंदबाजी के लिए १ अवॉर्ड दिया गया।",
    ]
    expected_romanized = [
        ("apnī ūnchaī ke karṇ uchhal pane men bhī kamyab hote hain.", True),
        ("mainne kitab khrīdī hai.", True),
        ("ishant ko shandar gendbaji ke lie 1 avord diya gya.", False),
    ]

    for txt, (exp_rom, diacritics) in zip(texts, expected_romanized):
        assert exp_rom == nlp.romanize(txt, add_diacritics=diacritics)
