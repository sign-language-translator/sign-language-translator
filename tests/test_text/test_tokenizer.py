import string

from sign_language_translator.text.tokenizer import SignTokenizer


def get_tokenizer():
    tokenizer = SignTokenizer(
        word_regex=r"[a-zA-Z]+",
        compound_words=["ice-cream", "book shop"],
        acronym_periods=["."],
        end_of_sentence_tokens=[".", "?", "!"],
        non_sentence_end_words=list(string.ascii_uppercase),
        tokenized_word_sense_pattern=[
            r"\w+",
            r"\(",
            r"\w+",
            ([r"-", r"\w+"], (0, None)),  # interval quantifier
            r"\)",
        ],
    )
    return tokenizer


def test_word_tokenize():
    tokenizer = get_tokenizer()

    text = "i like ice-cream from book shop#3!"
    tokens = [
        "i",
        " ",
        "like",
        " ",
        "ice-cream",
        " ",
        "from",
        " ",
        "book shop",
        "#",
        "3",
        "!",
    ]
    assert tokenizer.tokenize(text) == tokens


def test_sentence_tokenize():
    tokenizer = get_tokenizer()

    text = "hello, I am J. Doe. Who are you? "
    sentences = ["hello, I am J. Doe.", " Who are you?", " "]
    assert tokenizer.sentence_tokenize(text) == sentences


def test_detokenize():
    tokenizer = get_tokenizer()

    tokens = ["hello", " ", "world", "!", " ", "1", "9", "9", "9"]
    assert tokenizer.detokenize(tokens) == "hello world! 1999"


def test_compound_word_map():
    tokenizer = get_tokenizer()

    expected_word_map = {
        "ice": {("ice", "-", "cream"), ("ice", " ", "sickle")},
        "sweet": {("sweet", " ", "heart")},
    }
    word_map = tokenizer._make_compound_word_map(
        ["ice-cream", "ice sickle", "sweet", "sweet heart"]
    )
    word_map = {k: {tuple(tokens) for tokens in v} for k, v in word_map.items()}
    assert word_map == expected_word_map


def test_join_subwords():
    tokenizer = get_tokenizer()

    tokens = ["ice", "-", "cream", " ", "from", " ", "book", " ", "shop"]
    expected_joint = ["ice-cream", " ", "from", " ", "book shop"]
    assert tokenizer._join_subwords(tokens) == expected_joint


def test_join_word_sense():
    tokenizer = get_tokenizer()

    tokens = [
        "this",
        " ",
        "is",
        " ",
        "a",
        "spring",
        "(",
        "metal",
        "-",
        "coil",
        ")",
        ".",
    ]
    expected_joint = ["this", " ", "is", " ", "a", "spring(metal-coil)", "."]
    assert tokenizer._join_word_sense(tokens) == expected_joint
