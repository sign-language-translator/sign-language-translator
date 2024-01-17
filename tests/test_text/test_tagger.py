from sign_language_translator.text.tagger import Rule, Tagger, Tags


def get_tagger():
    rules = [
        Rule.from_pattern(r"^\s$", Tags.SPACE, 5),
        Rule.from_pattern(r"^[\.,\?\-!]$", Tags.PUNCTUATION, 5),
        Rule.from_pattern(r"^\w+$", Tags.WORD, 5),
        Rule.from_pattern(r"^\d+$", Tags.NUMBER, 4),
        Rule.from_pattern(r"^[A-Z]{2,7}$", Tags.ACRONYM, 4),
        Rule.from_pattern(r"^\d{4}-\d{2}-\d{2}$", Tags.DATE, 5),
        Rule(lambda token: token.lower() in ["mudassar", "iqbal"], Tags.NAME, 3),
    ]
    tagger = Tagger(rules=rules, default=Tags.DEFAULT)

    return tagger


def test_tagger():
    tagger = get_tagger()

    tokens = [
        "I",
        " ",
        "am",
        " ",
        "Mudassar",
        ".",
        ".",
        ".",
        " ",
        "COVID",
        "-",
        "19",
        "!",
    ]
    tags = [
        Tags.WORD,
        Tags.SPACE,
        Tags.WORD,
        Tags.SPACE,
        Tags.NAME,
        Tags.PUNCTUATION,
        Tags.PUNCTUATION,
        Tags.PUNCTUATION,
        Tags.SPACE,
        Tags.ACRONYM,
        Tags.PUNCTUATION,
        Tags.NUMBER,
        Tags.PUNCTUATION,
    ]
    assert len(tokens) == len(tags), "fix the test-case"
    assert tagger.get_tags(tokens) == tags
