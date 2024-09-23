__all__ = [
    "English",
]

import re
from string import ascii_lowercase, ascii_uppercase, digits
from typing import Any, Dict, Iterable, List, Set, Tuple, Union

from sign_language_translator.config.assets import Assets
from sign_language_translator.config.enums import TextLanguages
from sign_language_translator.languages.text.text_language import TextLanguage
from sign_language_translator.languages.vocab import Vocab
from sign_language_translator.text.preprocess import remove_space_before_punctuation
from sign_language_translator.text.tagger import Rule, Tagger, Tags
from sign_language_translator.text.tokenizer import SignTokenizer


class English(TextLanguage):
    """NLP class for English text. Extends `slt.languages.text.TextLanguage` class.

    English is originally a West Germanic language and potentially an international language in the 21st century.
    English uses the Latin script, which consists of 26 letters and is written from left to right.
    There are two variants of these letters: uppercase (capital letters) and lowercase.
    See unicode details at: https://unicode.org/charts/PDF/U0000.pdf
    """

    @staticmethod
    def name() -> str:
        return TextLanguages.ENGLISH.value

    @classmethod
    def token_regex(cls) -> str:
        return f"({cls.NUMBER_REGEX}|{cls.WORD_REGEX})"

    @classmethod
    def allowed_characters(cls) -> Set[str]:
        return cls.ALLOWED_CHARACTERS

    def __init__(self) -> None:
        # TODO: filter vocab datasets via args
        self.vocab = self.__get_vocab()
        self.tokenizer = self.__get_tokenizer()
        self.tagging_rules = self.__get_tagging_rules()
        self.tagger = Tagger(rules=self.tagging_rules, default=Tags.DEFAULT)
        self.omitted_tokens = {"", " ", "\t"}

    def preprocess(self, text: str) -> str:
        text = text.translate(English.CHARACTER_TRANSLATOR)
        text = self.delete_unallowed_characters(text)
        text = re.sub(r"[^\S\n]{2,}", " ", text)
        text = remove_space_before_punctuation(text, self.PUNCTUATION)
        text = text.strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(  # todo: join 'nt 's 'm 're 've 'll 'd 't 'em
            text, join_compound_words=True, join_word_sense=True
        )
        tokens = [t for t in tokens if t not in self.omitted_tokens]
        return tokens

    def sentence_tokenize(self, text: str) -> List[str]:
        sentences = self.tokenizer.sentence_tokenize(text)
        sentences = [
            s for sen in sentences if (s := sen.strip()) not in self.omitted_tokens
        ]
        return sentences

    def detokenize(self, tokens: Iterable[str]) -> str:
        text = " ".join(tokens)
        text = remove_space_before_punctuation(text, self.PUNCTUATION)
        text = text.strip()

        return text

    def tag(self, tokens: Union[str, Iterable[str]]) -> List[Tuple[str, Any]]:
        if isinstance(tokens, str):
            tokens = [tokens]

        tagged = self.tagger.tag(tokens)

        return tagged

    def get_tags(self, tokens: Union[str, Iterable[str]]) -> List[Any]:
        if isinstance(tokens, str):
            tokens = [tokens]

        tags = self.tagger.get_tags(tokens)

        return tags

    def get_word_senses(self, tokens: Union[str, Iterable[str]]) -> List[List[str]]:
        if isinstance(tokens, str):
            tokens = [tokens]

        word_senses = [
            self.vocab.ambiguous_to_unambiguous.get(token.lower(), [])
            for token in tokens
        ]

        return word_senses

    def romanize(self, text: str, *args, add_diacritics=True, **kwargs) -> str:
        return text

    # ================== #
    #     Characters     #
    # ================== #

    UNICODE_RANGE = (32, 126)  # 0x0020 - 0x007E

    FULL_STOPS: List[str] = ["."]
    QUESTION_MARKS: List[str] = ["?"]
    END_OF_SENTENCE_MARKS: List[str] = FULL_STOPS + QUESTION_MARKS + ["!"]

    PUNCTUATION: List[str] = END_OF_SENTENCE_MARKS + list(",;:")

    BRACKETS: List[str] = ["(", ")", "[", "]", "{", "}"]
    QUOTES: List[str] = ['"', "'"]
    SYMBOLS: List[str] = PUNCTUATION + BRACKETS + QUOTES + list("@#$%&*+<>=^|/-_")

    ALPHABET: List[str] = list(ascii_uppercase) + list(ascii_lowercase)

    ALLOWED_CHARACTERS = set(ALPHABET) | set(digits) | set(SYMBOLS) | set(" \n")

    CHARACTER_MAP: Dict[str, str] = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }
    CHARACTER_TRANSLATOR = {ord(c): d for c, d in CHARACTER_MAP.items()}

    # ========== #
    #    Regex   #
    # ========== #

    WORD_REGEX = r"[^\W_\d]+"
    NUMBER_REGEX = r"\d+(?:[\.:]\d+)*"
    UNALLOWED_CHARACTERS_REGEX = (
        "[^" + "".join(map(re.escape, ALLOWED_CHARACTERS)) + "]"
    )

    # ====================== #
    #    Helper Functions    #
    # ====================== #

    def delete_unallowed_characters(self, text: str) -> str:
        text = re.sub(self.UNALLOWED_CHARACTERS_REGEX, " ", text)

        return text

    # ================ #
    #    initialize    #
    # ================ #

    def __get_vocab(
        self,
        region=r"[a-z]+",
        org=r"[a-z]+",
        part_num=r"[0-9]+",
        data_dir=Assets.ROOT_DIR,
        is_regex=True,
    ):
        vocab = Vocab(
            language=f"^{self.name()}$",  # r"^en$"
            country=region,
            organization=org,
            part_number=part_num,
            data_root_dir=data_dir,
            arg_is_regex=is_regex,
        )
        return vocab

    def __get_tokenizer(self):
        tokenizer = SignTokenizer(
            word_regex=self.token_regex(),
            compound_words=(
                self.vocab.supported_tokens
                | set(self.vocab.words_to_numbers.keys())
                | set(self.vocab.person_names)
            ),  # TODO: | one-hundred twenty-three (\d[ \d]*): ["100", "23"] --> ["123"]
            end_of_sentence_tokens=self.END_OF_SENTENCE_MARKS,
            acronym_periods=self.FULL_STOPS,
            non_sentence_end_words=list(ascii_uppercase),  # (acronyms)
            tokenized_word_sense_pattern=[self.WORD_REGEX, r"\(", [r"name"], r"\)"],
            # todo: generalize the tokenized_word_sense_pattern arg for more ListRegex patterns  e.g. numbers & contractions
        )
        return tokenizer

    def __get_tagging_rules(self):
        punctuation_set = set(self.PUNCTUATION)
        tagging_rules = [
            # e.g. " "
            Rule.from_pattern(r"^\s+$", Tags.SPACE, 5),
            # e.g. "," "."
            Rule(lambda token: token in punctuation_set, Tags.PUNCTUATION, 5),
            # e.g. "word"
            Rule.from_pattern("^" + self.WORD_REGEX + "$", Tags.WORD, 5),
            # e.g. COVID
            Rule.from_pattern(r"^([A-Z]\.){2,8}|[A-Z]{2,8}$", Tags.ACRONYM, 4),
            # e.g. 2002-02-20
            Rule.from_pattern(r"^\d{4}-\d{2}-\d{2}$", Tags.DATE, 4),
            # e.g. 09:30:25.333
            Rule.from_pattern(r"^\d+(?::\d+)?(?::\d+(?:\.\d+)?)$", Tags.TIME, 4),
            # e.g. John, Doe(name)
            Rule(
                lambda token: token in self.vocab.person_names
                or token.endswith("(name)"),
                Tags.NAME,
                2,
            ),
            # e.g. Cow, airplane, 1
            Rule(
                lambda token: (token.lower() in self.vocab.supported_tokens),
                Tags.SUPPORTED_WORD,
                3,
            ),
            # e.g. forty-five, 45, 4.50
            Rule(
                lambda token: (
                    bool(re.match(r"^\d+(?:\.\d+)?$", token))
                    or token in self.vocab.words_to_numbers
                ),
                Tags.NUMBER,
                4,
            ),
            # e.g. "spring" -> ["spring(coil)", "spring(season)"]
            Rule(
                lambda token: token.lower() in self.vocab.ambiguous_to_unambiguous,
                Tags.AMBIGUOUS,
                2,
            ),
        ]
        return tagging_rules
