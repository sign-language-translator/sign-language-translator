__all__ = [
    "Hindi",
]

import re
from string import ascii_uppercase, digits
from typing import Any, Dict, Iterable, List, Set, Tuple, Union

from sign_language_translator.config.assets import Assets
from sign_language_translator.config.enums import TextLanguages
from sign_language_translator.languages.text.text_language import TextLanguage
from sign_language_translator.languages.vocab import Vocab
from sign_language_translator.text.preprocess import remove_space_before_punctuation
from sign_language_translator.text.tagger import Rule, Tagger, Tags
from sign_language_translator.text.tokenizer import SignTokenizer


class Hindi(TextLanguage):
    """NLP class for Hindi text. Extends `slt.languages.text.TextLanguage` class.

    Hindi is an Indo-Aryan language spoken mostly in India. Hindi uses the Devanagari script, which consists of 11 vowels and 33 consonants and is written from left to right.
    See unicode details at: https://unicode.org/charts/PDF/U0900.pdf
    """

    @staticmethod
    def name() -> str:
        return TextLanguages.HINDI.value

    @classmethod
    def token_regex(cls) -> str:
        return f"({cls.NUMBER_REGEX}|{cls.WORD_REGEX})"

    @classmethod
    def allowed_characters(cls) -> Set[str]:
        return cls.ALLOWED_CHARACTERS

    def __init__(self) -> None:
        # TODO: args to filter dataset
        self.vocab = self.__get_vocab()
        self.tokenizer = self.__get_tokenizer()
        self.tagging_rules = self.__get_tagging_rules()
        self.tagger = Tagger(rules=self.tagging_rules, default=Tags.DEFAULT)
        self.omitted_tokens = {"", " ", "\t"}

    def preprocess(self, text: str) -> str:
        text = self.normalize_characters(text)
        text = self.delete_unallowed_characters(text)
        text = re.sub(r"[^\S\n]{2,}", " ", text)
        text = remove_space_before_punctuation(text, self.PUNCTUATION)
        text = text.strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(
            text, join_compound_words=True, join_word_sense=True
        )
        tokens = [token for token in tokens if token not in self.omitted_tokens]
        return tokens

    def sentence_tokenize(self, text: str) -> List[str]:
        sentences = self.tokenizer.sentence_tokenize(text)
        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence not in self.omitted_tokens
        ]
        return sentences

    def detokenize(self, tokens: Iterable[str]) -> str:
        text = " ".join(tokens)
        text = remove_space_before_punctuation(text, self.PUNCTUATION)

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
        """Map Hindi characters to phonetically similar characters of the English language.
        Transliteration is useful for readability.

        ALA-LC Romanization Table: https://www.loc.gov/catdir/cpso/romanization/hindi.pdf

        Args:
            text (str): Hindi text to be mapped to Latin script.
            add_diacritics (bool, optional): Whether to use diacritics over English characters to help pronunciation. Defaults to True.

        Examples:

        .. code-block:: python

            import sign_language_translator as slt

            nlp = slt.languages.text.Hindi()

            text = "मैंने किताब खरीदी है।"
            romanized_text = nlp.romanize(text)
            print(romanized_text)
            # 'mainne kitab khrīdī hai.'

            text = "ईशांत शर्मा को उनकी शानदार गेंदबाजी के लिए १ प्लेयर ऑफ द मैच का अवॉर्ड दिया गया।"
            text = nlp.preprocess(text)
            romanized_text = nlp.romanize(text)
            print(romanized_text)
            # 'īshant shrma ko unkī shandar gendbajī ke lie 1 pleyr ôph d maich ka avôrḍ diya gya.'
        """
        text = super().romanize(
            text,
            *args,
            add_diacritics=add_diacritics,
            character_translation_table=self.ROMANIZATION_CHARACTER_TRANSLATOR,
            n_gram_map=self.NGRAM_ROMANIZATION_MAP,
            **kwargs,
        )

        return text

    # ================== #
    #     Characters     #
    # ================== #

    UNICODE_RANGE: Tuple[int, int] = (2304, 2431)  # 0x0900 - 0x097F

    FULL_STOPS: List[str] = [".", "।", "॥"]
    QUESTION_MARKS: List[str] = ["?"]
    ACRONYM_PERIODS: List[str] = ["॰"]
    END_OF_SENTENCE_MARKS: List[str] = FULL_STOPS + QUESTION_MARKS + ["!"]
    PUNCTUATION: List[str] = END_OF_SENTENCE_MARKS + ACRONYM_PERIODS + list(",;:")

    BRACKETS: List[str] = ["(", ")", "[", "]", "{", "}"]
    SYMBOLS: List[str] = PUNCTUATION + BRACKETS + list("-_/")

    CHARACTERS: List[str] = str(
        """
        ऀ  ँ  ं  ः  ऄ  अ  आ  इ  ई  उ  ऊ  ऋ  ऌ  ऍ  ऎ  ए
        ऐ  ऑ  ऒ  ओ  औ  क  ख  ग  घ  ङ  च  छ  ज  झ  ञ  ट
        ठ  ड  ढ  ण  त  थ  द  ध  न  ऩ  प  फ  ब  भ  म  य
        र  ऱ  ल  ळ  ऴ  व  श  ष  स  ह  ऺ  ऻ  ़  ऽ  ा  ि
        ी  ु  ू  ृ  ॄ  ॅ  ॆ  े  ै  ॉ  ॊ  ो  ौ  ्  ॎ  ॏ
        ॐ  ॑  ॒  ॓  ॔  ॕ  ॖ  ॗ  क़  ख़  ग़  ज़  ड़  ढ़  फ़  य़
        ॠ  ॡ  ॢ  ॣ  ।  ॥  ०  १  २  ३  ४  ५  ६  ७  ८  ९
        ॰  ॱ  ॲ  ॳ  ॴ  ॵ  ॶ  ॷ  ॸ  ॹ  ॺ  ॻ  ॼ  ॽ  ॾ  ॿ
        """
    ).split()
    DIACRITICS = str("ऀ  ँ  ं  ः  ॄ  ॅ  ़  ा  ि  ी  ु  ू  ृ  े  ै  ॉ  ो  ौ  ्").split()

    ALLOWED_CHARACTERS: Set[str] = (
        set(CHARACTERS)
        | set(DIACRITICS)
        | set(SYMBOLS)
        | set(ascii_uppercase)
        | set(digits)
        | set("()!.,?/[]{}<> \n")
    )

    CHARACTER_TO_DECOMPOSED: Dict[str, str] = {
        "क़": "क़",
        "ख़": "ख़",
        "ग़": "ग़",
        "ज़": "ज़",
        "ड़": "ड़",
        "ढ़": "ढ़",
        "फ़": "फ़",
        "य़": "य़",
    }
    CHARACTER_TRANSLATOR = {ord(c): d for c, d in CHARACTER_TO_DECOMPOSED.items()}

    # ========== #
    #    Regex   #
    # ========== #

    NUMBER_REGEX = r"\d+(?:[\.:]\d+)*"
    WORD_REGEX = r"[^\W_\d]([^\W_\d]|[" + "".join(DIACRITICS) + r"])*"
    UNALLOWED_CHARACTERS_REGEX = (
        "[^" + "".join(map(re.escape, ALLOWED_CHARACTERS)) + "]"
    )

    # ====================== #
    #    Helper Functions    #
    # ====================== #

    def delete_unallowed_characters(self, text: str) -> str:
        text = re.sub(self.UNALLOWED_CHARACTERS_REGEX, " ", text)

        return text

    def normalize_characters(self, text: str) -> str:
        text = text.translate(self.CHARACTER_TRANSLATOR)
        return text

    # ================ #
    #    initialize    #
    # ================ #

    def __get_vocab(self):
        vocab = Vocab(
            language=f"^{self.name()}$",  # r"^hi$"
            country=r"[a-z]+",
            organization=r"[a-z]+",
            part_number=r"[0-9]+",
            data_root_dir=Assets.ROOT_DIR,
            arg_is_regex=True,
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
            acronym_periods=self.ACRONYM_PERIODS+["."],
            # spelled out english letters (acronyms)
            non_sentence_end_words=[
                "बी",  # B
                "सी",  # C
                "एफ",  # F
                "एच",  # H
                "जे",  # J
                "एल",  # L
                "एम",  # M
                "एन",  # N
                "एस",  # S
                "डब्ल्यू",  # W
                "एक्स",  # X
            ],
            tokenized_word_sense_pattern=[self.WORD_REGEX, r"\(", [r"नाम"], r"\)"],
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
            Rule.from_pattern(r"^[A-Z]{2,8}$", Tags.ACRONYM, 4),
            # e.g. 2002-02-20
            Rule.from_pattern(r"^\d{4}-\d{2}-\d{2}$", Tags.DATE, 4),
            # e.g. 09:30:25.333
            Rule.from_pattern(r"^\d+(?::\d+)?(?::\d+(?:\.\d+)?)$", Tags.TIME, 4),
            # e.g. John, Doe(name)
            Rule(
                lambda token: token in self.vocab.person_names
                or token.endswith("(नाम)"),
                Tags.NAME,
                2,
            ),
            # e.g. Cow, airplane, 1
            Rule(
                lambda token: (token.lower() in self.vocab.supported_tokens),
                Tags.SUPPORTED_WORD,
                3,
            ),
            # e.g. forty-five, 45
            Rule(
                lambda token: (
                    bool(re.match(r"^\d+(?:\.\d+)?$", token))
                    or token in self.vocab.words_to_numbers
                ),
                Tags.NUMBER,
                4,
            ),
            # e.g. "सोना" -> ["सोना(gold)", "सोना(sleep)"]
            Rule(
                lambda token: token.lower() in self.vocab.ambiguous_to_unambiguous,
                Tags.AMBIGUOUS,
                2,
            ),
        ]
        return tagging_rules

    # ================== #
    #    Romanization    #
    # ================== #
    # https://www.loc.gov/catdir/cpso/romanization/hindi.pdf
    # https://www.loc.gov/catdir/cpso/romanization/hindi-1997.pdf
    ROMANIZATION_MAP_VOWELS_AND_DIPHTHONGS = {
        "अ": "a",
        "आ": "ā",
        "ा": "a",  # अा
        "इ": "i",
        "ि": "i",  # अि
        "ई": "ī",
        "ी": "ī",  # अी
        "उ": "u",
        "ु": "u",  # अु
        "ऊ": "ū",
        "ू": "ū",  # अू
        "ऋ": "r",
        "ृ": "r",  # अृ
        "ॠ": "r̄",
        "ॄ": "r̄",  # अॄ
        "ऌ": "l",
        "ऄ": "ĕ",
        "ॆ": "ĕ",  # अॆ
        "ए": "e",
        "े": "e",  # अे
        "ॲ": "ê",
        "ॅ": "ê",  # अॅ
        # "": "ăi",
        # "": "ăi",
        "ऐ": "ai",
        "ै": "ai",  # अै
        "ऒ": "ŏ",
        "ॊ": "ŏ",  # ऒ
        "ओ": "o",
        # ! "आे": "o",
        "ो": "o",  # ओ
        "ऑ": "ô",
        "ॉ": "ô",  # ऑ
        "औ": "au",
        "ौ": "au",  # औ
        "ऎ": "ĕ",  # ! not in PDF
    }
    ROMANIZATION_MAP_CONSONANTS_GUTTURALS = {
        "क": "k",
        "क़": "q",
        "ख": "kh",
        "ख़": "k̲h̲",
        "ग": "g",
        "ग़": "g̲h̲",
        "घ": "gh",
        "घ़": "g̲̲h̲̲",
        "ङ": "ngh",  # ? different from PDF
    }
    ROMANIZATION_MAP_CONSONANTS_PALATAS = {
        "च": "ch",  # ? different from PDF
        "छ": "chh",  # ? different from PDF
        "ज": "j",
        "ज़": "z",
        "झ": "jh",
        "ञ": "ñ",
    }
    ROMANIZATION_MAP_CONSONANTS_CEREBRALS = {
        "ट": "ṭ",
        "ट़": "t̤",
        "ठ": "ṭh",
        "ड": "ḍ",
        "ड़": "ṛ",
        "ढ": "ḍh",
        "ढ़": "ṛh",
        "ण": "ṇ",
    }
    ROMANIZATION_MAP_CONSONANTS_DENTALS = {
        "त": "t",
        "थ": "th",
        "द": "d",
        "ध": "dh",
        "न": "n",
    }
    ROMANIZATION_MAP_CONSONANTS_LABIALS = {
        "प": "p",
        "फ": "ph",
        "फ़": "f",
        "ब": "b",
        "भ": "bh",
        "म": "m",
    }
    ROMANIZATION_MAP_CONSONANTS_SEMIVOWELS = {
        "य": "y",
        "र": "r",
        "ल": "l",
        "व": "v",
    }
    ROMANIZATION_MAP_CONSONANTS_SIBILANTS = {
        "श": "sh",  # ? different from PDF
        "ष": "s",  # ? different from PDF
        "स": "s",
        "स़": "s̤",
    }
    ROMANIZATION_MAP_CONSONANTS_ASPIRATE = {
        "ह": "h",
        "ह़": "h̤",
    }
    ROMANIZATION_MAP = {
        **ROMANIZATION_MAP_VOWELS_AND_DIPHTHONGS,
        **ROMANIZATION_MAP_CONSONANTS_GUTTURALS,
        **ROMANIZATION_MAP_CONSONANTS_PALATAS,
        **ROMANIZATION_MAP_CONSONANTS_CEREBRALS,
        **ROMANIZATION_MAP_CONSONANTS_DENTALS,
        **ROMANIZATION_MAP_CONSONANTS_LABIALS,
        **ROMANIZATION_MAP_CONSONANTS_SEMIVOWELS,
        **ROMANIZATION_MAP_CONSONANTS_SIBILANTS,
        **ROMANIZATION_MAP_CONSONANTS_ASPIRATE,
        # === Diacritics === #
        "ं": "n",
        "ँ": "m̐",
        "ः": "ḥ",
        "्": "",
        "ऽ": "'",
        # === Numbers === #
        "०": "0",
        "१": "1",
        "२": "2",
        "३": "3",
        "४": "4",
        "५": "5",
        "६": "6",
        "७": "7",
        "८": "8",
        "९": "9",
        # === Punctuation === #
        "।": ".",
        "॥": ".",
        "॰": ".",
    }
    ROMANIZATION_CHARACTER_TRANSLATOR = {
        ord(h): r for h, r in ROMANIZATION_MAP.items() if len(h) == 1
    }
    NGRAM_ROMANIZATION_MAP = {
        **{ng: r for ng, r in ROMANIZATION_MAP.items() if len(ng) > 1},
        r"(?<=" + "|".join(ROMANIZATION_MAP_CONSONANTS_LABIALS) + ")ं": "m",
        r"(?<="
        + "|".join(
            (2 - len(c)) * "." + c
            for c in sorted(
                list(ROMANIZATION_MAP_CONSONANTS_GUTTURALS)
                + list(ROMANIZATION_MAP_CONSONANTS_PALATAS)
                + list(ROMANIZATION_MAP_CONSONANTS_CEREBRALS)
                + list(ROMANIZATION_MAP_CONSONANTS_DENTALS)
            )
        )
        + ")ँ": "n",
    }
