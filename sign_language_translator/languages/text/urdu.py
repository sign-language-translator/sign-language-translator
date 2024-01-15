import re
from string import ascii_uppercase, digits
from typing import Any, Dict, Iterable, List, Set, Tuple, Union

from sign_language_translator.config.assets import Assets
from sign_language_translator.config.enums import TextLanguages
from sign_language_translator.languages.text.text_language import TextLanguage
from sign_language_translator.languages.vocab import Vocab
from sign_language_translator.text.preprocess import (
    remove_space_before_punctuation,
    replace_words,
)
from sign_language_translator.text.tagger import Rule, Tagger, Tags
from sign_language_translator.text.tokenizer import SignTokenizer

__all__ = [
    "Urdu",
]


class Urdu(TextLanguage):
    """NLP class for Urdu text. Extends `slt.languages.text.TextLanguage` class.

    Urdu is an Indo-Aryan language spoken mostly in Pakistan. Urdu uses the Perso-Arabic script, which consists of 46 Alphabets, 10 Digits, 6 Punctuations & 6 Diacritics, and is written from right to left.
    See unicode details at: https://unicode.org/charts/PDF/U0600.pdf
    """

    @staticmethod
    def name() -> str:
        return TextLanguages.URDU.value

    @classmethod
    def token_regex(cls) -> str:
        return cls.NUMBER_REGEX + r"|" + cls.WORD_REGEX

    @classmethod
    def allowed_characters(cls) -> Set[str]:
        return cls.ALLOWED_CHARACTERS

    def __init__(self) -> None:
        # TODO: args to filter dataset
        self.vocab = Vocab(
            language=r"^ur$",
            country=r".+",
            organization=r".+",
            part_number=r"[0-9]+",
            data_root_dir=Assets.ROOT_DIR,
            arg_is_regex=True,
        )

        self.non_sentence_end_tokens = {
            # letters (A.B.C.) & spelled out letters (Ay, Bee, See)
            w.upper()
            for wc in self.vocab.supported_words_with_word_sense
            for w in [self.vocab.remove_word_sense(wc)]
            if (("double-handed-letter)" in wc) and (not w.isascii()))
            or (len(w) == 1 and w.isalpha())
        }

        self.tokenizer = SignTokenizer(
            word_regex=self.token_regex(),
            compound_words=(
                self.vocab.supported_words
                | self.vocab.supported_words_with_word_sense
                | set(self.vocab.words_to_numbers.keys())
                | set(self.vocab.person_names)
            ),  # TODO: | one-hundred twenty-three (\d[ \d]*): ["100", "23"] --> ["123"]
            end_of_sentence_tokens=self.END_OF_SENTENCE_MARKS,
            full_stops=self.FULL_STOPS,
            non_sentence_end_words=self.non_sentence_end_tokens,
            tokenized_word_sense_pattern=[self.WORD_REGEX, r"\(", [r"نام"], r"\)"],
        )

        # :TODO: {<unk>: id_}, def token_to_id, tokenize(..., as_input_ids = True),

        self.tagging_rules = [
            # e.g. " "
            Rule.from_pattern(r"^\s$", Tags.SPACE, 5),
            # e.g. "," "."
            Rule.from_pattern(
                r"^[" + "".join(map(re.escape, self.PUNCTUATION)) + r"]$",
                Tags.PUNCTUATION,
                5,
            ),
            # e.g. "word"
            Rule.from_pattern("^" + self.token_regex() + "$", Tags.WORD, 5),
            # e.g. COVID
            Rule.from_pattern(r"^[A-Z]{2,7}$", Tags.ACRONYM, 4),
            # e.g. 2002-02-20
            Rule.from_pattern(r"^\d{4}-\d{2}-\d{2}$", Tags.DATE, 4),
            # e.g. 09:30:25.333
            Rule.from_pattern(r"^\d+(?::\d+)?(?::\d+(?:\.\d+)?)$", Tags.TIME, 4),
            # e.g. John, Doe(name)
            Rule(
                lambda token: token in self.vocab.person_names
                or token.endswith("(نام)"),
                Tags.NAME,
                2,
            ),
            # e.g. Cow, airplane, 1
            Rule(
                lambda token: (
                    token.lower() in self.vocab.supported_words
                    or token.lower() in self.vocab.supported_words_with_word_sense
                ),
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
            # e.g. "میں" -> ["میں(i)", "میں(in)"]
            Rule(
                lambda token: token in self.vocab.ambiguous_to_unambiguous,
                Tags.AMBIGUOUS,
                2,
            ),
        ]
        self.tagger = Tagger(
            rules=self.tagging_rules,
            default=Tags.DEFAULT,
        )

    def preprocess(self, text: str) -> str:
        # TODO: optimize (especially regex)
        text = self.character_normalize(text)

        # spell fix
        text = replace_words(
            text,
            word_map=self.vocab.misspelled_to_correct,  # :TODO: split joint words
            word_regex=self.token_regex(),
        )
        text = self.delete_unallowed_characters(text)
        text = re.sub(r"[۔\.][۔\. ]+[\.۔]", "۔۔۔", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = remove_space_before_punctuation(text, self.PUNCTUATION)
        text = text.strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(
            text, join_compound_words=True, join_word_sense=True
        )
        return tokens

    def sentence_tokenize(self, text: str) -> List[str]:
        sentences = self.tokenizer.sentence_tokenize(text)
        if len(sentences) > 1:
            sentences[1:] = [sentence.lstrip() for sentence in sentences[1:]]
            sentences[:-1] = [sentence.rstrip() for sentence in sentences[:-1]]
        # sentences = [sen for sen in sentences if sen]
        return sentences

    def detokenize(self, tokens: Iterable[str]) -> str:
        text = self.tokenizer.detokenize(tokens)
        return text

    def tag(self, tokens: Union[str, Iterable[str]]) -> List[Any]:
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
            self.vocab.ambiguous_to_unambiguous.get(token, []) for token in tokens
        ]

        return word_senses

    # ====================== #
    #    Character Groups    #
    # ====================== #

    UNICODE_RANGE: Tuple[int, int] = (1536, 1791)

    FULL_STOPS: List[str] = [".", "۔"]
    QUESTION_MARKS: List[str] = ["?", "؟"]
    END_OF_SENTENCE_MARKS: List[str] = FULL_STOPS + QUESTION_MARKS + ["!"]

    PUNCTUATION: List[str] = END_OF_SENTENCE_MARKS + [",", "٫", "،", "؛"]

    QUOTATION_MARKS = """ ' " ” “ ’ ‘ """.split()
    BRACKETS: List[str] = ["(", ")"]
    SYMBOLS: List[str] = PUNCTUATION + QUOTATION_MARKS + BRACKETS + [" ", "-"]

    PUNCTUATION_REGEX = r"[" + "".join([re.escape(punc) for punc in PUNCTUATION]) + r"]"
    DIACRITICS = str(" ٍ ً ٰ َ ُ ِ ").split()
    EXTRA_DIACRITICS = str(" ؐ  ؑ ؒ ؓ").split()
    WORD_REGEX = r"[\w" + "".join(DIACRITICS) + r"]+"
    # TODO: r"[[^\W\d_]"+ "".join(DIACRITICS) + r"]+"

    NUMBER_REGEX = r"\d+(?:[\.:]\d+)*"

    CHARACTER_TO_WORD = {
        "ﷲ": "اللہ",
        "ﷺ": "صلی اللہ علیہ وسلم",
        "﷽": "بسم اللہ الرحمن الرحیم",
        "–": "-",
        "—": "-",
        "−": "-",
        "⋯": "...",
    }

    def delete_unallowed_characters(self, text: str) -> str:
        text = re.sub(self.UNALLOWED_CHARACTERS_REGEX, " ", text)

        return text

    # functions to preprocess specific datasets
    @staticmethod
    def poetry_preprocessor(text: str) -> str:
        text = ("؛ ").join(
            [
                line.strip("() '\"\t")
                for line in text.splitlines()
                if len(re.findall(Urdu.WORD_REGEX, line)) > 1
            ]
        )
        MISSPELLED_TO_CORRECT = {
            "مرا": "میرا",
            "مری": "میری",
            "مري": "میری",
            "مرے": "میرے",
        }
        text = replace_words(
            text,
            word_map=MISSPELLED_TO_CORRECT,
            word_regex=Urdu.WORD_REGEX,
        )

        return text

    @staticmethod
    def passage_preprocessor(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    @staticmethod
    def wikipedia_preprocessor(text: str) -> str:
        text = text.strip(". !\"'\n\t")
        return text

    if True or "UrduHack":
        # Start of the code borrowed from "UrduHack/normalization/character.py"
        # """Following Dictionaries and code are copied from UrduHack package
        # Source Repo URL: https://github.com/urduhack/urduhack
        # Source Repo URL: https://github.com/urduhack/urdu-characters"""

        # Maps correct Urdu characters to list of visually similar non-urdu characters
        CORRECT_URDU_CHARACTERS_TO_INCORRECT: Dict[str, List[str]] = {
            "آ": ["ﺁ", "ﺂ"],
            "أ": ["ﺃ"],
            "ا": ["ﺍ", "ﺎ"],
            "ب": ["ﺏ", "ﺐ", "ﺑ", "ﺒ"],
            "پ": ["ﭖ", "ﭘ", "ﭙ"],
            "ت": ["ﺕ", "ﺖ", "ﺗ", "ﺘ"],
            "ٹ": ["ﭦ", "ﭧ", "ﭨ", "ﭩ"],
            "ث": ["ﺛ", "ﺜ", "ﺚ"],
            "ج": ["ﺝ", "ﺞ", "ﺟ", "ﺠ"],
            "ح": ["ﺡ", "ﺣ", "ﺤ", "ﺢ"],
            "خ": ["ﺧ", "ﺨ", "ﺦ"],
            "د": ["ﺩ", "ﺪ"],
            "ذ": ["ﺬ", "ﺫ"],
            "ر": ["ﺭ", "ﺮ"],
            "ز": ["ﺯ", "ﺰ"],
            "س": ["ﺱ", "ﺲ", "ﺳ", "ﺴ"],
            "ش": ["ﺵ", "ﺶ", "ﺷ", "ﺸ"],
            "ص": ["ﺹ", "ﺺ", "ﺻ", "ﺼ"],
            "ض": ["ﺽ", "ﺾ", "ﺿ", "ﻀ"],
            "ط": ["ﻃ", "ﻄ"],
            "ظ": ["ﻅ", "ﻇ", "ﻈ"],
            "ع": ["ﻉ", "ﻊ", "ﻋ", "ﻌ"],
            "غ": ["ﻍ", "ﻏ", "ﻐ"],
            "ف": ["ﻑ", "ﻒ", "ﻓ", "ﻔ"],
            "ق": ["ﻕ", "ﻖ", "ﻗ", "ﻘ"],
            "ل": ["ﻝ", "ﻞ", "ﻟ", "ﻠ"],
            "م": ["ﻡ", "ﻢ", "ﻣ", "ﻤ"],
            "ن": ["ﻥ", "ﻦ", "ﻧ", "ﻨ"],
            "چ": ["ﭺ", "ﭻ", "ﭼ", "ﭽ"],
            "ڈ": ["ﮈ", "ﮉ"],
            "ڑ": ["ﮍ", "ﮌ"],
            "ژ": ["ﮋ"],
            "ک": ["ﮎ", "ﮏ", "ﮐ", "ﮑ", "ﻛ", "ك"],
            "گ": ["ﮒ", "ﮓ", "ﮔ", "ﮕ"],
            "ں": ["ﮞ", "ﮟ"],
            "و": ["ﻮ", "ﻭ", "ﻮ"],
            "ؤ": ["ﺅ"],
            "ھ": ["ﮪ", "ﮬ", "ﮭ", "ﻬ", "ﻫ", "ﮫ"],
            "ہ": ["ﻩ", "ﮦ", "ﻪ", "ﮧ", "ﮩ", "ﮨ", "ه"],
            "ۂ": [],
            "ۃ": ["ة"],
            "ء": ["ﺀ"],
            "ی": ["ﯼ", "ى", "ﯽ", "ﻰ", "ﻱ", "ﻲ", "ﯾ", "ﯿ", "ي"],
            "ئ": ["ﺋ", "ﺌ"],
            "ے": ["ﮮ", "ﮯ", "ﻳ", "ﻴ"],
            "ۓ": [],
            "۰": ["٠"],
            "۱": ["١"],
            "۲": ["٢"],
            "۳": ["٣"],
            "۴": ["٤"],
            "۵": ["٥"],
            "۶": ["٦"],
            "۷": ["٧"],
            "۸": ["٨"],
            "۹": ["٩"],
            "۔": [],
            "؟": [],
            "٫": [],
            "،": [],
            "لا": ["ﻻ", "ﻼ"],
            # "": ["ـ"],
        }

        # Maps (character + diacritic) to single characters (beware RTL text rendering)
        SPLIT_TO_COMBINED_CHARACTERS: Dict[str, str] = {
            "آ": "آ",
            "أ": "أ",
            "ؤ": "ؤ",
            "ۂ": "ۂ",
            "یٔ": "ئ",
            "ۓ": "ۓ",
        }

        # Convert the dictionaries to a useable format
        CHARACTER_TRANSLATOR = {ord(c): w for c, w in CHARACTER_TO_WORD.items()} | {
            ord(non_urdu): urdu
            for urdu, others in CORRECT_URDU_CHARACTERS_TO_INCORRECT.items()
            for non_urdu in others
        }
        COMBINE_CHARACTERS_REGEX = r"|".join(SPLIT_TO_COMBINED_CHARACTERS.keys())
        DIACRITICS_REGEX = r"|".join(DIACRITICS)

        @staticmethod
        def character_normalize(text: str) -> str:
            """Replace characters that are rendered the same as Urdu characters in common fonts but actually belong to foreign unicode character ranges by Urdu characters.

            Args:
                text (str): a piece of urdu text that may contain foreign symbols

            Returns:
                str: normalized urdu text
            """

            text = text.translate(Urdu.CHARACTER_TRANSLATOR)
            text = re.sub(
                Urdu.COMBINE_CHARACTERS_REGEX,
                lambda match: Urdu.SPLIT_TO_COMBINED_CHARACTERS[match.group()],
                text,
            )

            return text

        @staticmethod
        def remove_diacritics(text: str) -> str:
            text = re.sub(Urdu.DIACRITICS_REGEX, "", text)

            return text

        # End of the code borrowed from UrduHack

    ALLOWED_CHARACTERS = (
        set(CORRECT_URDU_CHARACTERS_TO_INCORRECT.keys())
        | set(DIACRITICS)
        | set(SYMBOLS)
        | set(ascii_uppercase)  # acronyms
        | set(digits)
        | set(EXTRA_DIACRITICS)
        | set("()!.,?/[]{} \n")
    )
    UNALLOWED_CHARACTERS_REGEX = (
        "[^" + "".join(map(re.escape, ALLOWED_CHARACTERS)) + "]"
    )
