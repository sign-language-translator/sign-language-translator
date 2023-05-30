import re
from .text_language import TextLanguage
import string
from typing import Any, Iterable, Tuple, Dict, List, Set
from ..vocab import Vocab, BAD_CHARACTERS_REGEX
from ...text.preprocess import replace_words


class Urdu(TextLanguage):
    @classmethod
    def name(cls) -> str:
        return "urdu"

    @classmethod
    def word_regex(cls) -> str:
        return cls.URDU_WORD_REGEX

    @classmethod
    def allowed_characters(cls) -> Set[str]:
        return cls.ALLOWED_CHARACTERS

    def delete_unallowed_characters(self, text: str) -> str:
        text = re.sub(self.UNALLOWED_CHARACTERS_REGEX, " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def __init__(self) -> None:
        self.VOCAB = Vocab(Urdu.name())

        self.NON_SENTENCE_END_TOKENS = {
            w
            for wc in self.VOCAB.supported_words_with_context
            for w in [self.VOCAB.remove_context(wc)]
            if (("double-handed-letter)" in wc) and (not w.isascii()))
            or (len(w) == 1 and w.isalnum())
        }

    def preprocess(self, text: str) -> str:
        text = Urdu.character_normalize(text)

        # spell fix
        text = replace_words(
            text,
            word_map=self.VOCAB.preprocessing_map["misspelled_to_correct"],
            word_regex=self.word_regex(),
        )
        text = Urdu.delete_unallowed_characters(text)
        text = re.sub(r"[۔\.][۔\. ]+[\.۔]", "۔۔۔", text)
        text = re.sub(r"[ \t]+", " ", text)

        return text

    def is_word_supported(self, word: str) -> bool:
        return word in self.VOCAB.supported_words

    def is_word_and_context_supported(self, word: str) -> bool:
        return word in self.VOCAB.supported_words_with_context

    def get_word_contexts(self, word: str) -> List[str]:
        return self.VOCAB.ambiguous_to_context.get(word, [])

    FULL_STOPS: List[str] = [".", "۔"]
    QUESTION_MARKS: List[str] = ["?", "؟"]
    END_OF_SENTENCE_MARKS: List[str] = FULL_STOPS + QUESTION_MARKS + ["!"]

    QUOTATION_MARKS = """ ' " ” “ ’ ‘ """.split()
    PUNCTUATION: List[str] = END_OF_SENTENCE_MARKS + [",", "،"]
    SYMBOLS: List[str] = (
        PUNCTUATION + QUOTATION_MARKS + [" ", "-"] + ["؛"]  # + ["(", ")", "[", "]"]
    )

    PUNCTUATION_REGEX = (
        r"\s+[" + r"|".join([re.escape(punc) for punc in PUNCTUATION]) + r"]"
    )
    URDU_DIACRITICS = " ٍ ً ٰ َ ُ ِ ".replace(" ", "")
    URDU_WORD_REGEX = r"[\w" + URDU_DIACRITICS + r"]+"

    @staticmethod
    def poetry_preprocessor(text: str) -> str:
        text = ("؛ ").join(
            [
                line.strip("() '\"\t")
                for line in text.splitlines()
                if len(re.findall(Urdu.URDU_WORD_REGEX, line)) > 1
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
            word_regex=Urdu.URDU_WORD_REGEX,
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

    CHARACTER_TO_WORD = {
        "ﷲ": "اللہ",
        "ﷺ": "صلی اللہ علیہ وسلم",
        "﷽": "بسم اللہ الرحمن الرحیم",
        "–": "-",
        "—": "-",
        "−": "-",
        "⋯": "...",
    }

    # <UrduHack>
    # Following Dictionaries and code are copied from UrduHack package (https://github.com/urduhack/urduhack)
    # created by Ikram Ali (mrikram1989@gmail.com) to whom the entire Urdu NLP community can't thank enough.
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
    SPLIT_TO_COMBINED_URDU_CHARACTERS: Dict[str, str] = {
        "آ": "آ",
        "أ": "أ",
        "ؤ": "ؤ",
        "ۂ": "ۂ",
        "یٔ": "ئ",
        "ۓ": "ۓ",
    }
    CHARACTER_TRANSLATOR = {ord(c): w for c, w in CHARACTER_TO_WORD.items()} | {
        ord(nu): u
        for u, others in CORRECT_URDU_CHARACTERS_TO_INCORRECT.items()
        for nu in others
    }
    COMBINE_CHARACTERS_REGEX = r"|".join(SPLIT_TO_COMBINED_URDU_CHARACTERS.keys())
    DIACRITICS_REGEX = r"|".join(URDU_DIACRITICS)

    @staticmethod
    def character_normalize(text: str) -> str:
        text = text.translate(Urdu.CHARACTER_TRANSLATOR)
        text = re.sub(
            Urdu.COMBINE_CHARACTERS_REGEX,
            lambda match: Urdu.SPLIT_TO_COMBINED_URDU_CHARACTERS[match.group()],
            text,
        )

        return text

    @staticmethod
    def remove_diacritics(text: str) -> str:
        text = re.sub(Urdu.DIACRITICS_REGEX, "", text)

        return text

    # </UrduHack>

    ALLOWED_CHARACTERS = (
        set(CORRECT_URDU_CHARACTERS_TO_INCORRECT.keys())
        | set(URDU_DIACRITICS)
        | set(SYMBOLS)
        | set(string.ascii_uppercase)  # acronyms
        | set(string.digits)
        | set("()!.,?/[]{} ")
    )
    UNALLOWED_CHARACTERS_REGEX = (
        "[^" + "".join(map(re.escape, ALLOWED_CHARACTERS)) + "]"
    )
