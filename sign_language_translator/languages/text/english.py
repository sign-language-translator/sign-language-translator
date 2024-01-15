from typing import Any, Iterable, List, Set, Tuple, Union

from sign_language_translator.config.enums import TextLanguages
from sign_language_translator.languages.text.text_language import TextLanguage
from sign_language_translator.languages.vocab import Vocab


class English(TextLanguage):
    """NLP class for English text. Extends `slt.languages.text.TextLanguage` class.

    English is originally a West Germanic language and potentially an international language in the 21st century. English uses the Latin script, which consists of 26 letters and is written from left to right.
    See unicode details at: https://unicode.org/charts/PDF/U0000.pdf
    """

    @staticmethod
    def name() -> str:
        raise NotImplementedError
        # return TextLanguages.ENGLISH.value

    @classmethod
    def token_regex(cls) -> str:
        raise NotImplementedError

    @classmethod
    def allowed_characters(cls) -> Set[str]:
        raise NotImplementedError

    def __init__(self) -> None:
        raise NotImplementedError
        # self.vocab = Vocab(
        #     language=f"^{self.name()}$",  # r"^en$"
        #     country=r"[a-z]+",
        #     organization=r"[a-z]+",
        #     part_number=r"[0-9]+",
        #     data_root_dir=Assets.ROOT_DIR,
        #     arg_is_regex=True,
        # )

    def preprocess(self, text: str) -> str:
        raise NotImplementedError

    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    def sentence_tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    def detokenize(self, tokens: Iterable[str]) -> str:
        raise NotImplementedError

    def tag(self, tokens: Union[str, Iterable[str]]) -> List[Tuple[str, Any]]:
        raise NotImplementedError

    def get_tags(self, tokens: Union[str, Iterable[str]]) -> List[Any]:
        raise NotImplementedError

    def get_word_senses(self, tokens: Union[str, Iterable[str]]) -> List[List[str]]:
        raise NotImplementedError
