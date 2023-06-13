from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Any, Iterable, List, Set, Tuple, Union


class TextLanguage(ABC):
    @abstractstaticmethod
    def name() -> str:
        pass

    @abstractstaticmethod
    def word_regex() -> str:
        pass

    @abstractstaticmethod
    def allowed_characters() -> Set[str]:
        pass

    @abstractmethod
    def preprocess(self, text: str) -> str:
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def sentence_tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def detokenize(self, tokens: Iterable[str]) -> str:
        pass

    @abstractmethod
    def tag(self, tokens: Union[str, Iterable[str]]) -> List[Tuple[str, Any]]:
        pass

    @abstractmethod
    def get_tags(self, tokens: Union[str, Iterable[str]]) -> List[Any]:
        pass

    @abstractmethod
    def get_word_senses(self, tokens: Union[str, Iterable[str]]) -> List[List[str]]:
        pass

    # embed/similar
    # all_tags