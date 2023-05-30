from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Set, List


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

    # @abstractstaticmethod
    # def delete_unallowed_characters(str) -> str:
    #     pass

    @abstractmethod
    def preprocess(self, text: str) -> str:
        pass

    @abstractmethod
    def tokenize(self, text: str) -> str:
        pass

    @abstractmethod
    def sentence_tokenize(self, text: str) -> str:
        pass

    @abstractmethod
    def detokenize(self, text: str) -> str:
        pass

    @abstractmethod
    def tag(self, word: str) -> bool:
        pass

    @abstractmethod
    def get_tags(self, word: str) -> bool:
        pass

    # embed/similar
