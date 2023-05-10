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

    @abstractstaticmethod
    def delete_unallowed_characters(str) -> str:
        pass

    @abstractmethod
    def preprocess(self, text: str) -> str:
        pass

    @abstractmethod
    def is_word_supported(self, word: str) -> bool:
        pass

    @abstractmethod
    def get_word_contexts(self, word: str) -> List[str]:
        pass

