from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Set, Tuple, Union


class TextLanguage(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        """Returns the name of the language used everywhere else in datasets."""

    @staticmethod
    @abstractmethod
    def word_regex() -> str:
        """Returns a regular expression that matches words in this language."""

    @staticmethod
    @abstractmethod
    def allowed_characters() -> Set[str]:
        """Returns a set of all allowed characters in the language."""

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Preprocesses text before tokenization"""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Break apart text into words or phrases"""

    @abstractmethod
    def sentence_tokenize(self, text: str) -> List[str]:
        """Break text into sentences."""

    @abstractmethod
    def detokenize(self, tokens: Iterable[str]) -> str:
        """Joins tokens back into text."""

    @abstractmethod
    def tag(self, tokens: Union[str, Iterable[str]]) -> List[Tuple[str, Any]]:
        """Classify the tokens and mark them with appropriate tags."""

    @abstractmethod
    def get_tags(self, tokens: Union[str, Iterable[str]]) -> List[Any]:
        """Get the classifications of all tokens in the form of a sequence of tags"""

    @abstractmethod
    def get_word_senses(self, tokens: Union[str, Iterable[str]]) -> List[List[str]]:
        """Get all known meanings of the ambiguous word."""

    # embed/similar
    # all_tags