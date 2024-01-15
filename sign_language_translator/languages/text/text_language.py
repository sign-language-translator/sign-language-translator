from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Set, Tuple, Union


class TextLanguage(ABC):
    """Base NLP class for a language.

    Subclass it and provide the functionality to tokenize text and classify & disambiguate tokens.
    Each token should correspond to a sign language clip.
    """

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Returns the name of the language used everywhere else in datasets."""

    @classmethod
    @abstractmethod
    def token_regex(cls) -> str:
        """Returns a regular expression that matches words in this language."""

    @classmethod
    @abstractmethod
    def allowed_characters(cls) -> Set[str]:
        """Returns a set of all allowed characters in the language."""

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Preprocesses text before tokenization.
        Make sure no different unicode characters are used for the same word.
        Remove unnecessary symbols, spaces, etc."""

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
        """Get all known meanings of the ambiguous words."""

    # embed/similar
    # all_tags
