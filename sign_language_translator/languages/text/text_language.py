"""
text_language.py
----------------
This module defines the Base NLP class for text format of a spoken language.
It defines the interface for text processing functions needed by the rule-based translator.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union


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

    @staticmethod
    def romanize(
        text: str,
        *args,
        add_diacritics=True,
        character_translation_table: Optional[Dict[int, str]] = None,
        n_gram_map: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """Map characters to phonetically similar characters of the English language.
        Transliteration is useful for readability & simple text-to-speech.
        First maps (n>1)-grams, then unigrams.

        ALA-LC Standardized Romanization Tables (70 languages): https://www.loc.gov/catdir/cpso/roman.html

        Args:
            text (str): Non-English text to be mapped to Latin script.
            add_diacritics (bool, optional): Whether to use diacritics over English characters to help pronunciation. (Rules: 1. The under-dot ' ̣' indicates alternate soft/hard pronunciation of the letter. 2. The over-bar/macron ' ̄' means long pronunciation). Defaults to True.
            character_translation_table (Optional[Dict[int, str]], optional): A dictionary mapping unicode of single characters to their latin equivalent. Defaults to None.
            n_gram_map (Optional[Dict[str, str]], optional): A dictionary mapping bigrams, trigrams or more to their latin equivalent. Keys are expected to be regular expressions. Defaults to None.
        """

        # map (n>1)-grams
        if isinstance(n_gram_map, dict):
            re_operators = re.compile(r"[\+\*\?\|\[\]\{\}\^\$<=\!\(\)]|(\\[bdwWs])")
            for ngram in sorted(  # ToDo: optimize
                n_gram_map.keys(),
                key=lambda x: len(re_operators.sub("", x)),
                reverse=True,
            ):
                text = re.sub(ngram, n_gram_map[ngram], text)

        # map unigrams
        text = text.translate(character_translation_table or {})

        if not add_diacritics:
            text = re.sub("[ ̄ ̣ ̂ ̇ ̲ ̆ ̤ ̃ ́]".replace(" ", ""), "", text)

        return text
