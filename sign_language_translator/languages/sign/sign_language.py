"""contains abstract class for sign languages which map spoken language text to signs using rules"""

import enum
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from sign_language_translator.utils import PrintableEnumMeta


class SignLanguage(ABC):
    """This abstract class defines the structure and methods required for mapping
    spoken language text to signs in sign languages using rule-based approaches.

    Attributes:
        Keys (enum.Enum): Enumerates all keys that are used in a sign dict.

    Methods:
        name: Returns the name of the sign language.
        tokens_to_sign_dicts: Converts tokens to signs based on rules and returns a list of sign dictionaries.
        restructure_sentence: Restructures a sentence by adjusting grammar, dropping meaningless words, and normalizing synonyms.
        _make_equal_weight_sign_dict: Creates a sign dictionary with equal weights for the provided signs.
    """

    class SignDictKeys(enum.Enum, metaclass=PrintableEnumMeta):
        """Enumerates all keys that are used in a sign dict.

        Attributes:
            SIGNS (str): key for the 'signs' field in the sign dict mapping to list of sequence of video names.
            WEIGHTS (str): key for the 'weights' field in the sign dict mapping to the usage frequency of a video sequence.
        """

        SIGNS = "signs"
        WEIGHTS = "weights"

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Returns the name of the sign language."""

    @abstractmethod
    def tokens_to_sign_dicts(
        self,
        tokens: Iterable[str],
        tags: Optional[Iterable[Any]] = None,
        contexts: Optional[Iterable[Any]] = None,
    ) -> List[Dict[str, Union[List[List[str]], List[float]]]]:
        """Converts tokens to signs based on rules and returns a list of sign dictionaries.

        Args:
            tokens (Iterable[str]): Input tokens to be converted to signs.
            tags (Iterable[Any], optional): Additional tags associated with the tokens. Defaults to None.
            contexts (Iterable[Any], optional): Additional contexts associated with the tokens. Defaults to None.

        Returns:
            List[Dict[str, List[List[str]] | List[float]]]: A list of sign dictionaries, where each dictionary contains
                the 'signs' field mapping to a list of sign sequences and the 'weights' field mapping to the usage
                frequency of each sign sequence.
                e.g. "word" -> [{"signs": [[sign_1, sign_2], [alternate_1]], "weights": [10, 5]}, ...]
        """

    @abstractmethod
    def restructure_sentence(
        self,
        sentence: Iterable[str],
        tags: Optional[Iterable[Any]] = None,
        contexts: Optional[Iterable[Any]] = None,
    ) -> Tuple[Iterable[str], Iterable[Any], Iterable[Any]]:
        """Restructures a sentence by changing the grammar,
        removing stopwords, spaces & punctuation, and modifying token contents.

        Args:
            sentence (Iterable[str]): Input sentence to be restructured.
            tags (Iterable[Any], optional): Additional tags associated with the sentence. Defaults to None.
            contexts (Iterable[Any], optional): Additional contexts associated with the sentence. Defaults to None.

        Returns:
            Tuple[Iterable[str], Iterable[Any], Iterable[Any]]: The restructured sentence, associated tags, and contexts.
        """

    def _make_equal_weight_sign_dict(
        self, signs: List[List[str]]
    ) -> Dict[str, Union[List[List[str]], List[float]]]:
        """Creates a sign dictionary with equal weights for the provided signs.

        Args:
            signs (List[List[str]]): List of sign sequences.

        Returns:
            Dict[str, List[List[str]] | List[float]]: A sign dictionary with equal weights for each sign sequence.
        """

        return {
            self.SignDictKeys.SIGNS.value: signs,
            self.SignDictKeys.WEIGHTS.value: [1 / len(signs) for _ in signs],
        }
