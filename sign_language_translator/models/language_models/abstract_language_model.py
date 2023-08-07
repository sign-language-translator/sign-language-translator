"""This module provides an abstract base class for language models,
which can be used to sample the next token based on a given context.
It defines the common interface and methods that should be implemented by any language model.

Classes:
    LanguageModel: An abstract base class for language models.
"""

from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple


class LanguageModel(ABC):
    """Abstract Base Class for Language Models

    LanguageModel is an abstract base class that defines the common interface and methods for language models. It provides functionality for sampling the next token based on the given context.

    Attributes:
    - unknown_token (str): The token representation used for unknown or out-of-vocabulary tokens.
    - name (str): The name of the language model (optional).

    Methods:
    - next(self, context: Iterable) -> Tuple[Any, float]: Abstract method that should be implemented
        by subclasses to generate the next token and provide its probability.
    - next_all(self, context: Iterable) -> Tuple[Iterable[Any], Iterable[float]]: Abstract method
        that should be implemented by subclasses to return all next tokens and their probabilities.
    """

    def __init__(self, unknown_token="<unk>", name=None) -> None:
        super().__init__()
        self.unknown_token = unknown_token
        self.name = name

    @abstractmethod
    def next(self, context: Iterable) -> Tuple[Any, float]:
        """Generates the next token based on the given context and also returns its probability.

        Args:
            context (Iterable): A piece of sequence used as the context
                for generating the next token.

        Returns:
            Tuple[Any, float]: The next token and its associated probability.
                Token has the same type as the items in the context iterable.
        """

    @abstractmethod
    def next_all(self, context: Iterable) -> Tuple[Iterable[Any], Iterable[float]]:
        """Computes probabilities for all next tokens based on the given context and returns them both.

        Args:
            context (Iterable): A piece of sequence used as the context
                for generating the next tokens.

        Returns:
            Tuple[Iterable[Any], Iterable[float]]: All next tokens and their probabilities.
                The tokens have the same type as the items in the context iterable.
        """

    def __str__(self) -> str:
        return (
            f'name="{self.name}", ' if self.name else ""
        ) + f'unk_tok="{self.unknown_token}"'

    def __repr__(self) -> str:
        return str(self.__class__).split()[1][:-1] + "\n\n" + self.__str__()
