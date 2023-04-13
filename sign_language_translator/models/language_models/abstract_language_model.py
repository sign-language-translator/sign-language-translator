from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple


class LanguageModel(ABC):
    def __init__(self, unknown_token="<unk>", name=None) -> None:
        super().__init__()
        self.unknown_token = unknown_token
        self.name = name

    def __call__(self, context: Iterable) -> Tuple[Iterable, float]:
        """Samples the next token from the model based on the given context.

        Args:
            context (Iterable): A piece of sequence like the training examples.

        Returns:
            Tuple[Any, float]: The next token and its associated probability. Token has the same type as the context iterable.
        """

        return self.next(context)

    @abstractmethod
    def next(self, context: Iterable) -> Tuple[Any, float]:
        pass

    @abstractmethod
    def next_all(self, context: Iterable) -> Tuple[Iterable[Any], Iterable[float]]:
        pass

    def __str__(self) -> str:
        return (
            f'name="{self.name}", ' if self.name else ""
        ) + f'unk_tok="{self.unknown_token}"'

    def __repr__(self) -> str:
        return str(self.__class__).split()[1][:-1] + "\n\n" + self.__str__()
