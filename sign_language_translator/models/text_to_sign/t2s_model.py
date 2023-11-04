from abc import ABC, abstractmethod
from typing import Any, Iterable, Union


class TextToSignModel(ABC):
    @property
    @abstractmethod
    def sign_language(self):
        """The target sign language of the model."""

    @property
    @abstractmethod
    def text_language(self):
        "The source text language of the model."

    @property
    @abstractmethod
    def sign_format(self):
        """The format of the sign language (e.g. slt.Vision.sign.sign.Sign)."""

    @abstractmethod
    def translate(self, text: Union[str, Iterable[str]], *args, **kwargs) -> Any:  # -> VideoFeatures
        """Translate the text to sign language."""

    def __call__(self, text: Union[str, Iterable[str]], *args, **kwargs) -> Any:
        return self.translate(text, *args, **kwargs)
