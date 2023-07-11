from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Union, Iterable


class TextToSignModel(ABC):
    @abstractproperty
    def sign_language(self):
        pass

    @abstractproperty
    def text_language(self):
        pass

    @abstractproperty
    def sign_features(self):
        pass

    @abstractmethod
    def translate(self, text: Union[str, Iterable[str]]) -> Any: # -> VideoFeatures
        pass

    def __call__(self, text: Union[str, Iterable[str]]) -> Any:
        return self.translate(text)