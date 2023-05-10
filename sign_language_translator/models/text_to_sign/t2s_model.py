from abc import ABC, abstractmethod


class TextToSignModel(ABC):
    @property
    @abstractmethod
    def tokenizer(self):
        pass

    @property
    @abstractmethod
    def sign_language(self):
        pass

    @property
    @abstractmethod
    def text_language(self):
        pass

    @property
    @abstractmethod
    def sign_features(self):
        pass

    @abstractmethod
    def translate(self, text: str): # -> VideoFeatures
        pass