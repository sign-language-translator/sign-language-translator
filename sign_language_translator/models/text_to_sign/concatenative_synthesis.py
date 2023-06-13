from sign_language_translator.text.tokens import SignTokenizer
import random

from .t2s_model import TextToSignModel

class ConcatenativeSynthesis(TextToSignModel):
    def __init__(
        self,
        text_language,
        sign_language,
        sign_features,
        unknown=None,
    ) -> None:
        self.text_language = text_language
        self.sign_language = sign_language
        self.tokenizer = None
        self.sign_features = sign_features
        self.unknown = unknown
        self.pipeline = []

    def translate(self, sentence: str):
        for func in self.pipeline:
            sentence = func(sentence)

        return sentence