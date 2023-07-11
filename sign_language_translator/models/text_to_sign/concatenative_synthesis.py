"""This module defines the ConcatenativeSynthesis class, which represents a rule based model for translating text to sign language."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from sign_language_translator.languages import get_sign_language, get_text_language
from sign_language_translator.models.text_to_sign.t2s_model import TextToSignModel

if TYPE_CHECKING:
    from sign_language_translator.languages.sign import SignLanguage
    from sign_language_translator.languages.text import TextLanguage


class ConcatenativeSynthesis(TextToSignModel):
    """A class representing a Rule-Based model for translating text to sign language
    by concatenating sign language videos.

    Args:
        text_language (str | TextLanguage): (source) The text language or its identifier.
        sign_language (str | SignLanguage): (target) The sign language or its identifier.
        sign_features (str): (format) The sign features used for mapping labels to sign features.
    """

    def __init__(
        self,
        text_language: str | TextLanguage,
        sign_language: str | SignLanguage,
        sign_features: str,
    ) -> None:
        self._text_language = (
            get_text_language(text_language)
            if isinstance(text_language, str)
            else text_language
        )
        self._sign_language = (
            get_sign_language(sign_language)
            if isinstance(sign_language, str)
            else sign_language
        )
        self._sign_features = sign_features  # TODO: use feature loader

    @property
    def text_language(self):
        return self._text_language

    @property
    def sign_language(self):
        return self._sign_language

    @property
    def sign_features(self):
        return self._sign_features

    def translate(self, text: str):
        """
        Translate text to sign language.

        Args:
            text: The input text to be translated.

        Returns:
            The translated sign language sentence.

        """

        sign_language_sentence = None
        video_labels = []

        text = self.text_language.preprocess(text)
        sentences = self.text_language.sentence_tokenize(text)
        for sentence in sentences:
            tokens = self.text_language.tokenize(sentence)
            tags = self.text_language.get_tags(tokens)

            tokens, tags, contexts = self.sign_language.restructure_sentence(
                tokens, tags=tags
            )
            sign_dicts = self.sign_language.tokens_to_sign_dicts(
                tokens, tags=tags, contexts=contexts
            )

            video_labels.extend(
                [
                    label
                    for sign_dict in sign_dicts
                    for label in random.choices(
                        sign_dict[self.sign_language.SignDictKeys.SIGNS.value],
                        weights=sign_dict[self.sign_language.SignDictKeys.WEIGHTS.value],  # type: ignore
                        k=1,
                    )[0]
                ]
            )

        # TODO: map labels to sign features and concatenate
        sign_language_sentence = video_labels

        return sign_language_sentence
