"""This module defines the ConcatenativeSynthesis class, which represents a rule based model for translating text to sign language."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Type

from sign_language_translator.config.enums import SignFormats
from sign_language_translator.languages import get_sign_language, get_text_language
from sign_language_translator.models.text_to_sign.t2s_model import TextToSignModel
from sign_language_translator.vision._utils import get_sign_wrapper_class

if TYPE_CHECKING:
    from sign_language_translator.languages.sign import SignLanguage
    from sign_language_translator.languages.text import TextLanguage
    from sign_language_translator.vision.sign.sign import Sign


class ConcatenativeSynthesis(TextToSignModel):
    """A class representing a Rule-Based model for translating text to sign language
    by concatenating sign language videos.

    Args:
        text_language (str | TextLanguage): (source) The text language or its identifier.
        sign_language (str | SignLanguage): (target) The sign language or its identifier.
        sign_format (str): (format) The sign features used for mapping labels to sign features.
    """

    def __init__(
        self,
        text_language: str | TextLanguage,
        sign_language: str | SignLanguage,
        sign_format: str | Type[Sign],
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
        self._sign_format = (
            get_sign_wrapper_class(sign_format)
            if isinstance(sign_format, str)
            else sign_format
        )

    @property
    def text_language(self):
        """An object of TextLanguage class or its child that defines preprocessing, tokenization & other NLP functions."""
        return self._text_language

    @property
    def sign_language(self):
        """An object of SignLanguage class or its child that defines the mapping rules & grammar."""
        return self._sign_language

    @property
    def sign_format(self):
        """
        The format of the sign language (e.g. slt.Vision.sign.sign.Sign).

        Class that wraps the sign language features e.g. videos or images. This class can load the signs from dataset and concatenate its objects
        """

        return self._sign_format

    def translate(self, text: str) -> Sign:
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

        signs = self._map_labels_to_sign(video_labels)
        # TODO: Trim signs where hand is just being raised from or lowered to the resting position
        sign_language_sentence = self.sign_format.concatenate(signs)

        return sign_language_sentence

    def _map_labels_to_sign(self, labels: List[str]) -> List[Sign]:
        return [
            self.sign_format(self._prepare_resource_name(label)) for label in labels
        ]

    def _prepare_resource_name(self, label, person=None, camera=None, sep="_"):
        if person is not None:
            label = f"{label}{sep}{person}"
        if camera is not None:
            label = f"{label}{sep}{camera}"

        directory = (
            "videos"
            if self.sign_format.name() == SignFormats.VIDEO.value
            else "landmarks"
        )
        name = f"{directory}/{label}.mp4"

        return name
