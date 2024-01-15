"""This module defines the ConcatenativeSynthesis class, which represents a rule based model for translating text to sign language."""

from __future__ import annotations

import random
from enum import Enum
from typing import List, Type

from sign_language_translator.config.enums import SignFormats
from sign_language_translator.languages import get_sign_language, get_text_language
from sign_language_translator.languages.sign import SignLanguage
from sign_language_translator.languages.text import TextLanguage
from sign_language_translator.models.text_to_sign.t2s_model import TextToSignModel
from sign_language_translator.vision._utils import get_sign_wrapper_class
from sign_language_translator.vision.sign.sign import Sign


class ConcatenativeSynthesis(TextToSignModel):
    """A class representing a Rule-Based model for translating text to sign language
    by concatenating sign language videos.
    """

    def __init__(
        self,
        text_language: str | TextLanguage | Enum,
        sign_language: str | SignLanguage | Enum,
        sign_format: str | Type[Sign],
    ) -> None:
        """
        Args:
            text_language (str | TextLanguage | Enum): (source) The text language processor object or its identifier. (e.g. "urdu" or slt.languages.text.Urdu)
            sign_language (str | SignLanguage | Enum): (target) The sign language processor object or its identifier. (e.g. "pk-sl" or slt.languages.sign.PakistanSignLanguage)
            sign_format (str | Type[Sign]): (format) The sign features used for mapping labels to sign features. (e.g. "video" or slt.Video)
        """
        self._text_language = self.__get_text_language_object(text_language)
        self._sign_language = self.__get_sign_language_object(sign_language)
        self._sign_format = self.__get_sign_format_class(sign_format)

    @property
    def text_language(self):
        """An object of TextLanguage class or its child that defines preprocessing, tokenization & other NLP functions."""
        return self._text_language

    @text_language.setter
    def text_language(self, text_language: str | TextLanguage | Enum):
        self._text_language = self.__get_text_language_object(text_language)

    @property
    def sign_language(self):
        """An object of SignLanguage class or its child that defines the mapping rules & grammar."""
        return self._sign_language

    @sign_language.setter
    def sign_language(self, sign_language: str | SignLanguage | Enum):
        self._sign_language = self.__get_sign_language_object(sign_language)

    @property
    def sign_format(self):
        """
        The format of the sign language (e.g. slt.Vision.sign.sign.Sign or subclass).

        Class that wraps the sign language features e.g. videos or images. This class can load the signs from dataset and concatenate its objects
        """

        return self._sign_format

    @sign_format.setter
    def sign_format(self, sign_format: str | Type[Sign]):
        self._sign_format = self.__get_sign_format_class(sign_format)

    def translate(self, text: str, *args, **kwargs) -> Sign:
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

    def __get_text_language_object(
        self, text_language: str | TextLanguage | Enum
    ) -> TextLanguage:
        if isinstance(text_language, (str, Enum)):
            return get_text_language(text_language)
        if isinstance(text_language, TextLanguage):
            return text_language
        raise TypeError(f"Expected str or TextLanguage, got {type(text_language) = }.")

    def __get_sign_language_object(
        self, sign_language: str | SignLanguage | Enum
    ) -> SignLanguage:
        if isinstance(sign_language, (str, Enum)):
            return get_sign_language(sign_language)
        if isinstance(sign_language, SignLanguage):
            return sign_language
        raise TypeError(f"Expected str or SignLanguage, got {type(sign_language) = }.")

    def __get_sign_format_class(self, sign_format: str | Type[Sign]) -> Type[Sign]:
        if isinstance(sign_format, str | Enum):
            return get_sign_wrapper_class(sign_format)
        if isinstance(sign_format, type) and issubclass(sign_format, Sign):
            return sign_format
        raise TypeError(f"Expected str or type[Sign], got {sign_format = }.")
