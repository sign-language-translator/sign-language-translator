"""utility functions for language objects"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from sign_language_translator.config.enums import (
    SignLanguages,
    TextLanguages,
    normalize_short_code,
)
from sign_language_translator.languages.sign import PakistanSignLanguage
from sign_language_translator.languages.text import English, Hindi, Urdu

if TYPE_CHECKING:
    from enum import Enum

    from sign_language_translator.languages.sign import SignLanguage
    from sign_language_translator.languages.text import TextLanguage


# TODO: AbstractFactory to store str to class mappings
def get_text_language(language_name: Union[str, Enum]) -> TextLanguage:
    """
    Retrieves a TextLanguage object based on the provided language name.

    Args:
        language_name (str): The name of the language.

    Returns:
        TextLanguage: An instance of the TextLanguage class corresponding to the provided language name.

    Raises:
        ValueError: If no TextLanguage class is known for the provided language name.
    """

    code_to_class = {
        TextLanguages.URDU.value: Urdu,
        TextLanguages.HINDI.value: Hindi,
        TextLanguages.ENGLISH.value: English,
    }

    class_ = code_to_class.get(normalize_short_code(language_name), None)
    if class_:
        return class_()  # constructor called

    # Unknown
    raise ValueError(f"no text language class known for '{language_name = }'")


def get_sign_language(language_name: Union[str, Enum]) -> SignLanguage:
    """
    Retrieves a SignLanguage object based on the provided language name.

    Args:
        language_name (str): The name of the language.

    Returns:
        SignLanguage: An instance of SignLanguage class corresponding to the provided language name.

    Raises:
        ValueError: If no SignLanguage class is known for the provided language name.
    """

    code_to_class = {
        SignLanguages.PAKISTAN_SIGN_LANGUAGE.value: PakistanSignLanguage,
    }

    class_ = code_to_class.get(normalize_short_code(language_name), None)
    if class_:
        return class_()  # constructor called

    # Unknown
    raise ValueError(f"no sign language class known for '{language_name = }'")


__all__ = [
    "get_text_language",
    "get_sign_language",
]
