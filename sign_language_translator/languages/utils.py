"""utility functions for language objects"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Set, Type

from sign_language_translator.languages.sign import PakistanSignLanguage
from sign_language_translator.languages.text import Urdu

if TYPE_CHECKING:
    from sign_language_translator.languages.sign import SignLanguage
    from sign_language_translator.languages.text import TextLanguage


def __search_in_values_to_retrieve_key(
    code_name: str, class_to_codes: Dict[Type, Set[str]]
):
    # verify there is no repetition/reuse in language codes
    all_codes = [code for codes in class_to_codes.values() for code in codes]
    assert len(all_codes) == len(set(all_codes)), "code reused for multiple keys"

    for class_, codes in class_to_codes.items():
        if code_name.lower() in codes:
            return class_

    return None


def get_text_language(language_name: str) -> TextLanguage:
    """
    Retrieves a TextLanguage object based on the provided language name.

    Args:
        language_name (str): The name of the language.

    Returns:
        TextLanguage: An instance of the TextLanguage class corresponding to the provided language name.

    Raises:
        ValueError: If no TextLanguage class is known for the provided language name.
    """

    language_codes = {
        Urdu: {"urdu", "ur"},
    }

    class_ = __search_in_values_to_retrieve_key(language_name, language_codes)
    if class_:
        return class_()  # constructor called

    # Unknown
    raise ValueError(f"no text language class known for '{language_name = }'")


def get_sign_language(language_name: str) -> SignLanguage:
    """
    Retrieves a SignLanguage object based on the provided language name.

    Args:
        language_name (str): The name of the language.

    Returns:
        SignLanguage: An instance of SignLanguage class corresponding to the provided language name.

    Raises:
        ValueError: If no SignLanguage class is known for the provided language name.
    """

    language_codes = {
        PakistanSignLanguage: {"pakistansignlanguage", "pakistan_sign_language", "psl"},
    }

    class_ = __search_in_values_to_retrieve_key(language_name, language_codes)
    if class_:
        return class_()  # constructor called

    # Unknown
    raise ValueError(f"no sign language class known for '{language_name = }'")


__all__ = [
    "get_text_language",
    "get_sign_language",
]
