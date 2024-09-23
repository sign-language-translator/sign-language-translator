"""Module that contains Text Language Processors as classes to clean up, tokenize and tag texts of various languages."""

from sign_language_translator.languages.text.english import English
from sign_language_translator.languages.text.hindi import Hindi
from sign_language_translator.languages.text.text_language import TextLanguage
from sign_language_translator.languages.text.urdu import Urdu
from sign_language_translator.text.tagger import Tags

__all__ = [
    "TextLanguage",
    "Urdu",
    "English",
    "Hindi",
    "Tags",
]
