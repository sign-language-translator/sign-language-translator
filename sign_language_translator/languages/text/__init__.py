"""Module that contains Text Languages Processors as classes to clean up, tokenize and tag texts"""

from sign_language_translator.languages.text.text_language import TextLanguage
from sign_language_translator.languages.text.urdu import Urdu
from sign_language_translator.text.tagger import Tags

__all__ = [
    "TextLanguage",
    "Urdu",
    "Tags",
]
