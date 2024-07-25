"""This module provides access to configuration components for the Sign Language Translator package.

Contents:
- Assets: Class to manage the assets of the Sign Language Translator package.
- enums: Defines enumerations of short codes used in the Sign Language Translator package.
- settings: Contains constants and configurations for the Sign Language Translator package.
- utils: Contains helper functions for configuration of the package.
"""

from sign_language_translator.config import enums, utils
from sign_language_translator.config.assets import Assets
from sign_language_translator.config.colors import Colors
from sign_language_translator.config.settings import Settings

__all__ = [
    # classes
    "Assets",
    "Colors",
    "Settings",
    # modules
    "enums",
    "utils",
]
