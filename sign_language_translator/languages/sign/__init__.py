"""Module that contains Sign Languages as classes with rules 
to translate text tokens into sign language videos
"""

from sign_language_translator.languages.sign.mapping_rules import (
    CharacterByCharacterMappingRule,
    DirectMappingRule,
    LambdaMappingRule,
    MappingRule,
)
from sign_language_translator.languages.sign.pakistan_sign_language import (
    PakistanSignLanguage,
)
from sign_language_translator.languages.sign.sign_language import SignLanguage

__all__ = [
    "SignLanguage",
    "PakistanSignLanguage",
    "MappingRule",
    "LambdaMappingRule",
    "CharacterByCharacterMappingRule",
    "DirectMappingRule",
]
