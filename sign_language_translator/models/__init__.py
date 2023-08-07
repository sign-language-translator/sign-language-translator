"""
sign_language_translator.models
===============================

This module contains the various models in the sign language translator system
and their associated components.

Language Models:
----------------
- NgramLanguageModel: A language model based on n-grams.
- TransformerLanguageModel: A transformer-based language model.
- MixerLM: A language model that combines multiple language models using mixing weights.
- LanguageModel: An abstract base class for all language models in this package.
- BeamSampling: A utility class that performs beam search during text generation.

Text to Sign Translation:
-------------------------
- TextToSignModel: A model that translates text into sign language gestures.
- ConcatenativeSynthesis: A rule-based model for synthesizing sign language gestures from text.

Utilities:
----------
- get_model: A utility function to get any model by string name.
- utils: Miscellaneous utility functions for the sign language translator system.
"""

from sign_language_translator.models import (
    language_models,
    sign_to_text,
    text_to_sign,
    utils,
)
from sign_language_translator.models._utils import get_model
from sign_language_translator.models.language_models import (
    BeamSampling,
    LanguageModel,
    MixerLM,
    NgramLanguageModel,
    TransformerLanguageModel,
)
from sign_language_translator.models.text_to_sign import ConcatenativeSynthesis
from sign_language_translator.models.text_to_sign.t2s_model import TextToSignModel

__all__ = [
    "get_model",
    "language_models",
    "sign_to_text",
    "text_to_sign",
    "utils",
    "ConcatenativeSynthesis",
    "NgramLanguageModel",
    "TransformerLanguageModel",
    "BeamSampling",
    "MixerLM",
    "LanguageModel",
    "TextToSignModel",
]
