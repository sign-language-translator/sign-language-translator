"""
sign_language_translator.models.language_models
==============================================

This module contains various language models used in the sign language translator system.

Language Models:
----------------
- NgramLanguageModel: A language model based on n-grams.
- MixerLM: A language model that combines multiple language models using mixing weights or random selection.
- BeamSampling: A class that performs beam search during text generation.
- LanguageModel: An abstract base class for all language models in this package.
- TransformerLanguageModel: A transformer-based language model.

Usage:
------
To use the language models, import them directly from this module. For example:

.. code-block:: python

    from sign_language_translator.models.language_models import NgramLanguageModel, TransformerLanguageModel

    # Create and use the NgramLanguageModel
    model = NgramLanguageModel(window_size=2, unknown_token="")
    model.fit(["hello", "world"])
    text = "yel"
    next_word, probability = model.next(text)

    # Create and use the TransformerLanguageModel
    transformer_lm = TransformerLanguageModel.load("tlm.pt")
    tokens = ["how", "are", "you"]
    next_word, probability = transformer_lm.next(tokens)
"""

from sign_language_translator.models.language_models import transformer_language_model
from sign_language_translator.models.language_models.abstract_language_model import (
    LanguageModel,
)
from sign_language_translator.models.language_models.beam_sampling import BeamSampling
from sign_language_translator.models.language_models.mixer import MixerLM
from sign_language_translator.models.language_models.ngram_language_model import (
    NgramLanguageModel,
)
from sign_language_translator.models.language_models.transformer_language_model.model import (
    TransformerLanguageModel,
)

__all__ = [
    "NgramLanguageModel",
    "MixerLM",
    "BeamSampling",
    "LanguageModel",
    "TransformerLanguageModel",
    "transformer_language_model",
]
