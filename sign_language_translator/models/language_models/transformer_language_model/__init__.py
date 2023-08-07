"""
sign_language_translator.models.language_models.transformer_language_model
============================================================================

This module contains the implementation of the Transformer-based language model.

Components:
-----------
- TransformerLanguageModel: The main class representing the Transformer-based language model.
- layers: Custom implementation of neural network layers used in the Transformer model.
- train: Functions & Classes related to training the Transformer model including LM_Dataset and LM_Trainer.
"""

from sign_language_translator.models.language_models.transformer_language_model import (
    layers,
    train,
)
from sign_language_translator.models.language_models.transformer_language_model.model import (
    TransformerLanguageModel,
)

__all__ = [
    "TransformerLanguageModel",
    "layers",
    "train",
]
