"""utility functions for models"""

from sign_language_translator.config.enums import ModelCodes
from sign_language_translator.models.text_to_sign.concatenative_synthesis import (
    ConcatenativeSynthesis,
)
from sign_language_translator.config.enums import normalize_short_code


def get_model(
    model_code: str,
    sign_language: str | None = None,
    text_language: str | None = None,
    video_feature_model: str | None = None,
):
    """
    Get the model based on the provided model code and optional parameters.

    Args:
        model_code (str): The code representing the desired model.
        sign_language (str, optional): The sign language used in the model. Defaults to None.
        text_language (str, optional): The text language used in the model. Defaults to None.
        video_feature_model (str, optional): The video feature model to be used for translation. Defaults to None.

    Returns:
        Any: The instantiated sign language model if successful, or None if no model found.

    Raises:
        ValueError: If inappropriate argument values are provided for text_language, sign_language, or video_feature_model.
    """

    if normalize_short_code(model_code) == ModelCodes.CONCATENATIVE_SYNTHESIS.value:
        if text_language and sign_language and video_feature_model:
            # TODO: validate arg types
            return ConcatenativeSynthesis(
                text_language=text_language,
                sign_language=sign_language,
                sign_features=video_feature_model,
            )
        else:
            raise ValueError(
                "Inappropriate argument value for text_language, sign_language or video_feature_model"
            )
    elif model_code.lower() in ["stt", "sign-to-text", "sign_to_text"]:
        pass

    return None


__all__ = [
    "get_model",
]
