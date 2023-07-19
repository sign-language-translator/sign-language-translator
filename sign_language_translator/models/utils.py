"""utility functions for models"""

from os.path import join

from sign_language_translator.config.enums import ModelCodes, normalize_short_code
from sign_language_translator.config.settings import Settings
from sign_language_translator.models import ConcatenativeSynthesis, NgramLanguageModel
from sign_language_translator.utils import download_resource


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

    model_code = normalize_short_code(model_code)
    if model_code == ModelCodes.CONCATENATIVE_SYNTHESIS.value:
        if text_language and sign_language and video_feature_model:
            # TODO: validate arg types
            return ConcatenativeSynthesis(
                text_language=text_language,
                sign_language=sign_language,
                sign_features=video_feature_model,
            )
        raise ValueError(
            "Inappropriate argument value for text_language, sign_language or video_feature_model"
        )
    elif model_code in ModelCodes.ALL_NGRAM_LANGUAGE_MODELS.value:
        download_resource("models/" + model_code, progress_bar=True, leave=False)
        return NgramLanguageModel.load(
            join(Settings.RESOURCES_ROOT_DIRECTORY, "models", model_code)
        )
    elif model_code.lower() in ["stt", "sign-to-text", "sign_to_text"]:
        pass

    return None


__all__ = [
    "get_model",
]
