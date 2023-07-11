"""utility functions for models"""

from sign_language_translator.models.text_to_sign.concatenative_synthesis import (
    ConcatenativeSynthesis,
)


def get_model(
    model_code: str,
    sign_language: str | None = None,
    text_language: str | None = None,
    video_feature_model: str | None = None,
):
    if model_code.lower() in {
        "rule-based",
        "concatenative",
        "concatenativesynthesis",
        "concatenative-synthesis",
        "concatenative_synthesis",
    }:
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
