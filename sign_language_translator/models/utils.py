"""utility functions for models"""

from sign_language_translator.models.text_to_sign.concatenative_synthesis import (
    ConcatenativeSynthesis,
)


def get_model(
    task: str,
    sign_language: str,
    text_language: str,
    video_feature_model: str,
    approach: str | None = None,
    model_code: str | None = None,
):
    if task.lower() in ["tts", "text-to-sign", "text_to_sign"]:
        if approach and approach.lower() in {"rule-based", "concatenative"}:
            return ConcatenativeSynthesis(
                text_language=text_language,
                sign_language=sign_language,
                sign_features=video_feature_model,
            )
    elif task in ["stt", "sign-to-text", "sign_to_text"]:
        if model_code in {}:
            pass
    else:
        return None


__all__ = [
    "get_model",
]
