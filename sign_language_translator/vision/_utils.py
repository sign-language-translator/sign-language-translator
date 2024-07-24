from enum import Enum
from typing import Type, Union

from sign_language_translator.config.enums import SignFormats, normalize_short_code
from sign_language_translator.vision.sign.sign import Sign

__all__ = [
    "get_sign_wrapper_class",
]


def get_sign_wrapper_class(sign_code: Union[str, Enum]) -> Type[Sign]:
    """
    Retrieves a Sign wrapping class based on the provided string code.

    Args:
        sign_code (str): The name of the Sign wrapper class. e.g. "video", "landmarks" etc.

    Returns:
        Type[Sign]: An instance of the Sign wrapper class corresponding to the provided code.

    Raises:
        ValueError: If no Sign wrapper class is known for the provided code.
    """

    from sign_language_translator.vision.landmarks.landmarks import Landmarks
    from sign_language_translator.vision.video.video import Video

    code_to_class = {
        SignFormats.LANDMARKS.value: Landmarks,
        SignFormats.VIDEO.value: Video,
    }

    class_ = code_to_class.get(normalize_short_code(sign_code), None)
    if class_ is not None:
        return class_

    # Unknown
    raise ValueError(f"no sign wrapper class known for '{sign_code = }'")
