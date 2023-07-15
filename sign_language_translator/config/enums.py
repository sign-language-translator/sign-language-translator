from __future__ import annotations

import enum
from sign_language_translator.utils import search_in_values_to_retrieve_key


class Countries(enum.Enum):
    PAKISTAN = "pk"
    # USA = "usa"


class Organizations(enum.Enum):
    HFAD = "hfad"
    # NICE = "nice"
    # FESF = "fesf"


class SignCollections(enum.Enum):
    PK_HFAD_1 = f"{Countries.PAKISTAN}-{Organizations.HFAD}-1"
    WORDLESS = "wordless"


class TextLanguages(enum.Enum):
    URDU = "urdu"
    # ENGLISH = "english"


class SignLanguages(enum.Enum):
    PAKISTAN_SIGN_LANGUAGE = "pakistan-sign-language"


class VideoFeatures(enum.Enum):
    # Body Landmarks
    # 3d world coordinates (x,y,z,[visibility])
    MEDIAPIPE_POSE_V2_HAND_V1_3D = "mediapipe_pose_v2_hand_v1_3d"

    # Body Mesh Grid

    # Image Segmentation


class ModelCodes(enum.Enum):
    # text-to-sign
    CONCATENATIVE_SYNTHESIS = "concatenative-synthesis"

    # language-models
    SIMPLE_LM = "simple-language-model"
    MIXER_LM = "mixer"
    TRANSFORMER_LM = "transformer-language-model"


def normalize_short_code(short_code: str) -> str:
    """
    Normalize the provided short code to a standard form that is recognized package wide.

    Args:
        short_code (str): The short code to be normalized.

    Returns:
        str: The normalized short code.

    Raises:
        ValueError: If the provided short code is unknown.
    """

    normalized_to_codes = {
        ModelCodes.CONCATENATIVE_SYNTHESIS.value: {
            "rule-based",
            "concatenative",
            "concatenativesynthesis",
            "concatenative-synthesis",
            "concatenative_synthesis",
        },
        TextLanguages.URDU.value: {
            "urdu",
            "ur",
        },
        SignLanguages.PAKISTAN_SIGN_LANGUAGE.value: {
            "psl",
            "pk-sl",
            "pakistan-sign-language",
            "pakistansignlanguage",
            "pakistan_sign_language",
        },
    }
    normalized = search_in_values_to_retrieve_key(short_code, normalized_to_codes)
    if normalized:
        return normalized  # constructor called

    # Unknown
    raise ValueError(f"nothing identified by code: '{short_code = }'")


__all__ = [
    "Countries",
    "Organizations",
    "SignCollections",
    "TextLanguages",
    "SignLanguages",
    "VideoFeatures",
    "ModelCodes",
    "normalize_short_code",
]
