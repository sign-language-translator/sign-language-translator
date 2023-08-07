"""Centralized String Constants for Sign Language Translator Package

This module defines enumerations for various string constants used throughout the package.

Enumerations:
    Countries (Enum): Enumerates supported countries with their short codes.
    Organizations (Enum): Enumerates supported organizations with their short codes.
    SignCollections (Enum): Enumerates sign collections with their names.
    TextLanguages (Enum): Enumerates supported text languages with their short codes.
    SignLanguages (Enum): Enumerates supported sign languages with their names.
    VideoFeatures (Enum): Enumerates supported video feature models.
    ModelCodes (Enum): Enumerates model codes for different models.

Functions:
    normalize_short_code (function): Normalizes a provided short code to a recognized standard form.
"""

from __future__ import annotations

from enum import Enum

from sign_language_translator.utils import search_in_values_to_retrieve_key


class Countries(Enum):
    """
    Enumeration of countries with their corresponding short codes.

    Attributes:
        - PAKISTAN (str): Short code for Pakistan.
        ...
    """

    PAKISTAN = "pk"
    # USA = "usa"


class Organizations(Enum):
    """
    Enumeration of organizations with their corresponding short codes.

    Attributes:
        - HFAD (str): Short code for HFAD (e.g., an organization for deaf in Pakistan).
        ...
    """

    HFAD = "hfad"
    # NICE = "nice"
    # FESF = "fesf"


class SignCollections(Enum):
    """
    Enumeration of sign collections with their corresponding short codes.

    Attributes:
        - PK_HFAD_1 (str): Short code for the first sign collection in Pakistan made in HFAD.
        - WORDLESS (str): Short code for a sign collection with videos containing no signs.
        ...
    """

    PK_HFAD_1 = f"{Countries.PAKISTAN}-{Organizations.HFAD}-1"
    WORDLESS = "wordless"


class TextLanguages(Enum):
    """
    Enumeration of text languages with their corresponding short codes.

    Attributes:
        - URDU (str): Short code for the Urdu language.
        ...
    """

    URDU = "urdu"
    # ENGLISH = "english"


class SignLanguages(Enum):
    """
    Enumeration of sign languages with their corresponding short codes.

    Attributes:
        - PAKISTAN_SIGN_LANGUAGE (str): Short code for the Pakistan Sign Language.
        ...
    """

    PAKISTAN_SIGN_LANGUAGE = "pakistan-sign-language"


class VideoFeatures(Enum):
    """
    Enumeration of video feature models with their corresponding short codes.

    Attributes:
        - MEDIAPIPE_POSE_V2_HAND_V1_3D (str): Short code for MediaPipe Pose V2 Hand V1 model for 3D landmarks.
        ...
    """

    # Body Landmarks
    # 3d world coordinates (x,y,z,[visibility])
    MEDIAPIPE_POSE_V2_HAND_V1_3D = "mediapipe_pose_v2_hand_v1_3d"

    # Body Mesh Grid

    # Image Segmentation


class ModelCodes(Enum):
    """
    Enumeration of model codes with their corresponding short codes.

    Attributes:
        - CONCATENATIVE_SYNTHESIS (str): Short code for the our rule-based text-to-sign translation model.

        - NGRAM_LM_UNIGRAM_NAMES (str): Short code for ngram model trained with window size 1 on en/ur person names data.
        - NGRAM_LM_BIGRAM_NAMES (str): Short code for ngram model trained with window size 2 on en/ur person names data.
        - NGRAM_LM_TRIGRAM_NAMES (str): Short code for ngram model trained with window size 3 on en/ur person names data.
        ...
    """

    # text-to-sign translation
    CONCATENATIVE_SYNTHESIS = "concatenative-synthesis"
    """Short code for the core rule-based text to sign translation model that enables building synthetic training datasets."""

    # sign-to-text translation

    # language-models
    NGRAM_LM_UNIGRAM_NAMES = "names-stat-lm-w1.json"
    NGRAM_LM_BIGRAM_NAMES = "names-stat-lm-w2.json"
    NGRAM_LM_TRIGRAM_NAMES = "names-stat-lm-w3.json"
    ALL_NGRAM_LANGUAGE_MODELS = {
        NGRAM_LM_UNIGRAM_NAMES,
        NGRAM_LM_BIGRAM_NAMES,
        NGRAM_LM_TRIGRAM_NAMES,
    }

    MIXER_LM_NGRAM_URDU = "ur-supported-token-unambiguous-mixed-ngram-w1-w6-lm.pkl"
    ALL_MIXER_LANGUAGE_MODELS = {
        MIXER_LM_NGRAM_URDU,
    }

    TRANSFORMER_LM_UR_SUPPORTED = "tlm_14.0M.pt"
    ALL_TRANSFORMER_LANGUAGE_MODELS = {
        TRANSFORMER_LM_UR_SUPPORTED,
    }


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
        ModelCodes.NGRAM_LM_UNIGRAM_NAMES.value: {
            "unigram-names",
        },
        ModelCodes.NGRAM_LM_BIGRAM_NAMES.value: {
            "bigram-names",
        },
        ModelCodes.NGRAM_LM_TRIGRAM_NAMES.value: {
            "trigram-names",
        },
        ModelCodes.MIXER_LM_NGRAM_URDU.value: {
            "urdu-mixed-ngram",
        },
        ModelCodes.TRANSFORMER_LM_UR_SUPPORTED.value: {
            "ur-supported-gpt",
        },
    }
    normalized_to_codes = {k: v.union({k}) for k, v in normalized_to_codes.items()}

    normalized = search_in_values_to_retrieve_key(short_code, normalized_to_codes)
    if normalized:
        return normalized  # constructor called

    # Unknown
    raise ValueError(f"nothing identified by code: {short_code = }")


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
