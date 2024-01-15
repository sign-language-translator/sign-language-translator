"""Centralized String Constants for Sign Language Translator Package

This module defines enumerations for various string constants used throughout the package.

Enumerations:
    Countries (Enum): Enumerates supported countries with their short codes.
    Organizations (Enum): Enumerates supported organizations with their short codes.
    SignCollections (Enum): Enumerates sign collections with their names.
    TextLanguages (Enum): Enumerates supported text languages with their short codes.
    SignLanguages (Enum): Enumerates supported sign languages with their names.
    SignFormats (Enum): Enumerates supported video feature models.
    ModelCodes (Enum): Enumerates model codes for different models.

Functions:
    normalize_short_code (function): Normalizes a provided short code to a recognized standard form.
"""

from __future__ import annotations

from enum import Enum

from sign_language_translator.utils import (
    PrintableEnumMeta,
    search_in_values_to_retrieve_key,
)

__all__ = [
    "Countries",
    "Organizations",
    "SignCollections",
    "TextLanguages",
    "SignLanguages",
    "SignFormats",
    "ModelCodes",
    "normalize_short_code",
]


class Countries(Enum, metaclass=PrintableEnumMeta):
    """
    Enumeration of countries with their corresponding short codes.

    Attributes:
        - PAKISTAN (str): Short code for Pakistan.
        ...
    """

    PAKISTAN = "pk"
    # INDIA = "in"
    # USA = "us"


class Organizations(Enum, metaclass=PrintableEnumMeta):
    """
    Enumeration of organizations with their corresponding short codes.

    Attributes:
        - HFAD (str): Short code for HFAD (e.g., an organization for deaf in Pakistan).
        ...
    """

    HFAD = "hfad"
    # NISE = "nise"
    # FESF = "fesf"


class SignCollections(Enum, metaclass=PrintableEnumMeta):
    """
    Enumeration of sign collections with their corresponding short codes.

    Attributes:
        - PK_HFAD_1 (str): Short code for the first sign collection in Pakistan made in HFAD.
        - WORDLESS (str): Short code for a sign collection with videos containing no signs.
        ...
    """

    PK_HFAD_1 = f"{Countries.PAKISTAN}-{Organizations.HFAD}-1"
    WORDLESS = "wordless"


class TextLanguages(Enum, metaclass=PrintableEnumMeta):
    """
    Enumeration of text languages with their corresponding short codes.

    Attributes:
        - URDU (str): Short code for the Urdu language.
        - HINDI (str): Short code for the Hindi language.
        ...
    """

    URDU = "ur"
    # ENGLISH = "en"
    HINDI = "hi"


class SignLanguages(Enum, metaclass=PrintableEnumMeta):
    """
    Enumeration of sign languages with their corresponding short codes.

    Attributes:
        - PAKISTAN_SIGN_LANGUAGE (str): Short code for the Pakistan Sign Language.
        ...
    """

    PAKISTAN_SIGN_LANGUAGE = "pakistan-sign-language"


class SignFormats(Enum, metaclass=PrintableEnumMeta):
    """
    Enumeration of available sign formats with their corresponding short codes.
    For example, sign language can be a sequence of frames (video) or a sequence of pose vectors (landmarks) etc.

    Attributes:
        - VIDEO (str): Short code for raw video.
        - MEDIAPIPE_LANDMARKS (str): Short code for MediaPipe Pose & Hand landmarks.
        ...
    """

    # raw video
    VIDEO = "video"
    """sequence of numpy frames (num_frames, height, width, num_channels)"""

    # Body Landmarks
    MEDIAPIPE_LANDMARKS = "mediapipe-landmarks"
    """3d world coordinates (x,y,z,[visibility, presence]) and/or 2d image coordinates (x,y,depth,[visibility, presence])"""

    # Body Mesh Grid

    # Image Segmentation

    # Motion Vectors


class ModelCodes(Enum, metaclass=PrintableEnumMeta):
    """
    Enumeration of model codes with their corresponding short codes.

    Attributes:
        - CONCATENATIVE_SYNTHESIS (str): Short code for the our rule-based text-to-sign translation model.

        - NGRAM_LM_UNIGRAM_NAMES (str): Short code for ngram model trained with window size 1 on en/ur person names data.
        - NGRAM_LM_BIGRAM_NAMES (str): Short code for ngram model trained with window size 2 on en/ur person names data.
        - NGRAM_LM_TRIGRAM_NAMES (str): Short code for ngram model trained with window size 3 on en/ur person names data.
        - MIXER_LM_NGRAM_URDU (str): Short code for a mix of ngram models trained on urdu words of window size 1 to 6.
        - TRANSFORMER_LM_UR_SUPPORTED (str): Short code for a transformer-based language model trained on ur supported tokens.

        - MEDIAPIPE_POSE_V2_HAND_V1 (str): Short code for a video embedding model which uses MediaPipe pose_landmarker_heavy & hand_landmarker.
        - MEDIAPIPE_POSE_V1_HAND_V1 (str):
        - MEDIAPIPE_POSE_V0_HAND_V1 (str):
        ...
    """

    # text-to-sign translation
    CONCATENATIVE_SYNTHESIS = "concatenative-synthesis"
    """Short code for the core rule-based text to sign translation model that joins video clips for each word in a sentence."""
    # LANDMARK_GAN = "landmark-gan"

    # sign-to-text translation
    # GESTURE = "gesture"

    # language-models
    NGRAM_LM_UNIGRAM_NAMES = "names-stat-lm-w1.json"
    NGRAM_LM_BIGRAM_NAMES = "names-stat-lm-w2.json"
    NGRAM_LM_TRIGRAM_NAMES = "names-stat-lm-w3.json"
    MIXER_LM_NGRAM_URDU = "ur-supported-token-unambiguous-mixed-ngram-w1-w6-lm.pkl"
    TRANSFORMER_LM_UR_SUPPORTED = "tlm_14.0M.pt"

    # video-embedding-models
    MEDIAPIPE_POSE_V2_HAND_V1 = "mediapipe-pose-2-hand-1"
    MEDIAPIPE_POSE_V1_HAND_V1 = "mediapipe-pose-1-hand-1"
    MEDIAPIPE_POSE_V0_HAND_V1 = "mediapipe-pose-0-hand-1"

    # text-embedding-models

class ModelCodeGroups(Enum, metaclass=PrintableEnumMeta):
    """
    Enumeration class for grouping supported model codes, making it easier to filter various models.

    Attributes:
        ALL_LANGUAGE_MODELS (set): Set of model codes for all language models.
        ALL_VIDEO_FEATURE_MODELS (set): Set of model codes for all video embedding models.

        ALL_NGRAM_LANGUAGE_MODELS (set): Set of model codes for all n-gram language models.
        ALL_TRANSFORMER_LANGUAGE_MODELS (set): Set of model codes for all transformer-based language models.
        ALL_MIXER_LANGUAGE_MODELS (set): Set of model codes for all mixer-based language models.

        ALL_MEDIAPIPE_EMBEDDING_MODELS (set): Set of model codes for all MediaPipe-based video embedding models.
        ALL_VIDEO_EMBEDDING_MODELS (set): Set of model codes for all video embedding models.
    """

    # language models
    ALL_NGRAM_LANGUAGE_MODELS = {
        ModelCodes.NGRAM_LM_UNIGRAM_NAMES.value,
        ModelCodes.NGRAM_LM_BIGRAM_NAMES.value,
        ModelCodes.NGRAM_LM_TRIGRAM_NAMES.value,
    }
    ALL_TRANSFORMER_LANGUAGE_MODELS = {
        ModelCodes.TRANSFORMER_LM_UR_SUPPORTED.value,
    }
    ALL_MIXER_LANGUAGE_MODELS = {
        ModelCodes.MIXER_LM_NGRAM_URDU.value,
    }
    ALL_LANGUAGE_MODELS = (
        ALL_NGRAM_LANGUAGE_MODELS
        | ALL_TRANSFORMER_LANGUAGE_MODELS  # type: ignore
        | ALL_MIXER_LANGUAGE_MODELS
    )

    # video embedding models
    ALL_MEDIAPIPE_EMBEDDING_MODELS = {
        ModelCodes.MEDIAPIPE_POSE_V2_HAND_V1.value,
        ModelCodes.MEDIAPIPE_POSE_V1_HAND_V1.value,
        ModelCodes.MEDIAPIPE_POSE_V0_HAND_V1.value,
    }
    ALL_VIDEO_EMBEDDING_MODELS = ALL_MEDIAPIPE_EMBEDDING_MODELS

# TODO: move the mapping list outside of the function. maybe convert it to a class variable.
def normalize_short_code(short_code: str | Enum) -> str:
    """
    Normalize the provided short code to a standard form that is recognized package wide.

    Args:
        short_code (str): The short code to be normalized.

    Returns:
        str: The normalized short code.
    """

    if isinstance(short_code, Enum):
        short_code = str(short_code.value)

    normalized_to_codes = {
        ModelCodes.CONCATENATIVE_SYNTHESIS.value: {
            "rule-based",
            "concatenative",
            "concatenativesynthesis",
            "concatenative-synthesis",
            "concatenative_synthesis",
            "concat-synth",
        },
        TextLanguages.URDU.value: {
            "urdu",
            "urd",
            "ur",
        },
        TextLanguages.HINDI.value: {
            "hindi",
            "hin",
            "hi",
        },
        SignLanguages.PAKISTAN_SIGN_LANGUAGE.value: {
            "psl",
            "pk-sl",
            "pakistan-sign-language",
            "pakistan-sl",
            "pakistansignlanguage",
            "pakistan_sign_language",
        },
        SignFormats.VIDEO.value: {
            "vid",
            "videos",
            "vids",
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
            "urdu-supported-ngram",
            "ur-mixed-ngram",
            "ur-supported-ngram",
        },
        ModelCodes.TRANSFORMER_LM_UR_SUPPORTED.value: {
            "ur-supported-gpt",
            "urdu-supported-gpt",
        },
        ModelCodes.MEDIAPIPE_POSE_V2_HAND_V1.value: {
            "mediapipe",
        },
    }
    normalized_to_codes = {
        k: v.union(
            {
                k,
                k.replace("-", "_"),
                k.replace("_", "-"),
                *map(lambda x: x.replace("-", "_"), v),
            }
        )
        for k, v in normalized_to_codes.items()
    }

    normalized = search_in_values_to_retrieve_key(short_code, normalized_to_codes)

    return normalized or short_code
