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
from typing import Union

from sign_language_translator.utils.utils import (
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
    "SignEmbeddingModels",
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
    """Hamza Foundation Academy for the Deaf (Lahore, Pakistan)"""
    # NISE = "nise"
    # """National Institute of Special Education (Islamabad, Pakistan)"""
    # FESF = "fesf"
    # """Family Educational Services Foundation (Karachi, Pakistan)"""


class SignCollections(Enum, metaclass=PrintableEnumMeta):
    """
    Enumeration of sign collections with their corresponding short codes.

    Attributes:
        - PK_HFAD_1 (str): Short code for the first sign dictionary from HamzaFoundationAcademyDeaf, Pakistan.
        - PK_HFAD_2 (str): Short code for the second sign collection from HamzaFoundationAcademyDeaf, Pakistan.
        ...
    """

    PK_HFAD_1 = f"{Countries.PAKISTAN}-{Organizations.HFAD}-1"
    """Short code for the first sign dictionary from HamzaFoundationAcademyDeaf, Pakistan. (788 videos)"""

    PK_HFAD_2 = f"{Countries.PAKISTAN}-{Organizations.HFAD}-2"
    """Short code for the second sign dictionary from HamzaFoundationAcademyDeaf, Pakistan."""

    # PK_FESF_1 = f"{Countries.PAKISTAN}-{Organizations.FESF}-1"
    # PK_NISE_1 = f"{Countries.PAKISTAN}-{Organizations.NISE}-1"


class TextLanguages(Enum, metaclass=PrintableEnumMeta):
    """
    Enumeration of text languages with their corresponding short codes.

    Attributes:
        - URDU (str): Short code for the Urdu language.
        - HINDI (str): Short code for the Hindi language.
        - ENGLISH (str): Short code for the English language.
    """

    URDU = "ur"
    ENGLISH = "en"
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
        - LANDMARKS (str): Short code for body landmarks.
        ...
    """

    # raw video
    VIDEO = "video"
    """Sequence of RGB frames (num_frames, height, width, num_channels)"""

    # Body Landmarks
    LANDMARKS = "landmarks"
    """Sequence of points on the body (n_frames, n_points, n_coordinates)"""

    # MEDIAPIPE_LANDMARKS = "mediapipe-landmarks"
    # """3d world coordinates (x,y,z,[visibility, presence]) and/or 2d image coordinates (x,y,depth,[visibility, presence])"""

    # Body Mesh Grid

    # Image Segmentation

    # Motion Vectors


class ModelCodes(Enum, metaclass=PrintableEnumMeta):
    """
    Enumeration of model codes with their corresponding short codes.

    Attributes:
        - CONCATENATIVE_SYNTHESIS (str): Short code for the rule-based text-to-sign translation model.

        - NGRAM_LM_UNIGRAM_NAMES (str): Short code for ngram model trained with window size 1 on en/ur person names data.
        - NGRAM_LM_BIGRAM_NAMES (str): Short code for ngram model trained with window size 2 on en/ur person names data.
        - NGRAM_LM_TRIGRAM_NAMES (str): Short code for ngram model trained with window size 3 on en/ur person names data.
        - MIXER_LM_NGRAM_URDU (str): Short code for a mix of ngram models trained on urdu words of window size 1 to 6.
        - TRANSFORMER_LM_UR_SUPPORTED (str): Short code for a transformer-based language model trained on ur supported tokens.

        - MEDIAPIPE_POSE_V2_HAND_V1 (str): Short code for a video embedding model which uses MediaPipe pose_landmarker_heavy & hand_landmarker.
        - MEDIAPIPE_POSE_V1_HAND_V1 (str): Short code for a video embedding model which uses MediaPipe pose_landmarker_full & hand_landmarker.
        - MEDIAPIPE_POSE_V0_HAND_V1 (str): Short code for a video embedding model which uses MediaPipe pose_landmarker_lite & hand_landmarker.

        - LOOKUP_UR_FASTTEXT_CC (str): Short code for a text embedding model which uses a lookup table to embed Urdu tokens using fastText embeddings trained on Common Crawl.
    """

    # text-to-sign translation
    CONCATENATIVE_SYNTHESIS = "concatenative-synthesis"
    """The core rule-based text to sign translation model that joins sign clips for each word in a text sentence."""
    # LANDMARK_GAN = "landmark-gan"

    # sign-to-text translation
    # GESTURE = "gesture"

    # language-models
    NGRAM_LM_UNIGRAM_NAMES = "names-stat-lm-w1.json"
    NGRAM_LM_BIGRAM_NAMES = "names-stat-lm-w2.json"
    NGRAM_LM_TRIGRAM_NAMES = "names-stat-lm-w3.json"
    MIXER_LM_NGRAM_URDU = "ur-supported-token-unambiguous-mixed-ngram-w1-w6-lm.pkl"
    """Simple hash table based n-gram language model with context size of 1-6 that generates unambiguous Urdu tokens."""
    TRANSFORMER_LM_UR_SUPPORTED = "tlm_14.0M.pt"

    # video-embedding-models
    MEDIAPIPE_POSE_V2_HAND_V1 = "mediapipe-pose-2-hand-1"
    """Short code for the video embedding model which uses MediaPipe pose_landmarker_heavy & hand_landmarker to generate (33 pose + 2 * 21 hand) world & 75 image landmarks (x, y, z, visibility, presence) for each frame of the video."""
    MEDIAPIPE_POSE_V1_HAND_V1 = "mediapipe-pose-1-hand-1"
    """Short code for the video embedding model which uses MediaPipe pose_landmarker_full  & hand_landmarker to generate (33 pose + 2 * 21 hand) world & 75 image landmarks (x, y, z, visibility, presence) for each frame of the video."""
    MEDIAPIPE_POSE_V0_HAND_V1 = "mediapipe-pose-0-hand-1"
    """Short code for the video embedding model which uses MediaPipe pose_landmarker_lite  & hand_landmarker to generate (33 pose + 2 * 21 hand) world & 75 image landmarks (x, y, z, visibility, presence) for each frame of the video."""

    # text-embedding-models
    LOOKUP_UR_FASTTEXT_CC = "lookup-ur-fasttext-cc.pt"
    """Short code for the text embedding model which uses a lookup table to embed Urdu tokens using fastText embeddings trained on Common Crawl."""


class SignEmbeddingModels(Enum, metaclass=PrintableEnumMeta):
    """The Names of video embedding models that *have been* used to embed sign language videos in the available datasets.

    Attributes:
        MEDIAPIPE_WORLD (str): Short code for the video embedding model which uses MediaPipe pose_landmarker_heavy & hand_landmarker to generate (33 pose + 2 * 21 hand) world landmarks (x, y, z, visibility, presence) for each frame of the video. World landmarks are 3D coordinates in meters with origin at the center of the hips for pose_landmarker and at the center of each hand for hand_landmarker model.
        MEDIAPIPE_IMAGE (str): Short code for the video embedding model which uses MediaPipe pose_landmarker_heavy & hand_landmarker to generate (33 pose + 2 * 21 hand) image landmarks (x, y, z, visibility, presence) for each frame of the video. Image landmarks are 2D coordinates as fraction of the frame width or height with origin at the top-left corner of the frame and z value is the depth from the camera.
    """

    MEDIAPIPE_WORLD = "mediapipe-world"
    MEDIAPIPE_IMAGE = "mediapipe-image"

    # TODO: rename `Models` in class name to avoid confusion with `ModelCodes`


class ModelCodeGroups(Enum, metaclass=PrintableEnumMeta):
    """
    Enumeration class for grouping supported model codes, making it easier to filter various models.

    Attributes:
        ALL_LANGUAGE_MODELS (set): Set of model codes for all language models.
        ALL_NGRAM_LANGUAGE_MODELS (set): Set of model codes for all n-gram language models.
        ALL_TRANSFORMER_LANGUAGE_MODELS (set): Set of model codes for all transformer-based language models.
        ALL_MIXER_LANGUAGE_MODELS (set): Set of model codes for all mixer-based language models.

        ALL_VIDEO_EMBEDDING_MODELS (set): Set of model codes for all video embedding models.
        ALL_MEDIAPIPE_EMBEDDING_MODELS (set): Set of model codes for all MediaPipe-based video embedding models.

        ALL_TEXT_EMBEDDING_MODELS (set): Set of model codes for all text embedding models.
        ALL_VECTOR_LOOKUP_MODELS (set): Set of model codes for all vector lookup-based text embedding models.
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

    # text embedding models
    ALL_VECTOR_LOOKUP_MODELS = {
        ModelCodes.LOOKUP_UR_FASTTEXT_CC.value,
    }
    ALL_TEXT_EMBEDDING_MODELS = ALL_VECTOR_LOOKUP_MODELS


# TODO: move the mapping list outside of the function. maybe convert it to a class variable.
def normalize_short_code(short_code: Union[str, Enum]) -> str:
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
        TextLanguages.ENGLISH.value: {
            "english",
            "eng",
            "en",
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
        SignFormats.LANDMARKS.value: {
            "landmark",
            "lmrks",
            "lmrk",
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
        SignEmbeddingModels.MEDIAPIPE_IMAGE.value: {
            "landmark-mediapipe-image",
            "landmarks-mediapipe-image",
            "mediapipe-image-landmarks",
            "mediapipe-pose-2-hand-1-image",
            "mediapipe-pose-2-hands-1-image",
            "landmarks-mediapipe-pose-2-hand-1-image",
            "landmark-mediapipe-pose-2-hand-1-image",
            "landmarks-mediapipe-pose-2-hands-1-image",
            "landmark-mediapipe-pose-2-hands-1-image",
        },
        SignEmbeddingModels.MEDIAPIPE_WORLD.value: {
            "landmark-mediapipe-world",
            "landmarks-mediapipe-world",
            "mediapipe-world-landmarks",
            "mediapipe-pose-2-hand-1-world",
            "mediapipe-pose-2-hands-1-world",
            "landmarks-mediapipe-pose-2-hand-1-world",
            "landmark-mediapipe-pose-2-hand-1-world",
            "landmarks-mediapipe-pose-2-hands-1-world",
            "landmark-mediapipe-pose-2-hands-1-world",
        },
        ModelCodes.LOOKUP_UR_FASTTEXT_CC.value: {
            "lookup-ur-ft-cc",
            "lookup-ur-fasttext-cc",
            "ur-lookup-fasttext-cc",
            "ur-lookup-ftcc",
            "ur-lookup-ft-cc",
            "urdu-lookup-fasttext-cc",
            "urdu-lookup-ftcc",
            "urdu-lookup-ft-cc",
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
