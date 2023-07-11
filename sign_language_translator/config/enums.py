from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..languages.text.text_language import TextLanguage
    from ..languages.sign.sign_language import SignLanguage

import enum


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
    URDU = "Urdu"
    # ENGLISH = "English"


class SignLanguages(enum.Enum):
    PAKISTAN_SIGN_LANGUAGE = "PakistanSignLanguage"


class VideoFeatures(enum.Enum):
    # Body Landmarks
    # 3d world coordinates (x,y,z,[visibility])
    MEDIAPIPE_POSE_V2_HAND_V1_3D = "mediapipe_pose_v2_hand_v1_3d"

    # Body Mesh Grid

    # Image Segmentation


class ModelCodes(enum.Enum):
    CONCATENATIVE_SYNTHESIS = "concatenative-synthesis"


__all__ = [
    "Countries",
    "Organizations",
    "SignCollections",
    "TextLanguages",
    "SignLanguages",
    "VideoFeatures",
    "ModelCodes",
]
