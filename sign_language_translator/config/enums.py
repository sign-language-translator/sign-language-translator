from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..languages.text.text_language import TextLanguage
    from ..languages.sign.sign_language import SignLanguage


import enum
# from ..languages.vocab import RECORDING_LABELS
from ..languages.sign_collection import SignCollection


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

# SignCollection(
#     country=Countries.PAKISTAN.value,
#     organization=Organizations.HFAD.value,
#     collection_id="1",
#     labels=[],
#     # labels=RECORDING_LABELS[
#     #     f"{Countries.PAKISTAN.value}-{Organizations.HFAD.value}-1"
#     # ],
#     cameras=["front", "left", "top-left", "right", "top-right" "below"],
#     persons=["df1", "dm1"]
#     + [f"h{g}{i}" for g in ["f", "m"] for i in range(1, 5 + 1)],
#     # filenames = # correct camera
# )

class VideoFeatures(enum.Enum):
    # Body Landmarks
    MEDIAPIPE_POSE_V2_HAND_V1_3D = (
        "mediapipe_pose_v2_hand_v1_3d"  # 3d world coordinates (x,y,z,[visibility])
    )

    # Body Mesh Grid

    # Image Segmentation
