"""
This module contains the `MediaPipeLandmarksModel` class, which is a deep learning-based
video embedding model utilizing the MediaPipe framework for extracting pose and hand
landmarks from video frames.

Classes:
    MediaPipeLandmarksModel: A video embedding model that utilizes MediaPipe for pose and hand landmark extraction.

Example:

.. code-block:: python

    from sign_language_translator.models import MediaPipeLandmarksModel
    from sign_language_translator.vision.utils import iter_frames_with_opencv

    mediapipe_model = MediaPipeLandmarksModel(number_of_persons=1)

    frame_sequence = iter_frames_with_opencv("video.mp4")
    embedding = mediapipe_model.embed(frame_sequence, landmark_type="world")
    print(embedding.shape)
"""

from os.path import join
from typing import Dict, Iterable, List, Optional, Union

try:
    import mediapipe
except ImportError:
    mediapipe = None

import numpy as np
import torch
from numpy.typing import NDArray

from sign_language_translator.config.assets import Assets
from sign_language_translator.models.video_embedding.video_embedding_model import (
    VideoEmbeddingModel,
)
from sign_language_translator.utils import ProgressStatusCallback


class MediaPipeLandmarksModel(VideoEmbeddingModel):
    """
    A video embedding model using MediaPipe to extract pose and hand landmarks from video frames.

    Args:
        pose_model_name (str): The name of the pose estimation model.
        hand_model_name (str): The name of the hand estimation model.
        number_of_persons (int): The maximum number of persons to detect in each frame.

    Attributes:
        n_persons (int): The maximum number of persons to detect in each frame.

    Methods:
        embed: Embeds a sequence of frames using pose and hand landmarks.
    """

    def __init__(
        self,
        pose_model_name="pose_landmarker_heavy.task",
        hand_model_name="hand_landmarker.task",
        number_of_persons: int = 1,
    ) -> None:
        if mediapipe is None:
            raise ImportError(
                "The 'mediapipe' package is required to use the 'MediaPipeLandmarksModel'. "
                "Install it using `pip install sign-language-translator[mediapipe]`. "
                "(also make sure if your python version is compatible with mediapipe)."
            )

        self._pose_class = mediapipe.tasks.vision.PoseLandmarker
        self._hand_class = mediapipe.tasks.vision.HandLandmarker

        path = self.__download_and_get_model_path(f"models/mediapipe/{pose_model_name}")
        self._pose_options = mediapipe.tasks.vision.PoseLandmarkerOptions(
            base_options=mediapipe.tasks.BaseOptions(model_asset_path=path),
            running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
            num_poses=number_of_persons,
        )

        path = self.__download_and_get_model_path(f"models/mediapipe/{hand_model_name}")
        self._hand_options = mediapipe.tasks.vision.HandLandmarkerOptions(
            base_options=mediapipe.tasks.BaseOptions(model_asset_path=path),
            running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
            num_hands=number_of_persons * 2,
        )

        self.n_persons = number_of_persons

    def embed(
        self,
        frame_sequence: Iterable[Union[torch.Tensor, NDArray[np.uint8]]],
        landmark_type: str = "world" or "image" or "all",
        progress_callback: Optional[ProgressStatusCallback] = None,
        total_frames: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Embed a sequence of frames (video) into a sequence of pose & hand landmarks.

        Args:
            frame_sequence (Iterable[torch.Tensor | NDArray[np.uint8]]): A sequence of video frames as 3D arrays (W, H, c).
            landmark_type (str): The type of landmarks to include in the embedding ("world", "image", "all").

        Returns:
            torch.Tensor: A tensor containing the frame embeddings.
        """

        if mediapipe is None:
            raise ImportError(
                "The 'mediapipe' package is required to use the 'MediaPipeLandmarksModel'. "
                "Install it using `pip install sign-language-translator[mediapipe]`."
            )

        if landmark_type not in ("world", "image", "all"):
            raise ValueError(
                "landmark_type not supported, use 'world', 'image' or 'all'."
            )

        # TODO: Pose only or hands only
        if hasattr(frame_sequence, "__len__"):
            total_frames = len(frame_sequence)  # type: ignore

        embeddings = []

        # TODO: create here or in __init__ ??
        with self._pose_class.create_from_options(
            self._pose_options
        ) as pose_landmarker, self._hand_class.create_from_options(
            self._hand_options
        ) as hand_landmarker:
            for i, frame in enumerate(frame_sequence):
                # convert frame to mediapipe image
                mp_image = mediapipe.Image(
                    image_format=mediapipe.ImageFormat.SRGB,
                    data=np.array(frame),
                )

                # infer through models
                pose_result = pose_landmarker.detect_for_video(mp_image, i)
                hand_result = hand_landmarker.detect_for_video(mp_image, i)

                # create & append the frame embedding
                poses = self._extract_from_pose_results(pose_result)
                hands = self._extract_from_hand_results(hand_result)
                persons = self._arange_pose_and_hands(poses, hands)
                frame_embedding = self._create_frame_embedding(persons, landmark_type)

                embeddings.append(frame_embedding)
                if progress_callback and total_frames:
                    progress_callback(
                        {"file": f"{i / total_frames:.1%}" if total_frames else "?%"}
                    )

        return torch.Tensor(embeddings)

    def _flatten_landmarks(self, landmarks) -> List[float]:
        return [
            value
            for lm in landmarks
            for value in [lm.x, lm.y, lm.z, lm.visibility, lm.presence]
        ]

    def _extract_from_pose_results(self, pose_result) -> Dict[str, List[List[float]]]:
        poses = {"image": [], "world": []}

        for pose_image, pose_world in zip(
            pose_result.pose_landmarks, pose_result.pose_world_landmarks
        ):
            poses["image"].append(self._flatten_landmarks(pose_image))
            poses["world"].append(self._flatten_landmarks(pose_world))

        return poses

    def _extract_from_hand_results(self, hand_result) -> Dict[str, List[List[float]]]:
        hands = {
            "Left_image": [],
            "Left_world": [],
            "Right_image": [],
            "Right_world": [],
        }
        for hnd, image, world in zip(
            hand_result.handedness,
            hand_result.hand_landmarks,
            hand_result.hand_world_landmarks,
        ):
            # flatten & separate
            hands[hnd[0].display_name + "_image"].append(self._flatten_landmarks(image))
            hands[hnd[0].display_name + "_world"].append(self._flatten_landmarks(world))

        return hands

    def _arange_pose_and_hands(
        self,
        poses: Dict[str, List[List[float]]],
        hands: Dict[str, List[List[float]]],
    ) -> Dict[str, List[List[float]]]:
        # TODO: Match left & right hands to poses
        # by using minimum distance between hand image centers
        # np.linalg.norm(pose[left_hand_ids].mean(axis=...), hands.mean(axis=...).T).argmin(axis=...)
        default_hand = [0.0] * 5 * 21
        default_pose = [0.0] * 5 * 33

        for k in poses.keys():
            poses[k] += [default_pose] * (self.n_persons - len(poses[k]))

        for k in hands.keys():
            hands[k] += [default_hand] * (self.n_persons - len(hands[k]))

        return {
            key: [
                poses[key][p] + hands["Left_" + key][p] + hands["Right_" + key][p]
                for p in range(self.n_persons)
            ]  # TODO: order of persons should be the same across frames
            for key in ["image", "world"]
        }

    def _create_frame_embedding(
        self, persons: Dict[str, List[List[float]]], landmark_type: str
    ) -> List[float]:
        embedding = []
        # flatten & concat
        if landmark_type in ("world", "all"):
            embedding.extend([value for person in persons["world"] for value in person])
        if landmark_type in ("image", "all"):
            embedding.extend([value for person in persons["image"] for value in person])

        return embedding

    def __download_and_get_model_path(self, model_local_path: str):
        Assets.download(
            model_local_path,
            progress_bar=True,
            leave=False,
            chunk_size=1048576,
        )
        return join(Assets.ROOT_DIR, model_local_path)
