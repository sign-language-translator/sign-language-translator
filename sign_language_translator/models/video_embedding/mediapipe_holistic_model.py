"""
This module contains the `MediaPipeHolisticModel` class, a video embedding model that
uses the MediaPipe Holistic solution to extract pose and hand landmarks from video frames.

Holistic first tracks the body pose and then crops each hand around the pose's wrists,
so the left/right hand assignment always follows the body. It is the landmark setup used by
OpenHands (https://github.com/AI4Bharat/OpenHands) and many pose-based sign language
recognition models, so features extracted with it are directly comparable to theirs.

Classes:
    MediaPipeHolisticModel: A video embedding model that utilizes MediaPipe Holistic for pose and hand landmark extraction.

Example:

.. code-block:: python

    from sign_language_translator.models import MediaPipeHolisticModel
    from sign_language_translator.vision.utils import iter_frames_with_opencv

    holistic_model = MediaPipeHolisticModel(model_complexity=2)

    frame_sequence = iter_frames_with_opencv("video.mp4")
    embedding = holistic_model.embed(frame_sequence, landmark_type="image")
    print(embedding.shape)  # (n_frames, 75 * 5)
"""

from typing import Iterable, List, Optional, Union

try:
    import mediapipe
except ImportError:
    mediapipe = None

import numpy as np
import torch
from numpy.typing import NDArray

from sign_language_translator.models.video_embedding.video_embedding_model import (
    VideoEmbeddingModel,
)
from sign_language_translator.utils import ProgressStatusCallback


class MediaPipeHolisticModel(VideoEmbeddingModel):
    """
    A video embedding model using the MediaPipe Holistic solution
    (``mediapipe.solutions.holistic``) to extract pose and hand landmarks from video frames.

    The output layout is the same as ``MediaPipeLandmarksModel``: for every frame,
    33 pose landmarks followed by 21 left hand and 21 right hand landmarks,
    each as ``(x, y, z, visibility, presence)``. "Left" and "right" are the signer's own hands.
    Landmarks that are not detected in a frame (e.g. a hand out of view) are filled with zeros.
    Face mesh landmarks are not included.

    Args:
        model_complexity (int): Complexity of the pose landmark model: 0 (lite), 1 (full) or 2 (heavy). Defaults to 2.
            The model file for complexity 2 is downloaded by MediaPipe on first use.

    Attributes:
        model_complexity (int): Complexity of the pose landmark model.

    Methods:
        embed: Embeds a sequence of frames using pose and hand landmarks.
    """

    N_POSE_LANDMARKS = 33
    N_HAND_LANDMARKS = 21
    N_VALUES = 5  # x, y, z, visibility, presence

    def __init__(self, model_complexity: int = 2) -> None:
        if mediapipe is None:
            raise ImportError(
                "The 'mediapipe' package is required to use the 'MediaPipeHolisticModel'. "
                "Install it using `pip install sign-language-translator[mediapipe]`. "
                "(also make sure if your python version is compatible with mediapipe)."
            )
        if model_complexity not in (0, 1, 2):
            raise ValueError("model_complexity must be 0, 1 or 2.")

        self.model_complexity = model_complexity

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
            frame_sequence (Iterable[torch.Tensor | NDArray[np.uint8]]): A sequence of RGB video frames as 3D arrays (H, W, c).
            landmark_type (str): The type of landmarks to include in the embedding ("world", "image", "all").
                "image" landmarks are fractions of the frame width & height with origin at the top-left corner of the frame.
                "world" landmarks are 3D coordinates in meters with origin at the center of the hips.
                MediaPipe Holistic provides world landmarks for the pose only, so the hand part of "world" landmarks is zeros.
                "all" concatenates "world" and "image" landmarks (in that order).

        Returns:
            torch.Tensor: A tensor of shape (n_frames, 75 * 5) for "world" or "image" and (n_frames, 75 * 5 * 2) for "all".
        """

        if mediapipe is None:
            raise ImportError(
                "The 'mediapipe' package is required to use the 'MediaPipeHolisticModel'. "
                "Install it using `pip install sign-language-translator[mediapipe]`."
            )

        if landmark_type not in ("world", "image", "all"):
            raise ValueError(
                "landmark_type not supported, use 'world', 'image' or 'all'."
            )

        if hasattr(frame_sequence, "__len__"):
            total_frames = len(frame_sequence)  # type: ignore

        embeddings = []

        # a new tracker for every video: Holistic tracks the person across consecutive frames
        with mediapipe.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.model_complexity,
        ) as holistic:
            for i, frame in enumerate(frame_sequence):
                result = holistic.process(np.ascontiguousarray(frame, dtype=np.uint8))

                embeddings.append(self._create_frame_embedding(result, landmark_type))
                if progress_callback and total_frames:
                    progress_callback({"file": f"{(i + 1) / total_frames:.1%}"})

        return torch.Tensor(embeddings)

    def _flatten_landmarks(self, landmarks, n_landmarks: int) -> List[float]:
        if landmarks is None:
            return [0.0] * n_landmarks * self.N_VALUES

        return [
            value
            for lm in landmarks.landmark
            for value in [lm.x, lm.y, lm.z, lm.visibility, lm.presence]
        ]

    def _create_frame_embedding(self, result, landmark_type: str) -> List[float]:
        embedding = []
        if landmark_type in ("world", "all"):
            embedding.extend(
                self._flatten_landmarks(
                    result.pose_world_landmarks, self.N_POSE_LANDMARKS
                )
            )
            # Holistic has no world landmarks for hands
            embedding.extend([0.0] * self.N_HAND_LANDMARKS * self.N_VALUES * 2)
        if landmark_type in ("image", "all"):
            embedding.extend(
                self._flatten_landmarks(result.pose_landmarks, self.N_POSE_LANDMARKS)
            )
            embedding.extend(
                self._flatten_landmarks(
                    result.left_hand_landmarks, self.N_HAND_LANDMARKS
                )
            )
            embedding.extend(
                self._flatten_landmarks(
                    result.right_hand_landmarks, self.N_HAND_LANDMARKS
                )
            )

        return embedding
