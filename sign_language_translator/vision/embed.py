"""Extract landmarks from video
"""

import os
from typing import Any

import cv2
import mediapipe as mp
import moviepy.editor as mpy
import numpy as np


class MediapipeEmbeddingModel:
    result_keys = [
        "pose_image_landmarks",
        "left_hand_image_landmarks",
        "right_hand_image_landmarks",
        "pose_world_landmarks",
        "left_hand_world_landmarks",
        "right_hand_world_landmarks",
    ]

    def __init__(
        self,
        hands_model_complexity: int = 1,
        pose_model_complexity: int = 2,
        pose_min_detection_confidence: float = 0.5,
        pose_min_tracking_confidence: float = 0.5,
        hands_min_detection_confidence: float = 0.5,
        hands_min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
    ) -> None:
        self.pose_model_complexity = pose_model_complexity
        self.pose_min_detection_confidence = pose_min_detection_confidence
        self.pose_min_tracking_confidence = pose_min_tracking_confidence
        self.hands_model_complexity = hands_model_complexity
        self.hands_min_detection_confidence = hands_min_detection_confidence
        self.hands_min_tracking_confidence = hands_min_tracking_confidence
        self.max_num_hands = max_num_hands

        # Mediapipe Solutions
        self.pose = mp.solutions.pose.Pose(
            model_complexity=self.pose_model_complexity,
            min_detection_confidence=self.pose_min_detection_confidence,
            min_tracking_confidence=self.pose_min_tracking_confidence,
        )
        self.hands = mp.solutions.hands.Hands(
            model_complexity=self.hands_model_complexity,
            min_detection_confidence=self.hands_min_detection_confidence,
            min_tracking_confidence=self.hands_min_tracking_confidence,
            max_num_hands=self.max_num_hands,
        )

    def mplandmark_to_nparray(landmarks, has_visibility):
        return np.array(
            [
                ([l.x, l.y, l.z, l.visibility] if has_visibility else [l.x, l.y, l.z])
                for l in landmarks.landmark
            ]
        )

    def seperate_mp_hand_landmarks(multi_hand_landmarks, multi_handedness, is_mirrored):
        landmarks = {"left": None, "right": None}

        if multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                multi_hand_landmarks, multi_handedness
            ):
                # confidence value????
                # selfie mirror flip????
                if handedness.classification[0].label == "Left":
                    landmarks["left"] = hand_landmarks
                else:
                    landmarks["right"] = hand_landmarks

        return landmarks

    # process a video frame with mediapipe
    def extract_features(self, frame, is_mirrored=False):
        """
        Extracts the pose & hands landmarks from a frame of a video clip.
        Arguments:
            frame - A Numpy array of shape (None, None, 3) representing an image from a video clip containing a person.
        Returns:
            A Tuple of Numpy arrays of shapes (33, 4), (33, 4), (42, 3), (42, 3)
            where each row is a landmark (x, y, z, [visibility]).
            First 2 items in the tuple are pose vectors (image & world)
            followed by 2 hand vectors (image & world) each with first 21 landmarks for left hand and next 21 for right hand.
        """

        landmarks = {key: None for key in MediapipeEmbeddingModel.result_keys}

        # Make Detections
        pose_results = self.pose.process(frame)
        # if is_mirrored:
        #     frame = cv2.flip(frame, 1)
        hand_results = self.hands.process(frame)

        # Extract body Landmarks
        for pose_landmarks, label in [
            (pose_results.pose_landmarks, "image"),
            (pose_results.pose_world_landmarks, "world"),
        ]:
            landmarks[f"pose_{label}_landmarks"] = (
                self.mplandmark_to_nparray(pose_landmarks, has_visibility=True)
                if pose_landmarks
                else np.zeros((33, 4))
            )

        # Extract Hand Landmarks
        for hand_landmarks, label in [
            (hand_results.multi_hand_landmarks, "image"),
            (hand_results.multi_hand_world_landmarks, "world"),
        ]:
            hand_landmarks = self.seperate_mp_hand_landmarks(
                hand_landmarks, hand_results.multi_handedness, is_mirrored
            )
            for handedness in ["left", "right"]:
                landmarks[f"{handedness}_hand_{label}_landmarks"] = (
                    self.mplandmark_to_nparray(
                        hand_landmarks[handedness], has_visibility=False
                    )
                    if hand_landmarks[handedness]
                    else np.zeros((21, 3))
                )

        return landmarks


class SignEmbedding:
    supported_embedding_models = ["mediapipe"]

    def __init__(self, embedding_model="mediapipe", *args: Any, **kwds: Any) -> None:
        if embedding_model in SignEmbedding.supported_embedding_models:
            self.EMBEDDING_MODEL_NAME = embedding_model
            self.load_embedding_model(*args, **kwds)
        else:
            raise NotImplementedError(
                f"the provided embedding model is not supported yet. use from {SignEmbedding.supported_embedding_models}"
            )

    def load_embedding_model(self, *args: Any, **kwargs: Any) -> None:
        if self.EMBEDDING_MODEL_NAME == "mediapipe":
            self.model = MediapipeEmbeddingModel(*args, **kwargs)

    def __call__(self, video, is_mirrored=False):
        return self.extract_embedding(video, is_mirrored=is_mirrored)

    def extract_embedding(self, video, is_mirrored=False):
        if isinstance(video, str):
            video_iterator = self.cv2_videofile_iterator(video)

        elif isinstance(video, mpy.VideoFileClip):
            video_iterator = video.iter_frames()

        # detect np image(s) better
        elif isinstance(video, np.ndarray):
            if (video.ndim == 3) and (3 in video.shape):
                video_iterator = video[np.newaxis, ...]
            elif (video.ndim == 4) and (3 in video.shape):
                video_iterator = video
            else:
                raise ValueError("provide 3 channel numpy image(s)")
        else:
            raise NotImplementedError(
                "provide a str filepath or a moviepy.VideoFileClip object or numpy image(s) array"
            )

        return self.extract_embedding_from_video(
            video_iterator, is_mirrored=is_mirrored
        )

    def cv2_videofile_iterator(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"no file exists at {video_path}")

        cap = cv2.VideoCapture(video_path)
        for frame_number in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            yield cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        cap.release()

    def extract_embedding_from_video(self, video, is_mirrored=False):
        """
        Extracts the pose landmarks from all frames of a video clip.
        Arguments:
            videoPath - Path of video file to be processed.
        Returns:
            Tuple of Numpy arrays of shapes (None, 33, 4), (None, 33, 4), (None, 42, 4), (None, 42, 4)
            where each row is a landmark (x, y, z, visibility).
            First 2 items in the tuple are pose vectors (image & world)
            followed by 2 hand vectors (image & world) each with 21 landmarks for each hand.
        """

        landmarks = {key: [] for key in self.model.result_keys}

        for frame in video:
            frame_landmarks = self.model.extract_features(
                frame,
                is_mirrored=is_mirrored,
            )
            for key in self.model.result_keys:
                landmarks[key].append(frame_landmarks[key])

        return {key: np.stack(_landmarks) for key, _landmarks in landmarks}
