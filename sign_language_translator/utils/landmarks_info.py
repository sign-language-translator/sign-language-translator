"""encapsulates landmarks specific information such as how to connect them and color those connections
"""

from typing import List, Tuple, Union

import numpy as np
import torch


class LandmarksInfo:
    """encapsulates the ways to connect landmarks and color those connections"""

    POSE_CONNECTIONS = [
        [0, 1],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 7],
        [4, 5],
        [5, 6],
        [6, 8],
        [9, 10],
        [11, 12],
        [11, 13],
        [11, 23],
        [12, 14],
        [12, 24],
        [13, 15],
        [14, 16],
        [15, 17],
        [15, 19],
        [15, 21],
        [16, 18],
        [16, 20],
        [16, 22],
        [17, 19],
        [18, 20],
        [23, 24],
        [23, 25],
        [24, 26],
        [25, 27],
        [26, 28],
        [27, 29],
        [27, 31],
        [28, 30],
        [28, 32],
        [29, 31],
        [30, 32],
    ]
    HAND_CONNECTIONS = [
        [0, 1],
        [0, 5],
        [0, 17],
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 6],
        [5, 9],
        [6, 7],
        [7, 8],
        [9, 10],
        [9, 13],
        [10, 11],
        [11, 12],
        [13, 14],
        [13, 17],
        [14, 15],
        [15, 16],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    N_POSE_LANDMARKS = 33
    N_HAND_LANDMARKS = 21

    POSE_LEFT_WRIST_INDEX = 15
    POSE_RIGHT_WRIST_INDEX = 16
    HAND_WRIST_INDEX = 0

    BOTH_HANDS_CONNECTIONS = (
        sorted(HAND_CONNECTIONS)
        + (np.array(sorted(HAND_CONNECTIONS)) + N_HAND_LANDMARKS).tolist()
    )
    ALL_CONNECTIONS = (
        sorted(POSE_CONNECTIONS)
        + (np.array(BOTH_HANDS_CONNECTIONS) + N_POSE_LANDMARKS).tolist()
    )

    N_BOTH_HANDS_LANDMARKS = 2 * N_HAND_LANDMARKS
    N_ALL_LANDMARKS = N_POSE_LANDMARKS + 2 * N_HAND_LANDMARKS

    N_POSE_FEATURES = N_POSE_LANDMARKS * 4
    N_HAND_FEATURES = N_HAND_LANDMARKS * 3
    N_BOTH_HANDS_FEATURES = N_HAND_FEATURES * 2
    N_ALL_FEATURES = N_POSE_FEATURES + N_BOTH_HANDS_FEATURES

    VISIBILITY_INDEXES = (np.arange(N_POSE_LANDMARKS) * 4 + 3).tolist()

    def infer_connections(n_landmarks: int) -> List[List[int]]:
        """find an appropriate connections list based on number of landmarks

        Args:
            n_landmarks (int): number of landmarks

        Returns:
            List[List[int]]: Pairs of indices of the landmarks array to be connected by a line. Defaults to [].
        """

        return (
            LandmarksInfo.POSE_CONNECTIONS
            if n_landmarks == LandmarksInfo.N_POSE_LANDMARKS
            else LandmarksInfo.HAND_CONNECTIONS
            if n_landmarks == LandmarksInfo.N_HAND_LANDMARKS
            else LandmarksInfo.BOTH_HANDS_CONNECTIONS
            if n_landmarks == LandmarksInfo.N_BOTH_HANDS_LANDMARKS
            else LandmarksInfo.ALL_CONNECTIONS
            if n_landmarks == LandmarksInfo.N_ALL_LANDMARKS
            else []
        )

        # def repeat(lt, n_final):
        #     num_repeats = int(n_final / len(lt))

        #     if num_repeats == 1:
        #         return lt

        #     arr = np.array(lt)
        #     return np.concatenate([arr + i for i in range(num_repeats)]).tolist()

        # print(n_landmarks)
        # return (
        #     repeat(LandmarksInfo.POSE_CONNECTIONS, n_landmarks)
        #     if n_landmarks % LandmarksInfo.N_POSE_LANDMARKS == 0
        #     else repeat(LandmarksInfo.HAND_CONNECTIONS, n_landmarks)
        #     if n_landmarks % LandmarksInfo.N_HAND_LANDMARKS == 0
        #     else repeat(LandmarksInfo.ALL_CONNECTIONS, n_landmarks)
        #     if n_landmarks % LandmarksInfo.N_ALL_LANDMARKS == 0
        #     else []
        # )

    class Colors:
        """encapsulates colors applied to landmarks"""

        BLACK = (0, 0, 0)
        RED = (1, 0, 0)
        GREEN = (0, 1, 0)
        BLUE = (0, 0, 1)
        CYAN = (0, 1, 1)
        MAGENTA = (1, 0, 1)

        DEFAULT_CONNECTION = BLACK
        DEFAULT_LANDMARK = RED
        LEFT = CYAN
        RIGHT = MAGENTA
        THUMB = GREEN
        PINKY = BLUE

        def _get_default_connection_color(
            start: int = None, end: int = None
        ) -> Tuple[float, float, float]:
            """get a color based on which part of body is being drawn

            Args:
                start (int, optional): landmark index where line starts. Defaults to None.
                end (int, optional): landmark index where line ends. Defaults to None.

            Returns:
                Tuple[float, float, float]: RGB color value [0 to 1]
            """

            return LandmarksInfo.Colors.DEFAULT_CONNECTION

        def _get_pose_connection_color(
            start: int, end: int
        ) -> Tuple[float, float, float]:
            """get a color based on which part of body is being drawn

            Args:
                start (int, optional): landmark index where line starts. Defaults to None.
                end (int, optional): landmark index where line ends. Defaults to None.

            Returns:
                Tuple[float, float, float]: RGB color value [0 to 1]
            """

            return (
                LandmarksInfo.Colors.LEFT
                if (start % 2 == 1 and end % 2 == 1)
                else LandmarksInfo.Colors.RIGHT
                if (start % 2 == 0 and end % 2 == 0)
                else LandmarksInfo.Colors._get_default_connection_color(start, end)
            )

        def _get_hand_connection_color(
            start: int, end: int
        ) -> Tuple[float, float, float]:
            """get a color based on which part of body is being drawn

            Args:
                start (int, optional): landmark index where line starts. Defaults to None.
                end (int, optional): landmark index where line ends. Defaults to None.

            Returns:
                Tuple[float, float, float]: RGB color value [0 to 1]
            """

            return (
                LandmarksInfo.Colors.THUMB
                if end <= 4
                else LandmarksInfo.Colors.PINKY
                if start >= 17
                else LandmarksInfo.Colors._get_default_connection_color(start, end)
            )

        def _get_both_hands_connection_color(
            start: int, end: int
        ) -> Tuple[float, float, float]:
            """get a color based on which part of body is being drawn

            Args:
                start (int, optional): landmark index where line starts. Defaults to None.
                end (int, optional): landmark index where line ends. Defaults to None.

            Returns:
                Tuple[float, float, float]: RGB color value [0 to 1]
            """

            return (
                LandmarksInfo.Colors._get_hand_connection_color(start, end)
                if start < LandmarksInfo.N_HAND_LANDMARKS
                and end < LandmarksInfo.N_HAND_LANDMARKS
                else LandmarksInfo.Colors._get_hand_connection_color(
                    start - LandmarksInfo.N_HAND_LANDMARKS,
                    end - LandmarksInfo.N_HAND_LANDMARKS,
                )
            )

        def _get_all_connection_color(
            start: int, end: int
        ) -> Tuple[float, float, float]:
            """get a color based on which part of body is being drawn

            Args:
                start (int, optional): landmark index where line starts. Defaults to None.
                end (int, optional): landmark index where line ends. Defaults to None.

            Returns:
                Tuple[float, float, float]: RGB color value [0 to 1]
            """

            if (
                start < LandmarksInfo.N_POSE_LANDMARKS
                and end < LandmarksInfo.N_POSE_LANDMARKS
            ):
                color = LandmarksInfo.Colors._get_pose_connection_color(start, end)

            elif (
                start < LandmarksInfo.N_ALL_LANDMARKS
                and end < LandmarksInfo.N_ALL_LANDMARKS
            ):
                color = LandmarksInfo.Colors._get_both_hands_connection_color(
                    start - LandmarksInfo.N_POSE_LANDMARKS,
                    end - LandmarksInfo.N_POSE_LANDMARKS,
                )

            else:
                color = LandmarksInfo.Colors._get_default_connection_color(start, end)

            return color

    class Reshaper:
        """[USE THE SAME OBJECT TO FOLD & UNFOLD and only ONE array at a time.]
        Convert arrays into (..., n_landmarks, (x,y,z)) and back (..., n_landmarks*(3|4)).
        The problem is created by the visibility feature (x,y,z, v )
        which is available for pose only so stacking in 3D becomes jagged."""

        def __init__(self) -> None:
            """__init__ function. initializes variables that store infromation removed during folding with None."""

            self.last_dim_size = None
            self.landmarks_visibility_backup = None

        def fold(
            self, landmarks: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
            """Turn a (..., m*n+k) array into (..., m, n). The cut out k values are stored within this object.

            Args:
                landmarks (Union[np.ndarray, torch.Tensor]): the (x,y,z,[v]) body and hands landmarks of a body.

            Raises:
                NotImplementedError: use numpy.ndarray or torch.Tensor landmarks

            Returns:
                Union[np.ndarray, torch.Tensor]: the folded (x,y,z) body and hands landmarks of a body.
            """

            if isinstance(landmarks, torch.Tensor):
                concatenate = torch.concatenate
            elif isinstance(landmarks, np.ndarray):
                concatenate = np.concatenate
            else:
                raise NotImplementedError("use numpy.ndarray or torch.Tensor")

            self.last_dim_size = landmarks.shape[-1]

            if self.last_dim_size == 4:
                self.landmarks_visibility_backup = landmarks[..., 3:]#.copy()
                return landmarks[..., :3]

            elif self.last_dim_size == LandmarksInfo.N_POSE_FEATURES:
                landmarks = landmarks.reshape(landmarks.shape[:-1] + (-1, 4))
                self.landmarks_visibility_backup = landmarks[..., 3:]#.copy()
                return landmarks[..., :3]

            elif self.last_dim_size == LandmarksInfo.N_ALL_FEATURES:
                pose = landmarks[..., : LandmarksInfo.N_POSE_FEATURES]
                rest = landmarks[..., LandmarksInfo.N_POSE_FEATURES :]

                pose = pose.reshape(pose.shape[:-1] + (-1, 4))
                rest = rest.reshape(rest.shape[:-1] + (-1, 3))

                self.landmarks_visibility_backup = pose[..., 3:]#.copy()
                return concatenate([pose[..., :3], rest], axis=-2)

            elif self.last_dim_size in [
                LandmarksInfo.N_HAND_FEATURES,
                LandmarksInfo.N_BOTH_HANDS_FEATURES,
            ]:
                landmarks = landmarks.reshape(landmarks.shape[:-1] + (-1, 3))

            self.landmarks_visibility_backup = None

            return landmarks

        def unfold(
            self, landmarks: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
            """turn a (..., m, n) array into (..., m*n+k). The k values are pasted from values stored within this object during folding.

            Args:
                landmarks (Union[np.ndarray, torch.Tensor]): the folded (x,y,z) body and hands landmarks of a body.

            Raises:
                NotImplementedError: use numpy.ndarray or torch.Tensor landmarks

            Returns:
                Union[np.ndarray, torch.Tensor]: the unfolded (x,y,z,[v]) body and hands landmarks of a body.
            """

            if isinstance(landmarks, torch.Tensor):
                concatenate = torch.concatenate
            elif isinstance(landmarks, np.ndarray):
                concatenate = np.concatenate
            else:
                raise NotImplementedError("use numpy.ndarray or torch.Tensor")

            if self.last_dim_size == 4:
                unfolded = concatenate(
                    [landmarks, self.landmarks_visibility_backup], axis=-1
                )

            elif self.last_dim_size == LandmarksInfo.N_POSE_FEATURES:
                unfolded = concatenate(
                    [landmarks, self.landmarks_visibility_backup], axis=-1
                )
                unfolded = unfolded.reshape(landmarks.shape[:-2] + (-1,))

            elif self.last_dim_size == LandmarksInfo.N_ALL_FEATURES:
                pose = concatenate(
                    [
                        landmarks[..., : LandmarksInfo.N_POSE_LANDMARKS, :],
                        self.landmarks_visibility_backup,
                    ],
                    axis=-1,
                )
                rest = landmarks[..., LandmarksInfo.N_POSE_LANDMARKS :, :]

                pose = pose.reshape(pose.shape[:-2] + (-1,))
                rest = rest.reshape(rest.shape[:-2] + (-1,))

                unfolded = concatenate([pose, rest], axis=-1)

            elif self.last_dim_size in [
                LandmarksInfo.N_HAND_FEATURES,
                LandmarksInfo.N_BOTH_HANDS_FEATURES,
            ]:
                unfolded = landmarks.reshape(landmarks.shape[:-2] + (-1,))

            else:
                unfolded = landmarks

            self.last_dim_size = None
            self.landmarks_visibility_backup = None

            return unfolded
