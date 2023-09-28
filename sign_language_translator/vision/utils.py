"""This module provides utility functions for video processing.
"""

from mimetypes import guess_type
from typing import Generator, List

import cv2
from numpy.typing import NDArray
from numpy import uint8

__all__ = [
    "read_frames_with_opencv",
    "iter_frames_with_opencv",
]


def read_frames_with_opencv(path: str) -> List[NDArray[uint8]]:
    """
    Extracts individual frames from a video file or an image file.

    This function reads a video file using opencv and extracts its frames as numpy arrays.
    It can also read an image file and treat it as a single frame video.

    Args:
        path (str): The path to the video or image file.

    Returns:
        List[NDArray]: A list of numpy arrays, each representing a frame from the video.

    Raises:
        FileNotFoundError: If the video file is not found or cannot be opened.
    """

    return list(iter_frames_with_opencv(path))


def iter_frames_with_opencv(path: str) -> Generator[NDArray[uint8], None, None]:
    """
    Extracts individual frames from a video file or an image file.

    This function reads a video file using opencv and extracts its frames as numpy arrays.
    It can also read an image file and treat it as a single frame video.

    Args:
        path (str): The path to the video or image file.

    Returns:
        Generator[NDArray, None, None]: numpy array representing a frame from the video. Sends None. Returns None.

    Raises:
        FileNotFoundError: If the video file is not found or cannot be opened.
    """

    file_type = str(guess_type(path)[0])
    if not file_type.startswith(("image", "video")):
        raise ValueError(f"unknown file type: {file_type}")

    if file_type.startswith("image"):
        yield cv2.imread(path)[..., ::-1]  # pylint: disable = no-member

    elif file_type.startswith("video"):
        capture = cv2.VideoCapture(path)  # pylint: disable = no-member
        for _ in range(
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # pylint: disable = no-member
        ):
            ret, frame = capture.read()
            if not ret:
                break

            yield frame[..., ::-1]

        capture.release()
