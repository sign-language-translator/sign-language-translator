"""This module provides utility functions for video processing.
"""

from mimetypes import guess_type
from typing import Generator, List, Optional, Sequence, Tuple, Union

import cv2
from numpy import uint8
from numpy.typing import NDArray

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

    Yields:
        NDArray[np.uint8]: numpy arrays representing frames from the video with shape: (height, width, color_channels).

    Raises:
        FileNotFoundError: If the video file is not found or cannot be opened.
    """

    file_type = str(guess_type(path)[0])
    if not file_type.startswith(("image", "video")):
        raise ValueError(f"unknown file type: {file_type}")

    if file_type.startswith("image"):
        frame = cv2.imread(path)
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore

    elif file_type.startswith("video"):
        capture = cv2.VideoCapture(path)
        for _ in range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = capture.read()
            if not ret:
                break

            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore

        capture.release()


def _normalize_args_index_and_timestamp(
    timestamp: Optional[float],
    index: Optional[int],
    max_duration: float,
    max_index: int,
) -> Tuple[float, int]:
    if (timestamp is not None) and index is None:
        if not 0 <= timestamp <= max_duration:
            raise ValueError(f"'{timestamp=}' is not between 0 and {max_duration}")

        return timestamp, round(
            (timestamp / max_duration if max_duration else 1) * max_index
        )

    elif (index is not None) and timestamp is None:
        if not 0 <= index <= max_index:
            raise ValueError(f"'{index=}' is not between 0 and {max_index}")

        return index / (max_index or 1) * max_duration, index

    else:
        raise ValueError("provide either timestamp or index.")


def _validate_and_normalize_slices(
    keys: Union[int, slice, Sequence[Union[int, slice]]], max_n_dims: int = 4
) -> Tuple[slice, ...]:
    if not isinstance(keys, Sequence):
        keys = [keys]

    slices = []
    for i, key in enumerate(keys):
        if key is Ellipsis:
            slices += [slice(None)] * (max_n_dims - len(keys) + 1)
        elif isinstance(key, int):
            slices.append(slice(key, (key + 1) or None))
        elif isinstance(key, slice):
            slices.append(key)
        else:
            raise TypeError(
                f"Invalid argument: {key} at index {i}. Provide either an integer, slice or ellipsis."
            )

    return tuple(slices)
