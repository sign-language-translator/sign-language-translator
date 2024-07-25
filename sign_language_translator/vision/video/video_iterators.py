"""
This module provides a flexible and unified interface for working with video frames.

This module defines an abstract class and its implementations for efficient video frame retrieval.
It includes various classes for accessing video frames, seeking, and caching from various sources.

Classes:
- VideoFrames: An abstract base class for video frame retrieval.
- VideoCaptureFrames: A class for efficient video frame retrieval from a video file using OpenCV.
- SequenceFrames: A class for representing a sequence of video frames.
- IterableFrames: Represents an iterable video frame source, allowing random access to frames by index or timestamp.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from os.path import abspath
from time import time
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from sign_language_translator.vision.utils import _normalize_args_index_and_timestamp


class VideoFrames(ABC):
    """
    Abstract Base Class for Video Frames

    VideoFrames is an abstract base class that defines a common interface for video frame retrieval.
    Subclasses of VideoFrames are expected to implement methods for accessing video frames,
    releasing resources, and providing information about the video.

    Methods:
    - get_frame(timestamp: float = None, index: int = None) -> NDArray[np.uint8]:
        Get a frame at a given timestamp or index from the video object.

    - close():
        Release the resources occupied by the object.

    - __len__() -> int:
        Return the number of frames in the video object.

    Properties:
    - height: int
        Number of pixels vertically present in the video frame.

    - width: int
        Number of pixels horizontally present in the video frame.

    - n_channels: int
        Number of color channels in the video frames.
    """

    @abstractmethod
    def get_frame(
        self, timestamp: Optional[float] = None, index: Optional[int] = None
    ) -> NDArray[np.uint8]:
        """Get a frame at a given timestamp or index from the video object."""

    @abstractmethod
    def close(self):
        """Release the resources occupied by the object."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of frames in the video object"""

    @property
    @abstractmethod
    def height(self) -> int:
        """Number of pixels vertically present in the video frame."""

    @property
    @abstractmethod
    def width(self) -> int:
        """Number of pixels horizontally present in the video frame."""

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of color channels in the video frames."""


class VideoCaptureFrames(VideoFrames):
    """
    A class for efficient video frame retrieval from a video file using OpenCV.

    This class extends the functionality of the VideoFrames abstract class to provide features
    for video frame access, seeking, and caching.

    Args:
        path (str): The path to the video file.
        use_cache (bool, optional): Enable or disable frame caching. Default is False.
        cache_len (int, optional): Maximum number of frames to cache if use_cache is enabled.
            Default is 256.
        **kwargs: Additional keyword arguments to pass to the base VideoFrames class.

    Attributes:
        path (str): The path to the video file.
        fps (float): Frames per second of the video.
        total_frames (int): Total number of frames in the video.
        _width (int): Width of video frames.
        _height (int): Height of video frames.
        fourcc (int): FourCC code representing the video codec.
        duration (float): Duration of the video in seconds.
        _frames_cache (dict): A dictionary for frame caching.
        use_cache (bool): True if frame caching is enabled, False otherwise.
        _max_cache_len (int): Maximum number of frames to cache.
        _n_channels (int): Number of color channels in the video frames.

    Methods:
        get_frame(timestamp: float = None, index: int = None) -> NDArray[np.uint8]:
            Retrieve a video frame based on either a timestamp or an index.

        current_index() -> int:
            Get the current index of the video frame being read.

        seek(timestamp: float = None, index: int = None):
            Move the video frame position to the specified timestamp or index.

        read_frame() -> NDArray[np.uint8] | None:
            Read and return the next frame from the video.

        close():
            Close the video capture and release associated resources.

    Notes:
        - Frame caching can improve performance by storing previously accessed frames in memory.
        - The seek method employs efficient seeking techniques based on time and frame index.
        - When finished, remember to call the close method to release video resources.

    Example:

    .. code-block:: python

        video = VideoCaptureFrames("video.mp4", use_cache=True)
        frame = video.get_frame(timestamp=10.0)
        video.seek(index=100)
        frame = video.read_frame()
        video.close()
    """

    _seek_time: float = 0.1
    _read_time: float = 0.005

    def __init__(self, path: str, use_cache=False, cache_len=256, **kwargs):
        self.path = abspath(path)
        self.video_capture = cv2.VideoCapture(self.path)

        self.fps = float(self.video_capture.get(cv2.CAP_PROP_FPS)) or 30
        self.total_frames = int(
            self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )  # bug: not accurate, gotta fix on runtime.
        self._width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = int(self.video_capture.get(cv2.CAP_PROP_FOURCC))

        self._frames_cache = {}
        self.use_cache = use_cache
        self._max_cache_len = cache_len

        frame_shape = self.get_frame(0).shape
        self._n_channels = frame_shape[2] if len(frame_shape) == 3 else 1

    def get_frame(
        self,
        timestamp: Optional[float] = None,
        index: Optional[int] = None,
    ) -> NDArray[np.uint8]:
        """
        Retrieve a video frame at a specified timestamp or index.

        Args:
            timestamp (float | None): The timestamp in seconds.
            index (int | None): The frame index.

        Returns:
            NDArray[np.uint8]: The video frame as a NumPy array.

        Raises:
            RuntimeError: If frame retrieval fails.
        """

        # arguments
        timestamp, _index = _normalize_args_index_and_timestamp(
            timestamp, index, self.duration, self.total_frames - 1
        )

        # cache
        if self.use_cache and _index in self._frames_cache:
            return self._frames_cache[_index]

        # read
        self.seek(index=_index)
        frame = self.read_frame()

        # validate
        if frame is None:
            raise RuntimeError(
                f'Error reading frame from cv2.VideoCapture("{self.path}") at '
                + (f"{timestamp = }." if index is None else f"{index = }.")
            )

        # cache
        if self.use_cache:
            if len(self._frames_cache) > self._max_cache_len:
                self._frames_cache.pop(list(self._frames_cache.keys())[0])

            self._frames_cache[_index] = frame

        return frame

    @property
    def current_index(self) -> int:
        """Where the VideoCapture is currently pointing to."""
        return int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))

    def seek(self, timestamp: Optional[float] = None, index: Optional[int] = None):
        """
        Seek to a specified timestamp or frame index.

        Args:
            timestamp (float | None): The timestamp in seconds.
            index (int | None): The frame index.

        Returns:
            None
        """
        timestamp, target_index = _normalize_args_index_and_timestamp(
            timestamp, index, self.duration, self.total_frames - 1
        )

        if target_index == self.current_index:
            return

        new_seek_time = None
        new_read_time = None
        max_frames_to_read = int(self._seek_time / self._read_time * 0.9)

        if self.current_index < target_index < self.current_index + max_frames_to_read:
            # go forward by reading frames because it is faster than seeking
            n_frames = target_index - self.current_index
            start_time = time()
            for _ in range(n_frames):
                self.read_frame()
            new_read_time = (time() - start_time) / n_frames
        else:
            # seek using cv2
            start_time = time()
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            new_seek_time = time() - start_time

        # updated seek and read times using exponential moving average
        if new_seek_time is not None:
            self._seek_time = self._seek_time * 0.90 + 0.10 * new_seek_time
        if new_read_time is not None:
            self._read_time = self._read_time * 0.95 + 0.05 * new_read_time

    def read_frame(self) -> Optional[NDArray[np.uint8]]:
        """
        Read the next frame from the video.

        Returns:
            NDArray[np.uint8] | None: The next video frame as a NumPy array,
                or None if no more frames are available.
        """
        success, frame = self.video_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if success else None

        return frame  # type: ignore

    def __len__(self) -> int:
        return self.total_frames

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def duration(self) -> float:
        return self.total_frames / self.fps

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        """Release the video capture resource and clear the frame cache."""
        self.video_capture.release()
        self._frames_cache = {}

    def __copy__(self) -> VideoCaptureFrames:
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        # TODO: decide if create new videoCapture to prevent conflicts or point to old because this is shallow copy
        # new.video_capture = self.video_capture
        new.video_capture = cv2.VideoCapture(self.path)
        return new

    def __deepcopy__(self, memo):
        new_instance = self.__class__(
            self.path,
            use_cache=self.use_cache,
            cache_len=self._max_cache_len,
        )
        memo[id(self)] = new_instance
        return new_instance

    # TODO: make the class pickle-able


class SequenceFrames(VideoFrames):
    """A class for representing a sequence of video frames.

    This class extends the VideoFrames abstract class to work with a predefined sequence
    of frames, allowing easy access to individual frames within the sequence.

    Args:
        frames (Sequence[NDArray[np.uint8]]): A sequence of video frames, where each
            frame is represented as a NumPy array with data type `np.uint8`.
        fps (float | None, optional): The frames per second (FPS) of the video.
            If not specified, it can be set to None.

    Attributes:
        frames (Sequence[NDArray[np.uint8]]): The sequence of video frames.
        fps (float): The frames per second (FPS) of the video. Defaults to 30.0.

    Note:
        The SequenceFrames class inherits from the VideoFrames class.
    """

    def __init__(
        self,
        frames: Sequence[NDArray[np.uint8]],
        fps: float = 30.0,
    ) -> None:
        self.frames = frames
        self.fps = abs(fps) or 30.0

        frame_shape = self.frames[0].shape
        self._height, self._width = frame_shape[:2]
        self._n_channels = frame_shape[2] if len(frame_shape) == 3 else 1

    def get_frame(
        self, timestamp: Optional[float] = None, index: Optional[int] = None
    ) -> NDArray[np.uint8]:
        """
        Retrieve a video frame based on the specified timestamp or index.

        Args:
            timestamp (float | None, optional): The timestamp in seconds at which to
                retrieve the frame. If not provided, index is used.
            index (int | None, optional): The index of the frame to retrieve. If not
                provided, timestamp is used.

        Returns:
            NDArray[np.uint8]: The video frame as a NumPy array with data type `np.uint8`.
        """

        timestamp, _index = _normalize_args_index_and_timestamp(
            timestamp, index, self.duration, self.total_frames - 1
        )
        return self.frames[_index]

    def close(self):
        """
        Close the SequenceFrames instance by clearing the frames.

        This method releases resources associated with the frames by clearing the
        frames list. After calling this method, the frames will no longer be available.
        """
        self.frames = []

    def __len__(self) -> int:
        return self.total_frames

    @property
    def total_frames(self) -> int:
        """total number of frames present in the sequence. (dimension=0)"""
        return len(self.frames)

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def duration(self) -> float:
        """total time that the frames would take to play in a sequence. depends on fps."""
        return (self.total_frames / self.fps) if self.fps else float("inf")


class IterableFrames(VideoFrames):
    """
    Represents a read-once iterable video frame source, allowing random access to frames by index or timestamp.

    This class extends the VideoFrames abstract class and is specifically designed to work with iterable sources of video frames such as generators.
    It maintains an internal cache of frames to efficiently access frames by index or timestamp.

    Args:
        frames (Iterable[NDArray[np.uint8]]): An iterable source of video frames.
        total_frames (int): The total number of frames in the video.
        fps (float, optional): The frames per second of the video. Defaults to 30.0.
        use_cache (bool, optional): Whether to store the frames when they have been read from the iterable. Defaults to True.

    Attributes:
        frames_iterable (iter): An iterator over the provided frames.
        frames_cache (dict): A cache to store frames for efficient retrieval.
        fps (float): The frames per second of the video.
        total_frames (int): The total number of frames in the video.

    Methods:
        get_frame(timestamp: float = None, index: int = None) -> NDArray[np.uint8]:
            Retrieve a video frame by specifying either a timestamp or an index.

        close():
            Close the video frame source, resetting the frames_iterable and clearing the frames_cache.
    """

    def __init__(
        self,
        frames: Iterable[NDArray[np.uint8]],
        total_frames: int,
        fps: float = 30.0,
        use_cache=True,
    ) -> None:
        self.frames_iterable = iter(frames)
        self.frames_cache = {}
        self.fps = abs(fps) or 30.0
        self.use_cache = use_cache

        if (not isinstance(total_frames, int)) or total_frames <= 0:
            raise ValueError("total_frames must be a positive integer.")

        self.total_frames = total_frames

        self._current_index = 0

        frame = self.get_frame(index=0)
        self._height, self._width = frame.shape[:2]
        self._n_channels = frame.shape[2] if len(frame.shape) == 3 else 1

    def get_frame(
        self, timestamp: Optional[float] = None, index: Optional[int] = None
    ) -> NDArray[np.uint8]:
        """
        Retrieve a video frame by specifying either a timestamp or an index.

        Args:
            timestamp (float | None, optional): The timestamp (in seconds) of the desired frame. If provided, it will
                be used to locate the frame in the video. Defaults to None.
            index (int | None, optional): The index of the desired frame. If provided, it will be used to locate the
                frame in the video. Defaults to None.

        Returns:
            NDArray[np.uint8]: The video frame as a NumPy array of unsigned 8-bit integers.

        Raises:
            RuntimeError: If the specified timestamp or index is out of range or if there is an error reading the frame
                from the frames_iterable.

        Note:
            You can retrieve frames by either timestamp or index. The timestamp allows you to seek to a specific point
            in time, while the index allows you to access frames in a sequential order.
        """

        timestamp, target_index = _normalize_args_index_and_timestamp(
            timestamp, index, self.duration, self.total_frames - 1
        )

        for i in range(self._current_index, target_index + 1):
            frame = next(self.frames_iterable, None)
            if frame is not None:
                self.frames_cache[i] = frame
                self._current_index += 1
                continue

            # end of Video
            self.total_frames = self._current_index
            raise RuntimeError(
                "Error reading frame from frames_iterable at "
                + (f"{timestamp = }." if index is None else f"{index = }.")
            )

        return (
            self.frames_cache[target_index]
            if self.use_cache or target_index == 0
            else self.frames_cache.pop(target_index)
        )

    def close(self):
        """Close the video frame source, resetting the frames_iterable and clearing the frames_cache."""
        self.frames_iterable = iter([])
        self.frames_cache = {}

    def __len__(self) -> int:
        return self.total_frames

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def duration(self) -> float:
        """total time that the frames would take to play in a sequence. depends on fps."""
        return (self.total_frames / self.fps) if self.fps else float("inf")

    def __copy__(self) -> IterableFrames:
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


import math
from typing import Callable, List


class VideoSource(VideoFrames):
    def __init__(
        self,
        source: VideoFrames,
        start_index: int = 0,
        end_index: Optional[int] = None,
        step_size: int = 1,
        transformations: Optional[List[Callable]] = None,
    ) -> None:
        self.source = source
        if start_index < 0 or start_index >= len(self.source):
            raise ValueError(
                "start_index cannot be negative or larger than the length of the source."
            )
        self.start_index = start_index
        self.end_index = end_index or len(source) - 1
        if self.end_index < start_index or self.end_index >= len(self.source):
            raise ValueError(
                "end_index cannot be negative, less than start_index or larger than the length of the source."
            )
        self._step_size = step_size or 1
        self.transformations = transformations or []

        self.__current_index = 0

    def __iter__(self):
        self.__current_index = 0
        return self

    def __next__(self) -> NDArray[np.uint8]:
        if 0 <= self.__current_index < len(self):
            frame = self.get_frame(index=self.__current_index)
            self.__current_index += 1
            return frame

        raise StopIteration

    @property
    def step_size(self) -> int:
        return self._step_size

    @step_size.setter
    def step_size(self, value: int):
        if not isinstance(value, int):
            raise ValueError("step_size must be an integer.")
        if value > 0:
            self.end_index -= (self.end_index - self.start_index) % value
        elif value < 0:
            self.start_index -= (self.end_index - self.start_index) % value
        else:
            raise ValueError("step_size cannot be 0")

        self._step_size = value

    def base_index(self, relative_index: int) -> int:
        return (
            self.start_index + (relative_index * self.step_size)
            if self.step_size > 0
            else self.end_index + (relative_index * self.step_size)
        )

    def get_frame(
        self, timestamp: Optional[float] = None, index: Optional[int] = None
    ) -> NDArray[np.uint8]:
        timestamp, index = _normalize_args_index_and_timestamp(
            timestamp,
            index,
            getattr(self.source, "duration", float("inf")),
            len(self) - 1,
        )

        if (base_index := self.base_index(index)) <= self.end_index:
            frame = self.source.get_frame(index=base_index)
            for transformation in self.transformations:
                frame = transformation(frame)
            return frame

        raise IndexError(
            "Error reading frame at "
            + (f"{timestamp = }." if index is None else f"{index = }.")
        )

    def close(self):
        self.source.close()

    def __len__(self) -> int:
        return math.ceil((self.end_index - self.start_index + 1) / abs(self.step_size))

    @property
    def height(self) -> int:
        return self.source.height

    @property
    def width(self) -> int:
        return self.source.width

    @property
    def n_channels(self) -> int:
        return self.source.n_channels
