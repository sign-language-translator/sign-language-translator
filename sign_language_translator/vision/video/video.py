from __future__ import annotations

__all__ = [
    "Video",
]

import struct
from copy import copy, deepcopy
from mimetypes import guess_type
from os import makedirs, remove
from os.path import abspath, dirname, isfile, join
from typing import (
    Callable,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from warnings import warn

import cv2  # TODO: Compile OpenCV with GPU
import numpy as np
import torch
from numpy.typing import NDArray
from tqdm.auto import tqdm

from sign_language_translator.config.assets import Assets
from sign_language_translator.config.enums import SignFormats
from sign_language_translator.utils import in_jupyter_notebook, validate_path_exists
from sign_language_translator.vision.sign.sign import Sign
from sign_language_translator.vision.utils import (
    _normalize_args_index_and_timestamp,
    _validate_and_normalize_slices,
)
from sign_language_translator.vision.video.display import VideoDisplay
from sign_language_translator.vision.video.video_iterators import (
    IterableFrames,
    SequenceFrames,
    VideoCaptureFrames,
    VideoFrames,
)


class Video(Sign, VideoFrames):
    """A class to represent and manipulate videos or sequence of images.
    Inherits from the `slt.vision.Sign` class.

    Args:
        sign (Union[ str, Sequence[Union[NDArray[np.uint8], torch.Tensor]], NDArray[np.uint8], torch.Tensor, Generator[NDArray[np.uint8], None, None], ]): The video source. Can be a path to a video or image file, a sequence of frames, a single frame, or a generator of frames.

    Methods:
        load(): Load a video from the specified path.
        load_asset(): Load a video asset by its label from the built-in datasets.
        name(): Returns the name of the sign format.
        numpy(): Convert the video frames to a NumPy array.
        torch(): Convert the video frames to a PyTorch tensor.
        trim(): Cut the video to a specific time range or index range.
        stack(): Stack a list of Video objects along a specified dimension.
        concatenate(): Concatenate a sequence of Video objects into a single Video.
        append(): Append another Video object to the end of the current Video.
        transform(): Apply a transformation function to the individual frames of this object (lazy & in-place).
        iter_frames(): Iterate over the frames in the video from a start point to an end point with a certain step size.
        __iter__(): Makes the Video object an iterable
        __next__(): Get the next frame in the video.
        get_frame(): Get a frame from the video at a specific timestamp or index.
        __getitem__(): Get a frame or a sub-clip or a cropped clip from the video.
        frames_grid(): Create a grid of frames from the video as a single stacked image.
        show(): Display the video.
        show_frame(): Display a specific frame from the video.
        show_frames_grid(): Display a grid of frames from the video as a single stacked image.
        save(): Save the video frames to a file.
        save_frame(): Save a single frame from the video object to the specified path.
        save_frames_grid(): Write a grid of frames from the video to a single image file.
        close(): Release the resources occupied by the object and reset the properties.

    Properties:
        shape: Tuple of array dimensions (n_frames, height, width, n_channels).
        n_frames: The number of frames in the video.
        height: Number of vertical pixels in a video frame.
        width: Number of horizontal pixels in a video frame.
        n_channels: Number of color channels in a video frame.
        duration: Total time that the frames would take to play in a sequence.
        codec: The video codec used to encode the video.

    Example:

        .. code-block:: python

            import sign_language_translator as slt

            # Load a video from a file
            # video = slt.Video("path/to/video.mp4")
            # video = slt.Video("path/to/image.jpg")

            # load from a numpy array
            import numpy as np
            noise = slt.Video(np.random.randint(0, 255, (20,100,160,3), dtype=np.uint8), fps=5)

            # load a dataset file (auto-download)
            video = slt.Video.load_asset("videos/pk-hfad-1_airplane.mp4")
            print(video.duration, video.n_frames)  # 1.72 43

            # trim and concatenate
            video = video + video.trim(start_time=0.5, end_time=1.0)

            # crop
            video = video[:, 100:-100, 50:, :]

            # apply a transformation (flip horizontally)
            video.transform(lambda frame: frame[..., ::-1, :])

            # save & display
            video.save("new_video.mp4", overwrite=True)
            video.save_frames_grid("frames_grid.jpg")
            video.show()
    """

    def __init__(
        self,
        sign: Union[
            str,
            Sequence[Union[NDArray[np.uint8], torch.Tensor]],
            NDArray[np.uint8],
            torch.Tensor,
            Generator[NDArray[np.uint8], None, None],
        ],
        **kwargs,
    ) -> None:
        """Create a new Video object from a video file, a sequence of frames, or a generator of frames.

        Args:
            sign (Union[ str, Sequence[Union[NDArray[np.uint8], torch.Tensor]], NDArray[np.uint8], torch.Tensor, Generator[NDArray[np.uint8], None, None], ]): The video source. Can be a path to a video or image file, a sequence of frames, a single frame, or a generator of frames.
        """
        self._path: Optional[str] = None

        # ToDo: use list of sources instead of linked list of videos.
        # source
        self._source = None
        self.__next: Optional[Video] = None

        # play
        self._source_start_index: int = 0
        # TODO: handle len & end_index for ||step|| > 1
        self._default_step_size: int = 1
        self.transformations: List[Callable] = []

        # properties
        self.fps: float = 30.0
        self.fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        self._height = None
        self._width = None
        self._n_channels = None

        self.__initialize_from_arguments(sign, **kwargs)
        # TODO: from URL
        self._source_end_index = len(self.source) - 1
        self.__update_shape_components()

        # TODO: self.gpu_frame = cv2.cuda_GpuMat()
        self.__idx = 0

    @staticmethod
    def name() -> str:
        return SignFormats.VIDEO.value

    # ========================= #
    #    Iterate / get frame    #
    # ========================= #

    def get_frame(
        self, timestamp: Optional[float] = None, index: Optional[int] = None
    ) -> NDArray[np.uint8]:
        """Get a frame from the video at a specific timestamp or index.

        Args:
            timestamp (Optional[float], optional): The timestamp of the frame to get. If not provided, `index` will be used. If negative, selects the frame from the end of the video. Defaults to None.
            index (Optional[int], optional): The index of the frame to get. If not provided, `timestamp` will be used. if negative, selects the frame from the end of the video (-1 is the last frame). Defaults to None.

        Raises:
            IndexError: If the specified timestamp or index is out of range.

        Returns:
            NDArray[np.uint8]: The 3D RGB frame at the specified timestamp or index from the video.
        """
        # allow negative indexing
        if index and index < 0:
            index = index + len(self)
        if timestamp and timestamp < 0:
            timestamp = timestamp + self.duration

        # validate arguments
        timestamp, _index = _normalize_args_index_and_timestamp(
            timestamp,
            index,
            self.duration if timestamp or self.fps else float("inf"),
            len(self) - 1,
        )

        # get frame
        node, relative_index = self.__get_node(_index)
        if node is not None:
            frame = node.source.get_frame(index=relative_index)
            for transformation in node.transformations:
                frame = transformation(frame)
            return frame

        # error
        raise IndexError(
            "Error reading frame at "
            + (f"{timestamp = }" if index is None else f"{index = }")
        )

    def trim(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
    ) -> Video:
        """Cut the video to a specific time range or index range.
        Allows for negative indexing and negative timestamps which select the frame from the end of the video.
        for example: cut a minute long video with start at 5 sec and end at -15 sec to get a 40 sec video.

        Args:
            start_time (Optional[float], optional): The start time in seconds of the trimmed video. If only `end_time` is provided, trim will start from 0. Defaults to None.
            end_time (Optional[float], optional): The end time in seconds of the trimmed video. If only `start_time` is provided, trim will end at video duration. Defaults to None.
            start_index (Optional[int], optional): The start index of the trimmed video. If only `end_index` is provided, trim will start from 0. Defaults to None.
            end_index (Optional[int], optional): The end index of the trimmed video. If only `start_index` is provided, trim will end at last index (inclusive). Defaults to None.

        Raises:
            ValueError: If the start is not smaller than the end.
            IndexError: If the start index or end index is out of range.

        Returns:
            Video: The trimmed video.
        """

        # allow negative indexing
        if start_index and start_index < 0:
            start_index = start_index + len(self)
        if end_index and end_index < 0:
            end_index = end_index + len(self)
        if start_time and start_time < 0:
            start_time = start_time + self.duration
        if end_time and end_time < 0:
            end_time = end_time + self.duration

        # trim till end if only start is specified
        if start_time is not None and end_index is None and end_time is None:
            end_time = self.duration
        if start_index is not None and end_index is None and end_time is None:
            end_index = len(self) - 1

        # trim from 0 if only end is specified
        if end_time is not None and start_time is None and start_index is None:
            start_time = 0
        if end_index is not None and start_time is None and start_index is None:
            start_index = 0

        # validate arguments
        start_time, start_index = _normalize_args_index_and_timestamp(
            start_time, start_index, self.duration or float("inf"), len(self) - 1
        )
        end_time, end_index = _normalize_args_index_and_timestamp(
            end_time, end_index, self.duration or float("inf"), len(self) - 1
        )
        if start_index > end_index:
            raise ValueError(
                f"invalid cut points. {start_index = } is not smaller than {end_index = }."
            )

        # get nodes to be trimmed
        start_node, relative_start_index = self.__get_node(start_index)
        end_node, relative_end_index = self.__get_node(end_index)
        if not end_node or not start_node:
            raise IndexError(
                f"Error trimming frames at {start_index = }, {end_index = }."
            )

        # trim
        remaining_nodes = end_node.__next
        end_node_source_end_index = end_node._source_end_index
        end_node.__next = None
        end_node._source_end_index = relative_end_index
        new = copy(start_node)
        end_node.__next = remaining_nodes
        end_node._source_end_index = end_node_source_end_index
        new._source_start_index = relative_start_index

        return new

    def iter_frames(
        self, start: int = 0, end: Optional[int] = None, step: Optional[int] = None
    ) -> Generator[NDArray[np.uint8], None, None]:
        """Iterate over the frames in the video from `start` index to `end` index with a certain `step` size.

        Args:
            start (int, optional): The index of the start frame. Defaults to 0.
            end (Optional[int], optional): The index of the end frame. `None` will iterate till the end of the video. Defaults to None.
            step (Optional[int], optional): The step size for iteration. If `None`, uses the default step size of 1. Defaults to None.

        Yields:
            NDArray[np.uint8]: 3D array representing frames from the video with shape: (height, width, color_channels).
        """
        for i in range(start, end or len(self), (step or 1) * self._default_step_size):
            yield self.get_frame(index=i)

    def __get_node(self, index: int) -> Tuple[Optional[Video], int]:
        """
        Find the node in the video linked list that contains the specified index/frame.

        Args:
            index (int): The index to search for within the video linked list.

        Returns:
            Video | None: The node containing the specified index or None if out of range.
        """

        # (0,(5,10),20) + (0,(6,9),10)
        # ....0,5.............6,9.....

        if index < 0:
            return None, -1
        elif index + self._source_start_index <= self._source_end_index:
            return self, index + self._source_start_index
        elif self.__next:
            return self.__next.__get_node(
                index - (self._source_end_index - self._source_start_index + 1)
            )
        else:
            return None, -1

    def __iter__(self):
        self.__idx = 0 if self._default_step_size > 0 else len(self) - 1
        return self

    def __next__(self) -> NDArray[np.uint8]:
        if 0 <= self.__idx < len(self):
            frame = self.get_frame(index=self.__idx)
            self.__idx += self._default_step_size
            return frame

        raise StopIteration

    def __getitem__(
        self, key: Union[int, slice, Sequence[Union[int, slice]]]
    ) -> Union[Video, NDArray[np.uint8]]:
        slices = _validate_and_normalize_slices(key, max_n_dims=4)
        if len(slices) > 4:
            raise ValueError(f"Expected at most 4 slices. Got {len(slices)}: {key}")
        if isinstance(key, int) or (
            isinstance(key, Sequence) and isinstance(key[0], int)
        ):
            return self.get_frame(index=slices[0].start)[slices[1:]]

        time_slice, frame_slices = slices[0], slices[1:]
        new = self.trim(
            start_index=time_slice.start or 0,
            end_index=(
                time_slice.stop - 1 if time_slice.stop is not None else len(self) - 1
            ),
        )

        # TODO: add support for step / reverse video (negative step)
        # every_node._default_step_size *= time_slice.step
        # every_node._source_end_index -= (node._source_end_index - node._source_end_index) % node._default_step_size

        if [fs for fs in frame_slices if fs not in (slice(None), slice(0, None))]:
            new.transform(lambda frame: frame[frame_slices])

        return new

    # ========================= #
    #    Typecast / get data    #
    # ========================= #

    def numpy(self, *args, **kwargs):
        """
        Convert the video frames to a (4D) NumPy array.

        Args:
            *args: Positional arguments to be passed to `np.array()`.
            **kwargs: Keyword arguments to be passed to `np.array()`.

        Returns:
            np.ndarray: A NumPy array representing the video frames.
        """

        return np.array(list(self), *args, **kwargs)

    def __array__(self, *args, **kwargs):
        return self.numpy(*args, **kwargs)

    def torch(self, *args, **kwargs):
        """
        Convert the video frames to a (4D) PyTorch tensor.

        Args:
            *args: Positional arguments to be passed to `torch.tensor()`.
            **kwargs: Keyword arguments to be passed to `torch.tensor()`.

        Returns:
            torch.Tensor: A PyTorch tensor representing the video frames.
        """

        return torch.tensor(self.numpy(), *args, **kwargs)

    # ========================== #
    #    Display / show frame    #
    # ========================== #

    def show(
        self,
        inline_player: Literal["jshtml", "html5"] = "html5",
        codec="avc1",
        **kwargs,
    ) -> None:
        """Display the video. If the function is called from a Jupyter Notebook, the output
        is displayed inline using the specified player. If the video has a single frame,
        it is displayed as an image plot. If the function is called from command line, a
        matplotlib animation window shows the video.

        Note:
            - Clear previous video output in jupyter before displaying the video again to avoid issues.

        Args:
            inline_player (str, optional): The type of matplotlib inline player to use. Defaults to "html5".
            codec (str, optional): The codec to use for the video (should be installed on the system). Possible values are ["avc1", "h264", "mp4v", ...]. Defaults to "avc1".
        """
        # if unmodified
        if (
            in_jupyter_notebook()
            and self._path is not None
            and self.__next is None
            and self._source_start_index == 0
            and self._source_end_index == len(self.source) - 1
            and inline_player == "html5"
            and not self.transformations
        ):
            VideoDisplay.display_ipython_video_in_jupyter(self._path)

        # rewrite
        elif in_jupyter_notebook() and inline_player == "html5" and len(self) > 1:
            if isfile(temp_filename := abspath(join(".", "__temp__.mp4"))):
                remove(temp_filename)
            self.save(temp_filename, overwrite=True, codec=codec)
            VideoDisplay.display_ipython_video_in_jupyter(temp_filename)

        else:
            VideoDisplay.display_frames(
                self, fps=self.fps or 30.0, inline_player=inline_player
            )

    def show_frame(
        self, timestamp: Optional[float] = None, index: Optional[int] = None
    ) -> None:
        """Display a specific frame from the video.

        Args:
            timestamp (Optional[float], optional): The timestamp of the frame to display. If not provided, `index` will be used. Defaults to None.
            index (Optional[int], optional): The index of the frame to display. If not provided, `timestamp` will be used. Defaults to None.
        """
        frame = self.get_frame(timestamp=timestamp, index=index)

        if frame is not None:
            VideoDisplay.display_frames([frame], inline_player="jshtml")

    def frames_grid(
        self,
        rows=2,
        columns=3,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> NDArray[np.uint8]:
        """Create a grid of frames from the video as a single stacked image.
        Equally spaced timestamps are chosen across the video and arranged into a rows x columns grid.

        First frame is placed in the top-right cell. The immediately next frame is placed in the same row and in the column on the right
        and so on until columns in that row run out and then the row below is chosen

        Args:
            rows (int, optional): The number of rows in the grid. Defaults to 2.
            columns (int, optional): The number of columns in the grid. Defaults to 3.
            width (Optional[int], optional): The width of the grid. If only `height` is given, the resized width is calculated by maintaining the aspect ratio of the grid cell. If both are None, the grid is not resized. Defaults to None.
            height (Optional[int], optional): The height of the grid. If only `width` is given, the resized height is calculated by maintaining the aspect ratio of the grid cell. If both are None, the grid is not resized. Defaults to None.

        Returns:
            NDArray[np.uint8]: an RGB 3D numpy array containing the stacked frames. shape: (height, width, color_channels).
        """
        grid = np.concatenate(
            [
                np.concatenate([self.get_frame(index=id) for id in ids_in_row], axis=1)
                for ids_in_row in np.linspace(0, len(self) - 1, rows * columns)
                .reshape(rows, columns)
                .round()
                .astype(int)
            ],
            axis=0,
        )

        if width or height:
            # calculate height or width of grid by maintaining aspect ratio of cell
            aspect_ratio = (self.width * columns) / (self.height * rows)

            if width and not height:
                height = int(width / aspect_ratio)
            elif height and not width:
                width = int(height * aspect_ratio)

            grid = cv2.resize(grid, (width, height))  # type: ignore

        return grid

    def show_frames_grid(
        self,
        rows=2,
        columns=3,
        width: Optional[int] = 800,
        height: Optional[int] = None,
    ):
        """Display a grid of frames from the video as a single stacked image.
        Top left cell contains the first frame. sequence grows towards the right and then down.

        Args:
            rows (int, optional): The number of rows in the grid. Defaults to 2.
            columns (int, optional): The number of columns in the grid. Defaults to 3.
            width (Optional[int], optional): The width of the grid. If only `height` is given, the resized width is calculated by maintaining the aspect ratio of the grid cell. If both are None, the grid is not resized. Defaults to 800.
            height (Optional[int], optional): The height of the grid. If only `width` is given, the resized height is calculated by maintaining the aspect ratio of the grid cell. If both are None, the grid is not resized. Defaults to None.
        """
        grid = self.frames_grid(rows=rows, columns=columns, width=width, height=height)

        VideoDisplay.display_frames([grid], inline_player="jshtml")

    # ================= #
    #    Concatenate    #
    # ================= #

    @staticmethod
    def concatenate(objects: Iterable[Video]) -> Video:
        """
        Concatenate a sequence of Video objects in time dimension into a single Video.

        Args:
            objects (Iterable[Video]): A sequence of Video objects to concatenate.

        Returns:
            Video: A new Video object that is a linked list of all the input videos.

        Raises:
            ValueError: If the input sequence of videos is empty.
        """

        video_chain = None
        for video in objects:
            if video_chain is None:
                video_chain = deepcopy(video)
            else:
                video_chain.append(video)

        if video_chain is None:
            raise ValueError("Cannot concatenate an empty sequence of videos.")

        return video_chain

    @classmethod
    def stack(
        cls,
        videos: Sequence[Video],
        dim=1,
        fps: Union[
            float, Literal["first", "max", "average", "weighted", "min"]
        ] = "max",
    ):
        """
        Stack a list of Video objects along a specified dimension (0=time, 1=height, 2=width, 3=channels).
        All videos will be aligned at the start and padded with black frames at the end if necessary.

        Args:
            videos (Sequence[Video]): A list of Video objects to stack.
            dim (int, optional): The dimension along which to stack the videos (0=time, 1=height, 2=width, 3=channels). Negative values represent dimensions relative to the end (e.g., -1 is the channel dimension). Defaults to 1.
            fps (Union[float, Literal["first", "max", "average", "weighted"]], optional): The frames per second of the output video. If "first", it will use the fps of the first video. If "max", it will use the maximum fps of all videos. If "average", it will use the average fps of all videos. If "weighted", it will use the weighted average fps of all videos (weights=duration). If "min", it will use the minimum fps of all videos.  Not used for time dimension. Defaults to "max".

        Returns:
            Video: A new Video object resulting from stacking the input videos.
        """

        # --- validate arguments --- #

        if dim < -4 or dim > 3:
            raise ValueError(f"Dimension should be in range -4<=dim<=3. Got {dim = }")
        if dim < 0:
            dim = dim + 4
        if dim == 0:
            return cls.concatenate(videos)

        if fps == "first":
            fps = videos[0].fps
        elif fps == "max":
            fps = max(v.fps for v in videos)
        elif fps == "min":
            fps = min(v.fps for v in videos)
        elif fps == "average":
            fps = sum(v.fps for v in videos) / len(videos)
        elif fps == "weighted":
            fps = sum(len(v) for v in videos) / sum(v.duration for v in videos)
        if not isinstance(fps, (float, int)):
            raise ValueError(f"Invalid value for {fps = }")

        # --- stack videos --- #

        def stacked_frames(videos: Sequence[Video], fps: float):
            duration = max(v.duration for v in videos)

            for t in np.arange(0.5 / fps, duration, 1 / fps):
                frames = []
                for video in videos:
                    if t <= video.duration:
                        frames.append(video.get_frame(timestamp=t))
                    else:
                        frames.append(np.zeros(video.shape[1:], dtype=np.uint8))

                # todo: resize frames to the same height/width while maintaining aspect ratio
                yield np.concatenate(frames, axis=dim - 1)

        return cls(
            stacked_frames(videos, fps),
            fps=fps,
            total_frames=int(max(v.duration for v in videos) * fps + 0.5),
        )

    def append(self, other: Video):
        """
        Append another Video object to the end of the current Video.

        Args:
            other (Video): The Video object to append.

        Returns:
            None
        """

        # prevent cycle in the video linked list
        other = copy(other)

        if self.__next is None:
            self.__next = other
        else:
            self.__next.append(other)

    def __add__(self, other: Video):
        new = copy(self)
        new.append(copy(other))
        return new

    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update({k: copy(v) for k, v in self.__dict__.items()})
        new._source_end_index = self._source_end_index

        return new

    # =============== #
    #    Transform    #
    # =============== #

    def transform(
        self, transformation: Callable[[NDArray[np.uint8]], NDArray[np.uint8]]
    ):
        """
        Apply a transformation function to the individual frames of this object (in place).

        Args:
            transformation (Callable[[NDArray[np.uint8]], NDArray[np.uint8]]): The transformation function to be applied that must map 1 numpy array to another numpy array.

        Raises:
            ValueError: If the transformation is not callable.

        Returns:
            None
        """

        if not isinstance(transformation, (Callable,)):
            raise ValueError(
                f"Cannot apply this transformation because it is not callable. {transformation = }"
            )

        self.transformations.append(transformation)
        if self.__next:
            self.__next.transform(transformation)

        self.__update_shape_components()

    # ================ #
    #    Dimensions    #
    # ================ #

    def __len__(self) -> int:
        # TODO: handle steps e.g. video[::3, ...]
        # todo: fix len when transformation is applied
        return (self._source_end_index - self._source_start_index + 1) + len(
            self.__next or ()
        )

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Tuple of array dimensions (n_frames, height, width, n_channels)."""
        return len(self), self.height, self.width, self.n_channels

    @property
    def height(self) -> int:
        """number of vertical pixels in a video frame (dimension=1)"""
        if self._height is None:
            raise ValueError("self._height is not defined.")
        return self._height

    @property
    def width(self) -> int:
        """number of horizontal pixels in a video frame (dimension=2)"""
        if self._width is None:
            raise ValueError("self._width is not defined.")
        return self._width

    @property
    def n_channels(self) -> int:
        """number of color channels in a video frame (e.g. RGB) (dimension=3)"""
        if self._n_channels is None:
            raise ValueError("self._n_channels is not defined.")
        return self._n_channels

    @property
    def duration(self) -> float:
        """total time that the frames would take to play in a sequence. depends on fps."""
        if self.fps is None:
            # TODO: remove this
            raise RuntimeError("FPS is not set. Can not get duration without fps.")

        return len(self) / self.fps if self.fps else float("inf")

    @property
    def n_frames(self) -> int:
        """The number of frames in the video."""
        return len(self)

    # ================== #
    #    Save to disk    #
    # ================== #

    @staticmethod
    def save_(  # TODO: overload save to be static and instance method simultaneously
        frames_iterable: Iterable[NDArray[np.uint8]],
        path: str,
        overwrite=False,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: float = 30.0,
        codec: str = "mp4v",
        progress_bar=True,
        leave=False,
        total_frames: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Class method to save a sequence of frames into a video file.
        A frame is a 3D numpy uint8 array of shape (height, width, channels)

        Args:
            path (str): The path to the output file.
            overwrite (bool, optional): Whether to overwrite the output file if it already exists. Defaults to False.
            height (Optional[int], optional): The number of vertical pixels in the output video file. If None, the shape (index=0) of the first frame is used. Defaults to None.
            width (Optional[int], optional): The number of horizontal pixels in the output video file. If None, the shape (index=1) of the first frame is used. Defaults to None.
            fps (float, optional): The frames per second of the output video. Defaults to 30.0.
            codec (str, optional): The codec used for the output video e.g ["h264", "mp4v", "xvid", "avc1", "hvc1"]. Make sure the codec is already installed in your system (some do not ship with OpenCV because of license mismatch). Defaults to "mp4v".
            progress_bar (bool, optional): Whether to display a progress bar while saving the video. Defaults to True.
            leave (bool, optional): Whether to leave the progress bar after the operation is complete. Defaults to False.
            total_frames (Optional[int], optional): total number of frames in the frames iterable. Used by the progress bar. Defaults to None.

        Raises:
            FileExistsError: If a file already exists at the output path and `overwrite` is False.
        """
        path = validate_path_exists(path, overwrite=overwrite)

        _frames_iterable = iter(
            frames_iterable
            if not progress_bar
            else tqdm(
                frames_iterable,
                total=getattr(frames_iterable, "__len__", None) or total_frames,
                leave=leave,
            )
        )
        first_frame = next(_frames_iterable)
        height = height or first_frame.shape[0]
        width = width or first_frame.shape[1]

        # TODO: Compile opencv with GPU support and write to disk form CUDA.
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*codec), fps, (width, height)  # type: ignore
        )

        try:
            writer.write(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            for frame in _frames_iterable:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.write(frame)
        finally:
            writer.release()

    def save(
        self,
        path: str,
        overwrite=False,
        fps: Optional[float] = None,
        codec: Optional[str] = None,
        progress_bar=True,
        leave=False,
        **kwargs,
    ) -> None:
        """Save the video frames to a file.

        Args:
            path (str): The path to the output file.
            fps (float | None, optional): The frames per second of the output video. If None, it will use the fps of the video source or 30 if that is not set either. Defaults to None.
            codec (str | None, optional): The codec used for the output video e.g ["h264", "mp4v", "xvid", "avc1", "hvc1"]. Make sure the codec is already installed in your system (some do not ship with OpenCV because of license mismatch). If None, it will use the source video codec or "mp4v" if that is not set. Defaults to None.
            overwrite (bool, optional): Whether to overwrite the output file if it already exists. Defaults to False.
            progress_bar (bool, optional): Whether to display a progress bar while saving the video. Defaults to True.
            leave (bool, optional): Whether to leave the progress bar after the operation is complete. Defaults to False.
            **kwargs: Additional keyword arguments. (not used yet.)
        """

        self.save_(
            self.iter_frames(),
            path,
            overwrite=overwrite,
            width=self.width,
            height=self.height,
            fps=fps or self.fps or 30,
            codec=codec or self.codec or "mp4v",
            progress_bar=progress_bar,
            leave=leave,
            total_frames=len(self),
            **kwargs,
        )

    def save_frame(
        self,
        path: str,
        timestamp: Optional[float] = None,
        index: Optional[int] = None,
        overwrite=False,
    ) -> None:
        """
        Saves a single frame from the video object to the specified path.

        Args:
            path (str): The path where the frame will be saved.
            timestamp (float | None, optional): The timestamp of the frame to be saved. If None, `index` will be used. Defaults to None.
            index (int | None, optional): The index of the frame to be saved. If None, `timestamp` will be used. Defaults to None.
            overwrite (bool, optional): Whether to overwrite the image file if it already exists. Defaults to False.

        Raises:
            ValueError: If both or neither `timestamp` and `index` are provided or If the specified timestamp or index is out of range.
            FileExistsError: If a file already exists at the output path and `overwrite` is False.
        """
        path = validate_path_exists(path, overwrite=overwrite)

        frame = self.get_frame(timestamp=timestamp, index=index)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, frame)

    def save_frames_grid(
        self,
        path: str,
        rows: int = 2,
        columns: int = 3,
        width: Optional[int] = 1024,
        height: Optional[int] = None,
        overwrite=False,
    ) -> None:
        """Write a grid of frames from the video to a single image file.
        The grid is created by arranging frames from the video taken at
        equally spaced timestamps in a rows x columns grid.

        Args:
            path (str): The path to the output image file.
            rows (int, optional): The number of rows in the grid. Defaults to 2.
            columns (int, optional): The number of columns in the grid. Defaults to 3.
            width (Optional[int], optional): The width of the grid. If only `height` is given, the resized width is calculated by maintaining the aspect ratio of the grid cell. If both are None, the grid is not resized. Defaults to 1024.
            height (Optional[int], optional): The height of the grid. If only `width` is given, the resized height is calculated by maintaining the aspect ratio of the grid cell. If both are None, the grid is not resized. Defaults to None.
            overwrite (bool, optional): Whether to overwrite the image file if it already exists. Defaults to False.

        Raises:
            FileExistsError: If a file already exists at the output path and `overwrite` is False.
        """
        path = validate_path_exists(path, overwrite=overwrite)

        grid = self.frames_grid(rows=rows, columns=columns, width=width, height=height)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, grid)

    # ============================= #
    #    Initialize Video object    #
    # ============================= #

    @classmethod
    def load(cls, path: str, **kwargs) -> Video:
        """
        Load a video from the specified path.

        Args:
            path (str): The path to the video file.
            **kwargs: Additional keyword arguments to be passed to the Video constructor.

        Returns:
            Video: The loaded video object.
        """
        return cls(path, **kwargs)

    @classmethod
    def load_asset(
        cls,
        label: str,
        archive_name: Optional[str] = None,
        overwrite=False,
        progress_bar=True,
        leave=True,
        **kwargs,
    ) -> Video:
        """Class method to load a video asset identified by the given label from
        the built-in datasets and return it as a new Video object.

        This method downloads dictionary videos from direct URLs if the corresponding
        archive is not already downloaded. Otherwise the video asset will be extracted
        from an archive which will be auto-downloaded once.

        Notes:
        - To view valid asset IDs, run `slt.Assets.get_ids(r"^videos/.*mp4$")` for dictionary videos
          or `slt.Assets.get_ids(r"^datasets/.*videos.*zip$")` for archived videos.
        - See `slt.Assets.ROOT_DIR` for download directory

        Args:
            label (str): The filename of the video asset to load. 'videos/' is prepended to the label if it does not start with it and '.mp4' is appended to the label if it does not end with it. An example is `videos/pk-hfad-1_airplane.mp4` for a dictionary video. General syntax is `videos/country-organization-number_text[_person_camera].mp4`.
            archive_name (str | None, optional): The name of the archive which contains the video asset. If None, the archive name is inferred from the label. Not necessary for dictionary videos. An example is `datasets/pk-hfad-1.videos-mp4.zip`. General syntax is `datasets/country-organization-number[_person_camera].videos-mp4.zip`. Defaults to None.
            overwrite (bool, optional): Whether to overwrite the video asset if it is already downloaded or extracted. Defaults to False.
            progress_bar (bool, optional): Whether to display a progress bar while downloading or extracting the asset. Defaults to True.
            leave (bool, optional): Whether to leave the progress bar after the operation is complete. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the Video constructor.

        Raises:
            FileNotFoundError: If no video assets are found for the given label.

        Warns:
            UserWarning: If multiple video assets match the given label and the only first asset is used.

        Returns:
            Video: An instance of the Video class representing the video that matched the label.

        Examples:

            .. code-block:: python

                import sign_language_translator as slt

                # Load a dictionary video asset
                video = slt.Video.load_asset("pk-hfad-1_airplane")

                # Load a replication video from the built-in datasets
                video = slt.Video.load_asset("videos/pk-hfad-1_airplane_dm0001_front.mp4", archive_name="datasets/pk-hfad-1_dm0001_front.videos-mp4.zip")
        """

        if "/" not in label:
            label = f"videos/{label}"

        if "." not in label:
            label = f"{label}.mp4"

        paths = []
        if Assets.is_dictionary_video(label):
            # if no corresponding archive is already downloaded
            if not any(
                isfile(archive_path)
                for archive_path in Assets.get_path(
                    archive_name or Assets.infer_archive_name(label)
                )
            ):
                # download video from direct URL
                paths = Assets.download(
                    label, progress_bar=progress_bar, leave=leave, overwrite=overwrite
                )

        # extract video from archive
        if not paths:
            paths = Assets.extract(
                label,
                archive_name_or_regex=archive_name,
                download_archive=True,
                overwrite=overwrite,
                progress_bar=progress_bar,
                leave=leave,
            )

        if len(paths) == 0:
            raise FileNotFoundError(f"No video assets found for '{label}'")
        if len(paths) > 1:
            warn(f"Multiple video assets matched '{label}'. Using the first of:{paths}")

        return cls.load(paths[0], **kwargs)

    def __initialize_from_arguments(self, sign, **kwargs):
        if isinstance(sign, str):
            if not isfile(sign):
                raise ValueError(f"Invalid path argument: '{sign}'")
            self._from_path(sign, **kwargs)

        elif isinstance(sign, (np.ndarray, torch.Tensor, Sequence)):
            self._from_frames(sign, **kwargs)  # type: ignore
        elif isinstance(sign, Iterable):
            self._from_iterable(
                sign,
                kwargs["total_frames"],
                **{k: v for k, v in kwargs.items() if k not in ("total_frames",)},
            )

        else:
            raise ValueError(
                f"Invalid argument: {sign}. provide either a path to a video or a sequence of frames."
            )

    def _from_path(self, path: str, **kwargs) -> None:
        self._path = abspath(path)
        file_type = str(guess_type(path)[0])

        if file_type.startswith("image"):
            self._from_frames(
                [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)],  # type: ignore
                fps=kwargs.get("fps", 1),
                **kwargs,
            )

        elif file_type.startswith("video"):
            self.source = VideoCaptureFrames(self._path, **kwargs)
            self.__extract_properties_from_source(self.source)

        else:
            raise ValueError(f"unknown file type: {file_type}")

    def __extract_properties_from_source(self, source: VideoFrames) -> None:
        self.fps = getattr(source, "fps", 30.0)
        self.fourcc = getattr(source, "fourcc", cv2.VideoWriter_fourcc(*"mp4v"))  # type: ignore

    @property
    def codec(self) -> str:
        """The video codec used to encode the video. (e.g. "mp4v", "h264", "xvid", "avc1", "hvc1")"""
        return struct.pack("<I", self.fourcc).decode("utf-8")

    def __update_shape_components(self) -> None:
        shape = self.get_frame(0).shape
        self._height = shape[0]
        self._width = shape[1]
        self._n_channels = shape[2]

    def _from_data(self, sign_data: Sequence[NDArray[np.uint8]], **kwargs) -> None:
        if isinstance(sign_data, Sequence):
            self._from_frames(sign_data, **kwargs)
        else:
            self._from_iterable(sign_data, kwargs["total_frames"], **kwargs)

    def _from_frames(
        self, frames: Sequence[NDArray[np.uint8]], fps: float = 30.0, **kwargs
    ):
        # validation
        if len(frames) <= 0:
            raise ValueError("Can not initialize from None or empty frames argument.")
        if frames[0].ndim != 3:
            raise ValueError("Each frame must be 3D (Height, Width, Channels).")
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy().astype(np.uint8)
        elif isinstance(frames[0], torch.Tensor):
            frames = torch.Tensor(frames).numpy().astype(np.uint8)

        # source
        self.source = SequenceFrames(frames, fps=fps)
        self.__extract_properties_from_source(self.source)

    def _from_iterable(self, frames_iterable: Iterable, total_frames: int, **kwargs):
        """load the frames from a read-once iterable such as a generator."""
        self.source = IterableFrames(frames_iterable, total_frames, **kwargs)
        self.__extract_properties_from_source(self.source)

    @property
    def source(self) -> VideoFrames:
        """A VideoFrames object which wraps around a frame sequence."""
        if self._source is None:
            raise ValueError("Video source is not set.")
        return self._source

    @source.setter
    def source(self, source: VideoFrames):
        if not isinstance(source, VideoFrames):
            raise ValueError(
                f"Invalid source type: {type(source)}. Should be slt.vision.video.VideoFrames"
            )
        self._source = source

    # ============================= #
    #    Cleaning / with _ as _:    #
    # ============================= #

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        """Release the resources occupied by the object and reset the properties."""
        # source
        if self._source is not None:
            self.source.close()
        if self.__next is not None:
            self.__next.close()
            self.__next = None

        # play
        self._source_start_index = 0
        self._source_end_index = 0
        self.transformations = []

        # properties
        self._path = None

        self.is_closed = True
