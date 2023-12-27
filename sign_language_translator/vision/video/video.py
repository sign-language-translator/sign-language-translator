from __future__ import annotations

__all__ = [
    "Video",
]

import struct
from copy import copy, deepcopy
from mimetypes import guess_type
from os import makedirs
from os.path import abspath, dirname, isfile, join
from typing import Callable, Generator, Iterable, List, Sequence, Tuple

import cv2  # TODO: Compile OpenCV with GPU
import numpy as np
import torch
from numpy.typing import NDArray
from tqdm.auto import tqdm

from sign_language_translator.config.assets import Assets
from sign_language_translator.config.enums import SignFormats
from sign_language_translator.utils import in_jupyter_notebook
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
    def __init__(
        self,
        sign: str
        | Sequence[NDArray[np.uint8] | torch.Tensor]
        | NDArray[np.uint8]
        | torch.Tensor
        | Generator[NDArray[np.uint8], None, None],
        **kwargs,
    ) -> None:
        self._path: str | None = None

        # ToDo: use list of sources instead of linked list of videos.
        # source
        self.source: VideoFrames
        self.__next: Video | None = None

        # play
        self._source_start_index: int = 0
        self._source_end_index: int
        # TODO: handle len & end_index for ||step|| > 1
        self._default_step_size: int = 1
        self.transformations: List[Callable] = []

        # properties
        self.fps: float = 30.0
        self.fourcc: int = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore
        self._height: int
        self._width: int
        self._n_channels: int

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
        self, timestamp: float | None = None, index: int | None = None
    ) -> NDArray[np.uint8]:
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
        start_time: float | None = None,
        end_time: float | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
    ) -> Video:
        # allow negative indexing
        if start_index and start_index < 0:
            start_index = start_index + len(self)
        if end_index and end_index < 0:
            end_index = end_index + len(self)
        if start_time and start_time < 0:
            start_time = start_time + self.duration
        if end_time and end_time < 0:
            end_time = end_time + self.duration

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
        self, start: int = 0, end: int | None = None, step: int | None = None
    ):
        for i in range(start, end or len(self), (step or 1) * self._default_step_size):
            yield self.get_frame(index=i)

    def __get_node(self, index: int) -> Tuple[Video | None, int]:
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
        self, key: int | slice | Sequence[int | slice]
    ) -> Video | NDArray[np.uint8]:
        slices = _validate_and_normalize_slices(key, max_n_dims=4)
        if len(slices) > 4:
            raise ValueError(f"Expected at most 4 slices. Got {len(slices)}: {key}")
        if isinstance(key, int) or (
            isinstance(key, Sequence) and isinstance(key[0], int)
        ):
            return self.get_frame(index=slices[0].start)[slices[1:]]

        time_slice, frame_slices = slices[0], slices[1:]
        new = self.trim(
            start_index=time_slice.start or self._source_start_index,
            end_index=(
                time_slice.stop - 1
                if time_slice.stop is not None
                else self._source_end_index
            ),
        )

        # TODO: add support for step
        # every_node._default_step_size *= time_slice.step
        # every_node._source_end_index -= (node._source_end_index - node._source_end_index) % node._default_step_size

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

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.Tensor or func is torch.tensor:
            dtype = kwargs.pop("dtype", torch.uint8)  # type: ignore
            return torch.tensor(self.numpy(), dtype=dtype, *args, **kwargs)  # type: ignore
        return NotImplemented

    # ========================== #
    #    Display / show frame    #
    # ========================== #

    def show(self, inline_player="html5" or "jshtml", **kwargs) -> None:
        if (
            in_jupyter_notebook()
            and self._path is not None
            and self.__next is None
            and inline_player == "html5"
            and not self.transformations
        ):
            VideoDisplay.display_ipython_video_in_jupyter(self._path)
        elif in_jupyter_notebook() and inline_player == "html5" and len(self) > 1:
            temp_filename = abspath(join(".", "__temp__.mp4"))
            self.save(temp_filename, overwrite=True, codec="avc1")
            VideoDisplay.display_ipython_video_in_jupyter(temp_filename)

        else:
            VideoDisplay.display_frames(
                self, fps=self.fps or 30.0, inline_player=inline_player
            )

    def show_frame(
        self, timestamp: float | None = None, index: int | None = None
    ) -> None:
        frame = self.get_frame(timestamp=timestamp, index=index)

        if frame is not None:
            VideoDisplay.display_frames([frame], inline_player="jshtml")

    def frames_grid(
        self, rows=2, columns=3, width: int | None = None, height: int | None = None
    ):
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
        self, rows=2, columns=3, width: int | None = 800, height: int | None = None
    ):
        grid = self.frames_grid(rows=rows, columns=columns, width=width, height=height)

        VideoDisplay.display_frames([grid], inline_player="jshtml")

    # ================= #
    #    Concatenate    #
    # ================= #

    @staticmethod
    def concatenate(objects: Iterable[Video]) -> Video:
        """
        Concatenate a sequence of Video objects into a single Video.

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

    @staticmethod
    def stack(videos: List[Video], dim=1):
        """
        Stack a list of Video objects along a specified dimension (0=time, 1=height, 2=width, 3=channels).

        Args:
            videos (List[Video]): A list of Video objects to stack.
            dim (int, optional): The dimension along which to stack the videos. Defaults to 1.

        Returns:
            Video: A new Video object resulting from stacking the input videos.
        """

        if dim == 0:
            return Video.concatenate(videos)

        videos = list(videos)

        def stacked_frames(videos: List[Video]):
            for i in range(max(len(v) for v in videos)):
                yield np.concatenate(
                    [v.get_frame(index=i) for v in videos], axis=dim - 1
                )

        return Video(stacked_frames(videos), fps=videos[0].fps)

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
        if self.fps is None:
            # TODO: remove this
            raise RuntimeError("FPS is not set. Can not get duration without fps.")

        return len(self) / self.fps if self.fps else float("inf")

    @property
    def length(self) -> int:
        return len(self)

    # ================== #
    #    Save to disk    #
    # ================== #

    @staticmethod
    def save_(  # TODO: overload save to be static and instance method simultaneously
        frames_iterable: Iterable[NDArray[np.uint8]],
        path: str,
        overwrite=False,
        height: int | None = None,
        width: int | None = None,
        fps: float = 30.0,
        codec: str = "XVID",
        progress_bar=True,
        leave=False,
        total_frames: int | None = None,
        **kwargs,
    ) -> None:
        path = abspath(path)
        if isfile(path) and not overwrite:
            raise FileExistsError(
                f"File '{path}' already exists. Use overwrite=True to overwrite."
            )
        makedirs(dirname(path), exist_ok=True)

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
        fps: float | None = None,
        codec: str | None = None,
        progress_bar=True,
        leave=False,
        **kwargs,
    ) -> None:
        """Save the video frames to a file.

        Args:
            path (str): The path to the output file.
            fps (float | None, optional): The frames per second of the output video. If None, it will use the fps of the video source or 30 if that is not set either. Defaults to None.
            codec (str | None, optional): The codec used for the output video e.g ["xvid", "h264", "hvc1"]. If None, it will use the source video codec or "XVID" if that is not set. Defaults to None.
            **kwargs: Additional keyword arguments. (not used yet.)
        """

        Video.save_(
            self.iter_frames(),
            path,
            overwrite=overwrite,
            width=self.width,
            height=self.height,
            fps=fps or self.fps or 30,
            codec=codec or self.codec or "XVID",
            progress_bar=progress_bar,
            leave=leave,
            total_frames=len(self),
            **kwargs,
        )

    def save_frame(
        self,
        path: str,
        timestamp: float | None = None,
        index: int | None = None,
        overwrite=False,
    ) -> None:
        """
        Saves a single frame from the video object to the specified path.

        Args:
            path (str): The path where the frame will be saved.
            timestamp (float | None, optional): The timestamp of the frame to be saved. If None, `index` will be used. Defaults to None.
            index (int | None, optional): The index of the frame to be saved. If None, `timestamp` will be used. Defaults to None.
        Raises:
            ValueError: If both or neither `timestamp` and `index` are provided or If the specified timestamp or index is out of range.
            RuntimeError: If there is an error reading the frame.
        """

        path = abspath(path)
        if isfile(path) and not overwrite:
            raise FileExistsError(
                f"File '{path}' already exists. Use overwrite=True to overwrite."
            )
        makedirs(dirname(path), exist_ok=True)

        frame = self.get_frame(timestamp=timestamp, index=index)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, frame)

    def save_frames_grid(
        self,
        path: str,
        rows: int = 2,
        columns: int = 3,
        width: int | None = 1024,
        height: int | None = None,
        overwrite=False,
    ) -> None:
        path = abspath(path)
        if isfile(path) and not overwrite:
            raise FileExistsError(
                f"File '{path}' already exists. Use overwrite=True to overwrite."
            )
        makedirs(dirname(path), exist_ok=True)

        grid = self.frames_grid(rows=rows, columns=columns, width=width, height=height)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, grid)

    # ============================= #
    #    Initialize Video object    #
    # ============================= #

    @staticmethod
    def load(path: str, **kwargs) -> Video:
        """
        Load a video from the specified path.

        Args:
            path (str): The path to the video file.
            **kwargs: Additional keyword arguments to be passed to the Video constructor.

        Returns:
            Video: The loaded video object.
        """
        return Video(path, **kwargs)

    def __initialize_from_arguments(self, sign, **kwargs):
        if isinstance(sign, str):
            if isfile(sign) and not kwargs.get("is_asset", False):
                self._from_path(sign, **kwargs)
            elif Assets.get_ids(sign):
                Assets.download(sign, leave=False, overwrite=False)
                self._from_path(Assets.get_path(sign)[0], **kwargs)
            else:
                raise ValueError(
                    f"Invalid argument: {sign}. provide path to a video or relative path to a video resource from dataset."
                )

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
        self.fourcc = getattr(source, "fourcc", cv2.VideoWriter_fourcc(*"XVID"))  # type: ignore

    @property
    def codec(self) -> str:
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
