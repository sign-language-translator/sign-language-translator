"""
This module defines the `Landmarks` wrapper class which inherits from the `Sign` class.
It is used to represent and manipulate landmarks data extracted from a video featuring 1 person.
It provides methods for saving & loading landmarks data from various sources
(including CSV, NPY, PT, and PTH files, numpy arrays, PyTorch tensors, and nested lists).
It enables data augmentation using the `.transform()` method and smart concatenation across time dimension.
It can visualize the sequence of 3D landmarks using the `.show()` method.

Classes:
    Landmarks: A class to represent and manipulate landmarks data.

Example:

    .. code-block:: python

        from sign_language_translator.vision.landmarks.landmarks import Landmarks

        landmarks = Landmarks([[[0,1,2], [1,2,3]]])  # 1 frames, 2 landmarks, 3 coordinates
        landmarks.show()
        landmarks.save('landmarks_file.csv')
        # landmarks = Landmarks.load('landmarks_file.csv')

        landmarks = Landmarks.load_asset('landmarks/pk-hfad-1_car.landmarks-mediapipe-world.csv')
        print(landmarks.shape)
        # (60, 75, 5)
"""

from __future__ import annotations

__all__ = [
    "Landmarks",
]

import os
from os.path import basename
from string import ascii_letters, digits
from typing import Callable, Iterable, List, Literal, Optional, Sequence, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray
from torch import Tensor
from tqdm.auto import tqdm

from sign_language_translator.config.assets import Assets
from sign_language_translator.config.enums import SignFormats
from sign_language_translator.utils import in_jupyter_notebook, validate_path_exists
from sign_language_translator.utils.arrays import ArrayOps
from sign_language_translator.vision.landmarks.connections import (
    BaseConnections,
    get_connections,
)
from sign_language_translator.vision.landmarks.display import (
    MatPlot3D,
    _reset_counter_in_animation_title,
)
from sign_language_translator.vision.sign.sign import Sign


class Landmarks(Sign):
    """A class to represent and manipulate landmarks data. Inherits from the Sign class.

    Args:
        sign (NDArray | Tensor | str): It can be provided as a path to a file (csv, npy, pt, pth), a NumPy array, a PyTorch tensor, or a sequence of arrays or tensors or numbers (3D: n_frames, n_landmarks, n_features).

    Methods:
        load(path: str, **kwargs): Class method to load landmarks data from a file and return a new Landmarks object.
        save(path: str, overwrite=False, precision=4, **kwargs): Saves the landmarks data to a file.
        name(): Static method which returns the string code of the sign format.
        numpy(*args, **kwargs): Returns the landmarks data as a NumPy array.
        torch(dtype=None, device=None): Returns the landmarks data as a PyTorch tensor.
        tolist(): Returns the landmarks data as a nested list.
        concatenate(objects: Iterable[Landmarks]): Concatenates a sequence of Landmarks objects along the first dimension (time) and returns a new Landmarks object
        transform(transformation: Callable): Applies a transformation function to the landmarks data.
        show(**kwargs): Displays the landmarks data.
        __getitem__(indices): Returns a new Landmarks object with the specified indices.
        __iter__(): Initializes the iteration over the frames of the landmarks data.
        __next__(): Returns the next frame of the landmarks data.

    Attributes:
        data: The landmarks data as a NumPy array or PyTorch tensor depending upon what it was initialized with.
        n_frames: The number of frames or time-steps in the data.
        n_landmarks: The number of landmarks in each frame of the data.
        n_features: The number of features per landmark (same as n_coordinates).
        shape: The shape of the landmarks data array as a tuple of integers.
        ndim: The number of dimensions of the landmarks data array (should be 3).
    """

    def __init__(
        self,
        sign: Union[
            str,
            NDArray,
            Tensor,
            Sequence[NDArray],
            Sequence[Tensor],
            # Sequence[Union[float, int]],  # 1D
            # Sequence[Sequence[Union[float, int]]],  # 2D
            Sequence[Sequence[Sequence[Union[float, int]]]],  # 3D
        ],
        connections: Union[BaseConnections, str, None] = None,
        **kwargs,
    ) -> None:
        """Initializes the Landmarks object.

        Args:
            sign (Union[str, NDArray, Tensor, Sequence]): It can be a string (path to a .csv, .npy or .pt/.pth file), an NDArray, a Tensor, or a sequence of these types or numbers but should be 3D data (n_frames, n_landmarks, n_features) except for the `.csv` which should be 2D (n_frames rows and n_landmarks*n_features columns).
        """
        self._data = None
        self._path = None
        self._caption = None
        self.__idx = 0

        self._animation = None
        self._connections = None

        self.__initialize_from_arguments(sign, connections=connections, **kwargs)

    @staticmethod
    def name() -> str:
        return SignFormats.LANDMARKS.value

    # ========================= #
    #    Iterate / get frame    #
    # ========================= #

    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self) -> Union[Tensor, NDArray]:
        if 0 <= self.__idx < len(self):
            frame = self.data[self.__idx]
            self.__idx += 1
            return frame

        raise StopIteration

    def __getitem__(self, indices) -> Landmarks:
        indices = self.__normalize_slice_indices(indices)
        return Landmarks(self.data[indices], connections=self._connections)

    # ========================= #
    #    Typecast / get data    #
    # ========================= #

    def numpy(self, *args, **kwargs) -> NDArray:
        """Returns the landmarks data as a numpy array.
        Additional arguments are passed to the numpy.array constructor.

        Returns:
            NDArray: The sign data as a NumPy array.

        Example:

        .. code-block:: python

            import sign_language_translator as slt

            landmarks = slt.Landmarks([[[0,1,2], [1,2,3]]])
            landmarks.numpy()
            # array([[[0, 1, 2], [1, 2, 3]]])
        """
        return np.array(self, *args, **kwargs)

    def __array__(self, *args, **kwargs):
        return np.array(self.data, *args, **kwargs)

    def torch(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Union[torch.device, str, None] = None,
    ) -> Tensor:
        """Returns the landmarks data as a PyTorch tensor.

        Args:
            dtype (torch.dtype, optional): The desired data type of the tensor. Defaults to None.
            device (Union[torch.device, str], optional): The desired device for the tensor. Defaults to None.

        Returns:
            torch.Tensor: The sign data as a PyTorch tensor.
        """
        if isinstance(self.data, torch.Tensor):
            return self.data.to(dtype=dtype, device=device)

        return torch.tensor(self.data, dtype=dtype, device=device)

    def tolist(self) -> List[List[List[Union[float, int]]]]:
        """Returns the landmarks data as a 3D nested list of numbers.

        Returns:
            List[List[List[Union[float, int]]]]: The sign data as a nested list.
        """
        return self.data.tolist()

    # ================= #
    #    Concatenate    #
    # ================= #

    @staticmethod
    def concatenate(objects: Iterable[Landmarks]) -> Landmarks:
        """Concatenates a sequence of Landmarks objects along the time dimension (dim=0) and returns a new Landmarks object.

        Args:
            objects (Iterable[Landmarks]): An iterable of Landmarks objects to concatenate.

        Returns:
            Landmarks: A new Landmarks object containing the data concatenated in time dimension.

        Raises:
            ValueError: If the connections of the Landmarks objects are not the same.
        """
        datas = []
        cons = set()
        for obj in objects:
            cons.add(obj.connections.name() if obj._connections else None)
            datas.append(obj.data)

        if len(datas) == 0:
            raise ValueError("No Landmarks objects provided for concatenation.")
        if len(cons) > 1:
            raise ValueError(f"All Landmarks objects must have same connections {cons}")

        return Landmarks(
            ArrayOps.concatenate(datas, dim=0), connections=obj._connections
        )

    def copy(self) -> Landmarks:
        """Creates a deep copy of the Landmarks object.

        Returns:
            Landmarks: A new Landmarks object with the same data and connections.
        """
        return Landmarks(ArrayOps.copy(self.data), connections=self._connections)

    def __copy__(self) -> Landmarks:
        return self.copy()

    # =============== #
    #    Transform    #
    # =============== #

    def transform(
        self,
        transformation: Union[
            Callable[[NDArray], NDArray],
            Callable[[Tensor], Tensor],
        ],
        inplace=False,
    ) -> Landmarks:
        data = transformation(self.data)  # type: ignore

        if inplace:
            self.data = data
            return self

        return Landmarks(data, connections=self._connections)

    # ========================== #
    #    Display / show frame    #
    # ========================== #

    def show(self, player: Literal["jshtml", "html5"] = "jshtml", **kwargs) -> None:
        """Displays the landmarks data as a 3D animation
        in a Jupyter notebook or as a video in a separate window if run from the terminal.

        Args:
            player (Literal['jshtml', 'html5'], optional): The visualization tool to use for displaying the animation. Defaults to "jshtml".
            **kwargs: Additional keyword arguments to pass to the `new_animation` method. See its docstring for details.
        """
        if self._animation is None or kwargs:
            self._animation = self.new_animation(**kwargs)

        if in_jupyter_notebook():
            from IPython.display import HTML, display  # type: ignore

            html = (
                self.animation.to_html5_video()
                if player == "html5"
                else self.animation.to_jshtml()
            )
            display(HTML(html))
            plt.close(self.animation._fig)  # type: ignore # pylint:disable = protected-access
        else:
            self._animation = self.new_animation(**kwargs)  # todo: show old plt object
            plt.pause(len(self) * self.animation._interval / 1000)  # type: ignore
            # plt.close()

        _reset_counter_in_animation_title(self.animation)

    def show_frames_grid(self, rows=3, columns=5, **kwargs):
        """
        Displays a grid of frames equally spaced in time
        drawn as 3D scatter plots & lines connecting the points.

        Args:
            rows (int): The number of rows in the grid. Default is 3.
            columns (int): The number of columns in the grid. Default is 5.
            **kwargs: Additional keyword arguments to be passed to the `slt.vision.landmarks.MatPlot3D.frames_grid` function.
        """
        fig = MatPlot3D.frames_grid(
            self.numpy(),
            (rows, columns),
            **(self.connections.matplot3d_config if self._connections else {}),
            figure_title=kwargs.pop(
                "figure_title", self._caption or basename(self._path or "")
            ),
            **kwargs,
        )
        plt.show(block=False)
        plt.pause(5)
        plt.close(fig)

    def new_animation(
        self,
        title: Optional[str] = "{frame_number}",
        style: Literal["dark_background", "default"] = "default",
        azimuth: float = 20,
        elevation: float = 15,
        roll: float = 0,
        azimuth_delta: float = 0,
        elevation_delta: float = 0,
        roll_delta: float = 0,
        scatter_size: float = 2,
        figure_scale: Optional[float] = 5,
        interval: Union[float, int] = 37,
        repeat_delay: Union[float, int] = 200,
        blit: bool = True,
    ) -> FuncAnimation:
        """Creates a new 3D animation object of the landmarks.

        Args:
            title (Optional[str]): The title of the animation. Can include the placeholder "{frame_number}" to display the frame number. Defaults to "{frame_number}".
            style (Literal["dark_background", "default"]): The color theme of the animation. Defaults to "default".
            azimuth (float): The azimuth angle (rotation around the vertical axis) of the camera view point. Defaults to 20.
            elevation (float): The elevation angle (amount of rise from the horizontal plane) of the camera view point. Defaults to 15.
            roll (float): The roll angle (rotation around the line of sight) of the camera view point. Defaults to 0.
            azimuth_delta (float): The change in azimuth angle per frame. Defaults to 0.
            elevation_delta (float): The change in elevation angle per frame. Defaults to 0.
            roll_delta (float): The change in roll angle per frame. Defaults to 0.
            scatter_size (float): The size of the scatter points. Defaults to 2.
            figure_scale (Optional[float]): The size of the figure. Defaults to 5.
            interval (Union[float, int]): The interval between frames in milliseconds. Defaults to 37.
            repeat_delay (Union[float, int]): The delay between animation replays in milliseconds. Defaults to 200.
            blit (bool): Whether to use blitting for faster updates (non-changing graphic elements are rendered once into a background image). Defaults to True.

        Returns:
            FuncAnimation: The created animation.
        """
        return MatPlot3D.animate(
            self.numpy(),
            **(self.connections.matplot3d_config if self._connections else {}),
            title=title,
            azimuth=azimuth,
            elevation=elevation,
            roll=roll,
            azimuth_delta=azimuth_delta,
            elevation_delta=elevation_delta,
            roll_delta=roll_delta,
            scatter_size=scatter_size,
            figure_scale=figure_scale,
            style=style,
            interval=interval,
            repeat_delay=repeat_delay,
            blit=blit,
        )

    # ========== #
    #    Save    #
    # ========== #

    def save(self, path: str, overwrite=False, precision=4, **kwargs) -> None:
        """Saves the current object's data to a file.
        Supported formats include `.npy`, `.pt`/`.pth` (which contain 3D data) and `.csv` which flattens each frame and puts it into a separate row.
        CSV files also contain a header with letters representing the coordinate axes and numbers identifying the landmark.

        Args:
            path (str): The file path to save the data to.
            overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.
            precision (int, optional): The number of decimal places for saving floating-point values in CSV. Defaults to 4.

        Raises:
            FileExistsError: If the file already exists and `overwrite` is False.
            ValueError: If the file format is not supported.
        """
        path = validate_path_exists(path, overwrite=overwrite)

        if (file_format := path.rsplit(".", 1)[-1].lower()) == "csv":
            np.savetxt(
                path,
                self.data.reshape(self.n_frames, -1),
                delimiter=",",
                fmt=f"%.{precision}f",
                header=self.__make_csv_header(self.n_landmarks, self.n_features),
                comments="",
            )
        elif file_format in ("pt", "pth"):
            torch.save(self.torch(), path)
        elif file_format == "npy":
            np.save(path, self.numpy())
        # elif file_format == "npz":
        #     np.savez_compressed(path, **{os.path.basename(path): self.numpy()})
        else:
            raise ValueError(f"Unsupported file format: '{file_format}'")

    def save_animation(
        self, path, overwrite=True, writer: Optional[str] = None, **kwargs
    ) -> None:
        """Save the video animation of the landmarks data to a file.

        Args:
            path (str): The path to save the animation file.
            overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to True.
            writer (Optional[str], optional): The name of the matplotlib writer to use for saving the animation. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the `new_animation` method.
        """

        path = validate_path_exists(path, overwrite=overwrite)

        if self._animation is None or kwargs:
            self._animation = self.new_animation(**kwargs)

        with tqdm(total=len(self), desc="Saving animation", unit="frames") as bar:
            self.animation.save(
                path, progress_callback=(lambda *args: bar.update()), writer=writer
            )

        _reset_counter_in_animation_title(self.animation)
        # plt.close()

    def save_frames_grid(
        self,
        path: str,
        rows: int = 3,
        columns: int = 5,
        overwrite=True,
        **kwargs,
    ) -> None:
        """Save an image file of a grid of 3D visualizations of the landmarks data.

        Args:
            path (str): The path to save the image.
            rows (int, optional): The number of rows in the grid. Defaults to 3.
            columns (int, optional): The number of columns in the grid. Defaults to 5.
            overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to True.
            **kwargs: Additional keyword arguments to customize the grid passed to the `slt.vision.landmarks.MatPlot3D.frames_grid` function.
        """
        path = validate_path_exists(path, overwrite=overwrite)

        fig = MatPlot3D.frames_grid(
            self.numpy(),
            (rows, columns),
            **(self.connections.matplot3d_config if self._connections else {}),
            figure_title=kwargs.pop(
                "figure_title", self._caption or basename(self._path or "")
            ),
            **kwargs,
        )
        fig.savefig(path)
        plt.close(fig)

    # ========== #
    #    Load    #
    # ========== #

    @classmethod
    def load(cls, path: str, **kwargs) -> Landmarks:
        """Class method to load landmarks data from a file and return a new Landmarks object.
        The supported file extensions are `.npy` & `.pt` with must contain 3D arrays (n_frames, n_landmarks, n_features)
        and `.csv` which must have `n_frames` rows and `n_landmarks * n_features` columns.

        The header row in `.csv` is optional if the filename contains the name of a supported embedding model (see `load_asset` function for example models).
        The columns in the .csv are expected to be in the format: [<axis-letter><landmark-number>,...] (e.g. x0, y0, z0, x1, y1, z1, ..., xn, yn, zn).
        Possible axis-letters: x, y, z, a-w, aa-zz, ... (only the first 3 are required to be in that order).

        Args:
            path (str): The file path to load the data from.

        Returns:
            Landmarks: A new Landmarks object containing the loaded data.
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
    ) -> Landmarks:
        """Class method to load a landmarks file from a one-time-auto-downloaded dataset archive
        and return a new Landmarks object.

        Args:
            label (str): The filename of the landmarks asset to load. 'landmarks/' is prepended to the label if it does not start with it. An example is 'landmarks/pk-hfad-1_airport.landmarks-mediapipe.csv') for embedding of a dictionary video. General syntax is `landmarks/country-organization-number_text[_person_camera].landmarks-model.extension`.
            archive_name (Optional[str], optional): The name of the archive which contains the landmarks asset. If None, the archive name is inferred from the label. An example is `datasets/pk-hfad-1.landmarks-mediapipe-csv.zip`. General syntax is `datasets/country-organization-number[_person_camera].landmarks-model-extension.zip`. Defaults to None.
            overwrite (bool, optional): Whether to overwrite the landmarks asset if it is already extracted. Defaults to False.
            progress_bar (bool, optional): Whether to display a progress bar while downloading the archive or extracting the asset. Defaults to True.
            leave (bool, optional): Whether to leave the progress bar after the operation is complete. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the Landmarks constructor.

        Raises:
            FileNotFoundError: If no landmarks assets are found for the given label.

        Warns:
            UserWarning: If multiple landmarks assets match the given label and the only first asset is used.

        Returns:
            Landmarks: An instance of the Landmarks class representing the dataset video embedding that matched the label.

        Example:

            .. code-block:: python

                import sign_language_translator as slt

                # Load a dictionary video's landmark embedding asset
                landmarks = slt.Landmarks.load_asset("pk-hfad-1_airplane.landmarks-mediapipe.csv")

                # Load a replication video's landmarks from the built-in datasets
                landmarks = slt.Landmarks.load_asset("landmarks/pk-hfad-1_airplane_dm0001_front.landmarks-mediapipe.csv", archive_name="datasets/pk-hfad-1_dm0001_front.landmarks-mediapipe-csv.zip")
        """

        if "/" not in label:
            label = f"landmarks/{label}"

        paths = Assets.extract(
            label,
            archive_name_or_regex=archive_name,
            download_archive=True,
            overwrite=overwrite,
            progress_bar=progress_bar,
            leave=leave,
        )

        if len(paths) == 0:
            raise FileNotFoundError(f"No landmarks assets found for '{label}'")
        if len(paths) > 1:
            warn(f"Multiple landmarks assets matched '{label}'. Using first of:{paths}")

        return cls.load(paths[0], **kwargs)

    def __initialize_from_arguments(
        self,
        sign: Union[
            str,
            NDArray,
            Tensor,
            Sequence[NDArray],
            Sequence[Tensor],
            Sequence[Sequence[Sequence[Union[float, int]]]],
        ],
        **kwargs,
    ):
        """
        Initialize the current landmarks object from the provided arguments.
        The sign parameter can be a string (path to a file), an NDArray, a Tensor, or a sequence of these types.
        """
        if isinstance(sign, str):
            self._from_path(sign)
        elif isinstance(sign, (np.ndarray, Tensor)):
            self._from_data(sign)
        elif isinstance(sign, Sequence):
            if any(isinstance(x, Tensor) for x in sign):
                sign = [(s if isinstance(s, Tensor) else torch.tensor(s)) for s in sign]
                sign = (torch.concatenate if sign[0].ndim == 3 else torch.stack)(sign)
                self._from_data(sign)
            elif isinstance(sign[0], (np.ndarray, float, int, Sequence)):
                self._from_data(np.array(sign))
            else:
                raise ValueError("Unsupported data structure for sign argument.")
        else:
            raise TypeError(
                "Unsupported type for sign argument. provide a path (csv, npy, pt, pth)"
                " or an array of shape (n_frames, n_landmarks, n_coordinates)"
            )

        if kwargs.get("connections") is not None:
            self.connections = kwargs["connections"]

    def _from_path(self, path: str):
        """Read the file specified by the provided path and load all required data into the current object.

        Args:
            path (str): The file path to load the data from.

        Raises:
            ValueError: If the file format is unsupported.
        """
        self._path = os.path.abspath(path)

        # infer connections from filename
        if self._connections is None and (
            inferred := self.__infer_landmarks_model_from_filename(basename(path))
        ):
            self.connections = inferred

        if path.endswith(".npy"):
            data = np.load(path)
        elif path.endswith(".pt"):
            data = torch.load(path)
        elif path.endswith(".csv"):
            data = self.__load_csv(path)
        # elif path.endswith(".npz"):
        # elif path.endswith(".json"):
        else:
            raise ValueError(f"Unsupported file format: {path}")

        self._from_data(data)

    def __load_csv(self, path: str, sep=",") -> NDArray:
        if header := self.__get_csv_header(path, sep=sep):
            columns = header.split(sep)
            n_landmarks = len({value.lstrip(ascii_letters) for value in columns})
            n_features = len({value.rstrip(digits) for value in columns})
        else:
            if self._connections is None:
                # ToDo: parse path for model name (currently assuming that it's already been parsed)
                raise ValueError(
                    "Could not infer the number of landmarks and coordinates "
                    f"from the file name '{basename(path)}' nor the header inside the file."
                )
            n_landmarks = self.connections.n_landmarks
            n_features = self.connections.n_features

        data = np.loadtxt(path, delimiter=sep, skiprows=int(bool(header)), ndmin=2)

        return data.reshape(data.shape[0], n_landmarks, n_features)

    def _from_data(self, sign_data: Union[NDArray, Tensor]):
        """Initialize the current landmarks object from a provided 3D array.

        Args:
            sign_data (Union[NDArray, Tensor]): A 3D array of shape (n_frames, n_landmarks, n_features).
        """
        if isinstance(sign_data, np.ndarray):
            if not np.issubdtype(sign_data.dtype, np.number):
                raise ValueError("Expected sign_data to be a numeric sequence.")

        if sign_data.ndim != 3:
            raise ValueError(
                "Expected sign_data to be 3D (n_frames, n_landmarks, n_features)"
                f" but got {sign_data.ndim}D"
            )
        self._data = sign_data

    # ================ #
    #    Properties    #
    # ================ #

    @property
    def data(self) -> Union[NDArray, Tensor]:
        """The landmarks data which is a 3D array or tensor of shape (n_frames, n_landmarks, n_features)."""
        if self._data is None:
            raise ValueError("No data has been loaded yet")
        return self._data

    @data.setter
    def data(self, value: Union[NDArray, Tensor]):
        if value.ndim != 3:
            raise ValueError(
                "Expected value to be 3D (n_frames, n_landmarks, n_features)"
                f" but got {value.ndim}D"
            )
        if self._connections and value.shape[1] != self.connections.n_landmarks:
            raise ValueError(
                f"Expected data to have {self.connections.n_landmarks} landmarks, got {value.shape[1]}"
            )
        self._data = value

    def __len__(self) -> int:
        return self.data.shape[0]

    @property
    def n_frames(self) -> int:
        """The number of frames or time-steps in the landmarks data object."""
        return len(self)

    @property
    def n_landmarks(self) -> int:
        """The number of landmarks in each frame."""
        return self.data.shape[1]

    @property
    def n_features(self) -> int:
        """The number of features (coordinates) for each landmark."""
        return self.data.shape[2]

    @property
    def n_coordinates(self) -> int:
        """The number of axes/coordinates (features) for each landmark."""
        return self.n_features

    @property
    def shape(self) -> Tuple[int, ...]:
        """number of elements in each of the data array's dimensions e.g. (n_frames, n_landmarks, n_features)"""
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the landmarks data array (should be 3)."""
        return self.data.ndim

    @property
    def connections(self) -> BaseConnections:
        """
        Object defining the order in which landmarks are connected during display
        and other properties depending on the model used to extract the landmarks.

        Raises:
            ValueError: If this property is accessed before landmarks connections have been defined.
        """
        if self._connections is None:
            raise ValueError("No landmarks connections have been defined yet.")
        return self._connections

    @connections.setter
    def connections(self, value: Union[BaseConnections, str]):
        if isinstance(value, str):
            value = get_connections(value)
        elif not isinstance(value, BaseConnections) and value is not None:
            raise TypeError(f"Expected BaseConnections object or string, got {value}")

        # check if connections are compatible with data
        if self._data is not None and value.n_landmarks != self.n_landmarks:
            raise ValueError(
                f"Expected connections to have {self.n_landmarks} landmarks, got {value.n_landmarks}"
            )
        self._connections = value

    @property
    def animation(self) -> FuncAnimation:
        """
        Visualization of the landmarks on a 3D graph.

        Note:
            For interactive display in a Jupyter notebook, use `%matplotlib widget` magic command and then run a cell with `landmarks_obj.animation` on last line.
        """
        if self._animation is None:
            self._animation = self.new_animation()
        return self._animation

    # =========== #
    #    Utils    #
    # =========== #

    def __normalize_slice_indices(self, indices) -> Tuple:
        if not isinstance(indices, tuple):
            indices = (indices,)

        # typecast indices into self.data's compatible type
        indices = tuple(
            (
                ArrayOps.cast(idx, type(self.data), _dtype=int)
                if isinstance(idx, (np.ndarray, Tensor))
                else slice(idx, (idx + 1) or None) if isinstance(idx, int) else idx
            )
            for idx in indices
        )
        return indices

    def __make_csv_header(self, n_landmarks: int, n_features: int):
        axes = list("xyz" + ascii_letters[:23])
        if n_features > 26:
            axes += [
                chr(i) + chr(j)
                for i in range(ord("a"), ord("z") + 1)
                for j in range(ord("a"), ord("z") + 1)
            ]
        header = ",".join(
            f"{axes[c]}{l}" for l in range(n_landmarks) for c in range(n_features)
        )
        return header

    def __get_csv_header(self, path: str, sep=",", encoding="utf-8") -> str:
        with open(path, "r", encoding=encoding) as f:
            line = f.readline().strip()

        for value in line.split(sep):
            try:
                if value:
                    float(value)
            except ValueError:
                return line
        return ""

    def __infer_landmarks_model_from_filename(self, filename: str) -> str:
        # todo: make robust!
        # todo: parse using asset name parser
        model_names = [
            "mediapipe-image",
            "mediapipe-world",
            # "mediapipe-all",
            # "mediapipe",
        ]
        for name in model_names:
            if name in filename:
                return name
        return ""
