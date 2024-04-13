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
    >>> from sign_language_translator.vision.landmarks.landmarks import Landmarks
    >>> landmarks = Landmarks([[[0,1,2], [1,2,3]]])  # 1 frames, 2 landmarks, 3 coordinates
    >>> landmarks.show()
    >>> landmarks.save('landmarks_file.csv')
    >>> landmarks = Landmarks.load('landmarks_file.csv')
    >>> print(landmarks.shape)
"""

from __future__ import annotations

import os
from os.path import basename
from string import ascii_letters, digits
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from sign_language_translator.config.enums import SignFormats
from sign_language_translator.utils.arrays import ArrayOps
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

        self.__initialize_from_arguments(sign, **kwargs)

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
        return Landmarks(self.data[indices])

    # ========================= #
    #    Typecast / get data    #
    # ========================= #

    def numpy(self, *args, **kwargs) -> NDArray:
        """Returns the landmarks data as a numpy array.
        Additional arguments are passed to the numpy.array constructor.

        Returns:
            NDArray: The sign data as a NumPy array.
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

    # ========================== #
    #    Display / show frame    #
    # ========================== #

    def show(self, **kwargs) -> None:
        raise NotImplementedError()

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
        """
        datas = []
        for obj in objects:
            datas.append(obj.data)

        return Landmarks(ArrayOps.concatenate(datas, dim=0))

    # =============== #
    #    Transform    #
    # =============== #

    def transform(
        self,
        transformation: Union[
            Callable[[NDArray], NDArray],
            Callable[[Tensor], Tensor],
        ],
    ):
        # todo: decide in-place transformation or return copy?
        self._data = transformation(self.data)  # type: ignore

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
            ValueError: If the file format is unsupported.
        """
        if not overwrite and os.path.exists(path):
            raise FileExistsError(f"File already exists: '{path}' (Use overwrite=True)")

        if (dir_name := os.path.dirname(path)) not in ("", ".", ".."):
            os.makedirs(dir_name, exist_ok=True)

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

    # ========== #
    #    Load    #
    # ========== #

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

    @classmethod
    def load(cls, path: str, **kwargs) -> Landmarks:
        """Class method to load landmarks data from a file and return a new Landmarks object.
        The supported file extensions are `.npy` & `.pt` with must contain 3D arrays (n_frames, n_landmarks, n_features)
        and `.csv` which must have `n_frames` rows and `n_landmarks * n_features` columns.

        The header row in `.csv` is optional if the filename contains the name of a supported embedding model (see our `load_asset` for examples).
        The columns in the .csv are expected to be in the format: [<axis-letter><landmark-number>,...] (e.g. x0, y0, z0, x1, y1, z1, ..., xn, yn, zn).
        Possible axis-letters: x, y, z, a-w, aa-zz, ... (only the first 3 are required to be in that order).

        Args:
            path (str): The file path to load the data from.

        Returns:
            Landmarks: A new Landmarks object containing the loaded data.
        """
        return cls(path, **kwargs)

    @classmethod
    def load_asset(cls, label: str, **kwargs) -> Landmarks:
        # ToDo: [down]load from dataset | extract from archive
        raise NotImplementedError()

    def _from_path(self, path: str):
        """Read the file specified by the provided path and load all required data into the current object.

        Args:
            path (str): The file path to load the data from.

        Raises:
            ValueError: If the file format is unsupported.
        """
        self._path = path
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
        # todo: handle connections from __init__ arguments
        n_landmarks, n_features = self.__infer_dimensions_from_filename(basename(path))

        if header := self.__get_csv_header(path, sep=sep):
            columns = header.split(sep)
            n_landmarks = len({value.lstrip(ascii_letters) for value in columns})
            n_features = len({value.rstrip(digits) for value in columns})

        data = np.loadtxt(path, delimiter=sep, skiprows=int(bool(header)))

        if n_landmarks < 1 or n_features < 1:
            raise ValueError(
                "Could not infer the number of landmarks and coordinates "
                f"from the file name '{basename(path)}' nor the header inside the file."
            )

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

    def __infer_dimensions_from_filename(self, filename: str) -> Tuple[int, int]:
        """figure out the number of landmarks and coordinates from a conventional filename.

        Args:
            filename (str): the name of the landmarks file. expected to be a csv dataset file.

        Returns:
            Tuple[int, int]: (n_landmarks, n_coordinates) if detected else (-1, -1)
        """
        model_name = self.__infer_landmarks_model_from_filename(filename)
        # todo: use connections class object
        if model_name in ("mediapipe-image", "mediapipe-world"):
            return 75, 5
        if model_name in ("mediapipe-all", "mediapipe"):
            return 150, 5
        return (-1, -1)

    def __infer_landmarks_model_from_filename(self, filename: str) -> str:
        # todo: make robust
        model_names = [
            "mediapipe-image",
            "mediapipe-world",
            "mediapipe-all",
            "mediapipe",
        ]
        for name in model_names:
            if name in filename:
                return name
        return ""
