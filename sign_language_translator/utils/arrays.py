from typing import Iterable, List, Sequence, Type

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch import long as torch_long

__all__ = [
    "ArrayOps",
    "linear_interpolation",
]


class ArrayOps:
    @staticmethod
    def floor(array: NDArray | Tensor) -> NDArray | Tensor:
        if isinstance(array, np.ndarray):
            return np.floor(array)
        elif isinstance(array, Tensor):
            return array.floor()
        else:
            raise TypeError(f"Invalid type for flooring: {type(array)}")

    @staticmethod
    def ceil(array: NDArray | Tensor) -> NDArray | Tensor:
        if isinstance(array, np.ndarray):
            return np.ceil(array)
        elif isinstance(array, Tensor):
            return array.ceil()
        else:
            raise TypeError(f"Invalid type for ceiling: {type(array)}")

    @staticmethod
    def take(
        array: NDArray | Tensor, index: NDArray | Tensor | List, dim: int = 0
    ) -> NDArray | Tensor:
        if isinstance(array, np.ndarray):
            if not isinstance(index, np.ndarray):
                index = np.array(index)
            return np.take(array, index.astype(int), axis=dim)
        elif isinstance(array, Tensor):
            if not isinstance(index, Tensor):
                index = Tensor(index)
            return array.index_select(dim, index.type(torch_long))
        else:
            raise TypeError(f"Invalid type for taking: {type(array)}")

    @staticmethod
    def cast(
        x: NDArray | Tensor | List | Iterable, data_type: Type[np.ndarray | Tensor]
    ) -> NDArray | Tensor:
        if data_type == np.ndarray:
            return np.array(x)
        elif data_type == Tensor:
            return Tensor(x)
        else:
            raise TypeError(f"Invalid type for array casting: {data_type}")


def linear_interpolation(
    array: NDArray[np.float64] | Tensor | List,
    new_indexes: Sequence[int | float] | None = None,
    old_x: Sequence[int | float] | None = None,
    new_x: Sequence[int | float] | None = None,
    dim: int = 0,
) -> NDArray | Tensor:
    """
    Perform linear interpolation on a multidimensional array or tensor along a dimension.

    This function essentially connects all consecutive values in a multidimensional array with
    straight lines along a specified dimension, so that intermediate values can be calculated.
    It takes the input array, a set of new indexes or alternatively old & new coordinate axes,
    and a dimension along which to perform interpolation.

    Parameters:
        array (NDArray[np.float64] | Tensor): The input array or tensor to interpolate.
        new_indexes (Sequence[int | float] | None, optional): The new indexes at which to interpolate the data.
            If None, it infers new_indexes from `old_x` and `new_x`.
        old_x (Sequence[int | float] | None, optional): The old coordinate values corresponding to the data in `array`.
            If None, it uses `new_indexes` argument.
        new_x (Sequence[int | float] | None, optional): The new coordinate values corresponding to the .
            If None, it uses `new_indexes` argument.
        dim (int, optional): The dimension along which to perform interpolation. Default is 0.

    Returns:
        NDArray | Tensor: The result of linear interpolation along the specified dimension.

    Raises:
        ValueError: If both or neither of `new_indexes` and `old_x` & `new_x` are provided.
        If `new_indexes` is not 1 dimensional.

    Examples:
        .. code-block: python
            data = np.array([1, 2, 3, 5])
            new_indexes = np.array([1.5, 0.5, 2.5])
            interpolated_data = linear_interpolation(data, new_indexes)
            print(interpolated_data)
            # array([2.5, 1.5, 4. ])

            old_x = [0, 4, 4.5, 5]
            new_x = [0, 1, 2, 2.5, 3, 4, 5]
            interpolated_data = linear_interpolation(data, old_x=old_x, new_x=new_x)
            print(interpolated_data)
            # array([1.   , 1.25 , 1.5  , 1.625, 1.75 , 2.   , 5.   ])

    Note:
        This function supports both NumPy arrays and PyTorch tensors as input.

    """
    if isinstance(array, list):
        array = np.array(array)

    indexes_, dim = __validate_linear_interpolation_args(
        array, new_indexes, old_x, new_x, dim
    )

    # interpolate
    floor_values = ArrayOps.floor(indexes_)
    ceiling_values = ArrayOps.ceil(indexes_)
    fraction = (indexes_ - floor_values).reshape(
        [len(indexes_) if axis == dim else 1 for axis in range(array.ndim)]
    )

    interpolated = ArrayOps.take(array, floor_values, dim) * (1 - fraction)
    interpolated += ArrayOps.take(array, ceiling_values, dim) * fraction

    return interpolated


def __validate_linear_interpolation_args(array, new_indexes, old_x, new_x, dim):
    if (new_indexes is None and (old_x is None or new_x is None)) or (
        new_indexes is not None and (old_x is not None or new_x is not None)
    ):
        raise ValueError(
            "Either `new_indexes` or both `old_x` and `new_x` must be provided."
        )
    if new_indexes is None:
        new_indexes = np.interp(new_x, old_x, np.arange(len(old_x)))  # type: ignore

    if not isinstance(array, (np.ndarray, Tensor)):
        array = np.array(array)
    indexes_ = ArrayOps.cast(new_indexes, type(array))  # type: ignore

    if indexes_.ndim != 1:
        raise ValueError(
            f"Invalid indexes shape: {indexes_.shape}. Must be 1-dimensional."
        )
    if dim >= array.ndim:
        raise ValueError(f"Invalid dim: {dim}. Must be between 0 and {array.ndim-1}.")

    out_of_bounds = (indexes_ < 0) & (indexes_ > array.shape[dim] - 1)
    if out_of_bounds.any():
        raise ValueError(
            f"Invalid indexes for interpolation: {indexes_[out_of_bounds.tolist()]}. "
            f"Must be between >= 0 and <= {array.shape[0]-1}."
        )
    if dim < 0 or dim >= array.ndim:
        raise ValueError(f"Invalid dim: {dim}. Must be between 0 and {array.ndim-1}.")

    return indexes_, dim
