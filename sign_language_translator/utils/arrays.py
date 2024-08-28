from typing import Dict, List, Optional, Sequence, Tuple, Type, Union, overload

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

__all__ = [
    "ArrayOps",
    "linear_interpolation",
    "adjust_vector_angle",
    "align_vectors",
]


class ArrayOps:
    @staticmethod
    def floor(
        array: Union[NDArray, Tensor, Sequence[Union[float, int]], float, int]
    ) -> Union[NDArray, Tensor]:
        if isinstance(array, (np.ndarray, list, tuple, float, int)):
            return np.floor(array)

        if isinstance(array, Tensor):
            return array.floor()

        raise TypeError(f"Invalid type for flooring: {type(array)}")

    @staticmethod
    def ceil(
        array: Union[NDArray, Tensor, Sequence[Union[float, int]], float, int]
    ) -> Union[NDArray, Tensor]:
        if isinstance(array, (np.ndarray, list, tuple, float, int)):
            return np.ceil(array)

        if isinstance(array, Tensor):
            return array.ceil()

        raise TypeError(f"Invalid type for ceiling: {type(array)}")

    @staticmethod
    def take(
        array: Union[NDArray, Tensor, List],
        index: Union[NDArray, Tensor, List, int],
        dim: int = 0,
    ) -> Union[NDArray, Tensor]:
        if isinstance(array, (np.ndarray, list, tuple)):
            if not isinstance(index, np.ndarray):
                index = np.array(index)
            return np.take(array, index.astype(int), axis=dim)

        if isinstance(array, Tensor):
            if not isinstance(index, Tensor):
                index = torch.tensor(index)
            return array.index_select(dim, index.type(torch.long))

        raise TypeError(f"Invalid type for taking: {type(array)}")

    @staticmethod
    def cast(
        x: Union[NDArray, Tensor, Sequence[Union[float, int]]],
        data_type: Type[Union[np.ndarray, Tensor]],
        _dtype: Optional[Union[Type[torch.dtype], Type[np.dtype], Type]] = None,
    ) -> Union[NDArray, Tensor]:
        """Typecast some multidimensional data to numpy array or torch Tensor.

        Args:
            x (Union[NDArray, Tensor, Sequence[Union[float, int]]]): The input array or tensor.
            data_type (Type[Union[np.ndarray, Tensor]]): The data type to cast the input array or tensor to.
            _dtype (Optional[Union[Type[torch.dtype], Type[np.dtype], Type]], optional): The new data type of the values inside the array. None means original dtype is kept. Defaults to None.

        Raises:
            ValueError: If the `data_type` is not np.ndarray or Tensor.

        Returns:
            Union[NDArray, Tensor]: The casted array or tensor.
        """

        if data_type == np.ndarray:
            return np.array(x, dtype=_dtype)

        if data_type == Tensor:
            type_map: Dict = {int: torch.int64, float: torch.float64}
            _dtype = type_map.get(_dtype, _dtype)
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if isinstance(x, Tensor):
                return x.to(dtype=_dtype)  # type: ignore
            return torch.tensor(x, dtype=_dtype)  # type: ignore

        raise ValueError(f"Invalid data_type for array casting: {data_type}")

    @staticmethod
    def norm(
        x: Union[NDArray, Tensor, Sequence[Union[float, int]]],
        dim: Optional[int] = None,
        keepdim=False,
    ) -> Union[NDArray, Tensor]:
        """
        Compute the norm of a given array or tensor along a specified dimension.

        Args:
            x (Union[NDArray, Tensor, Sequence[Union[float, int]]]): The input array or tensor.
            dim (Optional[int]): The dimension along which to compute the norm. If None, the norm is computed over the entire array or tensor. Default is None.
            keepdim (bool): Whether to keep the dimension of the input array or tensor after computing the norm. Default is False.

        Returns:
            Union[NDArray, Tensor]: The norm of the input array or tensor.

        Raises:
            TypeError: If the input type is not supported.
        """

        if isinstance(x, (np.ndarray, list, tuple)):
            return np.linalg.norm(x, axis=dim, keepdims=keepdim)  # type: ignore

        if isinstance(x, Tensor):
            return x.norm(dim=dim, keepdim=keepdim)

        raise TypeError(f"Invalid type for norm: {type(x)}")

    @staticmethod
    def svd(
        x: Union[NDArray, Tensor, Sequence[Sequence[Union[float, int]]]]
    ) -> Tuple[Union[NDArray, Tensor], Union[NDArray, Tensor], Union[NDArray, Tensor]]:
        """
        Compute the singular value decomposition of a given array or tensor.

        Args:
            x (Union[NDArray, Tensor, Sequence[Sequence[Union[float, int]]]]): The input array or tensor.

        Returns:
            Tuple[Union[NDArray, Tensor], Union[NDArray, Tensor], Union[NDArray, Tensor]]: The (Rotation, coordinate scaling, reflection) matrices of the input array or tensor.

        Raises:
            TypeError: If the input type is not supported.
        """

        if isinstance(x, (np.ndarray, list, tuple)):
            return np.linalg.svd(x)

        if isinstance(x, Tensor):
            U, S, V = x.svd()
            return U, S, V.T

        raise TypeError(f"Invalid type for svd: {type(x)}")

    @staticmethod
    def top_k(
        x: Union[NDArray, Tensor, Sequence[Union[float, int]]],
        k: int,
        dim: int = -1,
        largest=True,
    ) -> Union[Tuple[NDArray, NDArray], Tuple[Tensor, Tensor]]:
        """
        Compute the top k values and their indices along a specified dimension of a given array or tensor.

        Args:
            x (Union[NDArray, Tensor, Sequence[Union[float, int]]]): The input array or tensor.
            k (int): The number of top values to return.
            dim (int, optional): The dimension along which to compute the top k values. Default is -1.
            largest (bool, optional): Whether to return the largest or smallest k values. Default is True.

        Returns:
            Tuple[Union[NDArray, Tensor], Union[NDArray, Tensor]]: The top k values and their indices along the specified dimension.

        Raises:
            TypeError: If the input type is not supported.
        """

        x_tensor = ArrayOps.cast(x, Tensor)
        values, indices = x_tensor.topk(k, dim=dim, largest=largest, sorted=True)

        if isinstance(x, np.ndarray):
            return values.numpy(), indices.numpy()

        return values, indices

    @overload
    @staticmethod
    def concatenate(arrays: Sequence[NDArray], dim: int = 0) -> NDArray: ...

    @overload
    @staticmethod
    def concatenate(arrays: Sequence[Tensor], dim: int = 0) -> Tensor: ...

    @staticmethod
    def concatenate(
        arrays: Union[Sequence[NDArray], Sequence[Tensor]], dim: int = 0
    ) -> Union[NDArray, Tensor]:
        """
        Concatenate a sequence of arrays or tensors along a specified dimension.

        Args:
            arrays (Union[Sequence[NDArray], Sequence[Tensor]]): The sequence of arrays or tensors to concatenate.
            dim (int, optional): The dimension along which to concatenate the arrays or tensors. Default is 0.

        Returns:
            Union[NDArray, Tensor]: The concatenated array or tensor.

        Raises:
            TypeError: If the input type is not supported.
        """

        if isinstance(arrays[0], np.ndarray):
            return np.concatenate(arrays, axis=dim)
        if isinstance(arrays[0], Tensor):
            return torch.concatenate(
                [(x if isinstance(x, Tensor) else torch.tensor(x)) for x in arrays],
                dim=dim,
            )

        raise TypeError(f"Invalid type for concatenation: {type(arrays[0])}")

    @overload
    @staticmethod
    def abs(x: NDArray) -> NDArray: ...

    @overload
    @staticmethod
    def abs(x: Tensor) -> Tensor: ...

    @staticmethod
    def abs(x: Union[NDArray, Tensor]) -> Union[NDArray, Tensor]:
        """
        Compute the absolute value of a given array or tensor.

        Args:
            x (Union[NDArray, Tensor]): The input array or tensor.

        Returns:
            Union[NDArray, Tensor]: The absolute value of the input array or tensor.

        Raises:
            TypeError: If the input type is not supported.
        """

        if isinstance(x, np.ndarray):
            return np.abs(x)
        if isinstance(x, Tensor):
            return x.abs()

        raise TypeError(f"Invalid type for absolute value: {type(x)}")

    @overload
    @staticmethod
    def copy(x: NDArray) -> NDArray: ...

    @overload
    @staticmethod
    def copy(x: Tensor) -> Tensor: ...

    @staticmethod
    def copy(x: Union[NDArray, Tensor]) -> Union[NDArray, Tensor]:
        """
        Create a copy of a given array or tensor.

        Args:
            x (Union[NDArray, Tensor]): The input array or tensor.

        Returns:
            Union[NDArray, Tensor]: A deep copy of the input array or tensor.

        Raises:
            TypeError: If the input type is not supported.
        """

        if isinstance(x, np.ndarray):
            return np.copy(x)
        if isinstance(x, Tensor):
            return torch.clone(x)

        raise TypeError(f"Invalid type for copying array: {type(x)}")

    @staticmethod
    def linspace(
        start: Union[float, int],
        end: Union[float, int],
        n_steps: int,
        data_type: Type[Union[np.ndarray, Tensor]] = np.ndarray,
        endpoint=True,
    ) -> Union[NDArray, Tensor]:
        """Generate an array or tensor with equally spaced values between `start` and `end`.

        Args:
            start (Union[float, int]): The starting value of the sequence. The value is inclusive.
            end (Union[float, int]): The end value of the sequence. The value is inclusive if `endpoint` is True.
            n_steps (int): The number of samples to generate. Must be non-negative.
            data_type (Type[Union[np.ndarray, Tensor]], optional): The data type of the output array. Defaults to np.ndarray.
            endpoint (bool, optional): Whether to include the `end` value in the sequence. Defaults to True.

        Raises:
            ValueError: If `data_type` is not np.ndarray or Tensor.

        Returns:
            Union[NDArray, Tensor]: The generated array or tensor.
        """
        if data_type == np.ndarray:
            return np.linspace(start, end, n_steps, endpoint=endpoint)

        if data_type == Tensor:
            return torch.linspace(start, end, n_steps + int(not endpoint))[
                : -int(not endpoint) or None
            ]

        raise ValueError(f"Invalid `data_type` arg for creating linspace: {data_type}")

    @staticmethod
    def random_uniform(
        size: Sequence[int],
        start: Union[float, int] = 0,
        end: Union[float, int] = 1,
        data_type: Type[Union[np.ndarray, Tensor]] = np.ndarray,
    ) -> Union[NDArray, Tensor]:
        """Generate a random array of the specified size with values uniformly distributed between [`start`, `end`).

        Args:
            size (Sequence[int]): The shape of the output array or tensor.
            start (Union[float, int], optional): The lower bound of the uniform distribution. The value is inclusive. Defaults to 0.
            end (Union[float, int], optional): The upper bound of the uniform distribution. The value is exclusive. Defaults to 1.
            data_type (Type[Union[np.ndarray, Tensor]], optional): The data type of the output array. Defaults to np.ndarray.

        Raises:
            ValueError: If `data_type` is not np.ndarray or Tensor.

        Returns:
            Union[NDArray, Tensor]: The array or tensor filled with random values.

        Note:
            Uses torch's random number generator to generate random values even for NumPy arrays.
        """

        if data_type not in (np.ndarray, Tensor):
            raise ValueError(f"Invalid `data_type` arg for random uniform: {data_type}")

        random_values = torch.empty(size).uniform_(start, end)

        if data_type == np.ndarray:
            random_values = random_values.numpy()

        return random_values

    @staticmethod
    def random_normal(
        size: Sequence[int],
        loc: Union[float, int] = 0,
        scale: Union[float, int] = 1,
        start: Union[float, int] = float("-inf"),
        end: Union[float, int] = float("inf"),
        data_type: Type[Union[np.ndarray, Tensor]] = np.ndarray,
    ) -> Union[NDArray, Tensor]:
        """Generate an array or tensor of the specified shape filled with random values from a normal (Gaussian) distribution.
        Optionally truncate the distribution to the range [`start`, `end`].

        Args:
            size (Sequence[int]): The shape of the output array or tensor.
            loc (Union[float, int], optional): The mean ("centre") of the distribution. Defaults to 0.
            scale (Union[float, int], optional): The standard deviation (spread or "width") of the distribution. Must be non-negative. Defaults to 1.
            start (Union[float, int], optional): The lower bound of the distribution. Defaults to float("-inf").
            end (Union[float, int], optional): The upper bound of the distribution. Defaults to float("inf").
            data_type (Type[Union[np.ndarray, Tensor]], optional): The data type of the output array. Defaults to np.ndarray.

        Raises:
            ValueError: If `data_type` is not np.ndarray or torch.Tensor.

        Returns:
            Union[NDArray, Tensor]: The random values filled array or tensor.

        Note:
            Uses torch's random number generator to generate random values even for NumPy arrays.
        """
        if data_type not in (np.ndarray, Tensor):
            raise ValueError(f"Invalid `data_type` arg for random normal: {data_type}")

        random_values = (
            torch.normal(loc, scale, size)
            if start == float("-inf") and end == float("inf")
            else torch.nn.init.trunc_normal_(torch.empty(size), loc, scale, start, end)
        )

        if data_type == np.ndarray:
            random_values = random_values.numpy()

        return random_values

    @overload
    @staticmethod
    def steps(
        n_steps: int,
        anchors: Tensor = torch.Tensor([0, -1, 2]),
        random_uniform_frac: float = 0.2,
        random_normal_frac: float = 0.3,
        n_clusters: int = 1,
        cluster_std: Optional[float] = None,
        anchor_spacing_blend: float = 0.5,
    ) -> Tensor: ...

    @overload
    @staticmethod
    def steps(
        n_steps: int,
        anchors: Union[NDArray, Sequence[Union[float, int]]] = (0, 1),
        random_uniform_frac: float = 0.2,
        random_normal_frac: float = 0.3,
        n_clusters: int = 1,
        cluster_std: Optional[float] = None,
        anchor_spacing_blend: float = 0.5,
    ) -> NDArray: ...

    @staticmethod
    def steps(
        n_steps: int,
        anchors: Union[NDArray, Tensor, Sequence[Union[float, int]]] = (0, 1),
        random_uniform_frac: float = 0.2,
        random_normal_frac: float = 0.3,
        n_clusters: int = 1,
        cluster_std: Optional[float] = None,
        anchor_spacing_blend: float = 0.5,
    ) -> Union[NDArray, Tensor]:
        """Generates a sequence of steps based on a combination of linear interpolation,
        random uniform distribution, and random normal distribution.

        Args:
            n_steps (int): The total number of steps to generate.
            anchors (Union[NDArray, Tensor, Sequence[Union[float, int]]], optional): The points between & through which the steps are interpolated. Defaults to (0, 1).
            random_uniform_frac (float, optional): The fraction of steps generated using a random uniform distribution. Must be between 0 and 1. Defaults to 0.2.
            random_normal_frac (float, optional): The fraction of steps generated using a random normal distribution. Must be between 0 and 1. Defaults to 0.3.
            n_clusters (int, optional): The number of concentrated spots to add using the random normal (gaussian distribution) steps (around cluster centroids selected from a uniform distribution). Defaults to 1.
            cluster_std (Optional[float], optional): The standard deviation (spread) of the normal distribution generating the concentrated spots. If None, it is calculated based on the anchor gap and number of clusters (`std(gaps)/10/n_clusters`). Defaults to None.
            anchor_spacing_blend (float, optional): A blend factor between equal anchor spacing (1) and spacing based on the distances between anchor points (0). Defaults to 0.5.

        Raises:
            ValueError: If the sum of `random_uniform_frac` and `random_normal_frac` exceeds 1, or if either is negative.

        Returns:
            Union[NDArray, Tensor]: The generated sequence of steps.

        Examples:

        .. code-block:: python

            import torch
            from sign_language_translator.utils import ArrayOps

            # you should plot the following arrays on a graph for better understanding
            anchors = [0, 1, -2, 0, 5, 2]

            # Basic linear interpolation with no randomness and equal anchor spacing
            steps = ArrayOps.steps(9, anchors, 0, 0, 0, 0, anchor_spacing_blend=0)
            # array([ 0.  ,  0.25, -1.5 , -0.75,  1.  ,  2.75,  4.5 ,  3.75,  2.  ])

            # Linear interpolation with no randomness and anchor spacing based on distances
            steps = ArrayOps.steps(9, anchors, 0, 0, 0, 0, anchor_spacing_blend=1)
            # array([ 0.   ,  0.625,  0.25 , -1.625, -1.   ,  0.625,  3.75 ,  3.875,  2.   ])

            # A blend of equal and distance-based anchor spacing with no randomness
            steps = ArrayOps.steps(9, torch.Tensor(anchors), 0, 0, 0, 0, anchor_spacing_blend=0.5)
            # Tensor([ 0.   ,  0.921, -0.655, -1.625, -0.167,  1.987,  4.231,  3.81 ,  2.   ])

            # Adding uniform randomness to the steps
            steps = ArrayOps.steps(9, anchors, 0.5, 0, 0, 0, anchor_spacing_blend=1)
            # array([ 0.   ,  0.25 , -1.   , -0.895,  0.214,  1.346,  3.75 ,  3.777,  2.   ])

            # Adding 2 concentration spots using gaussian randomness
            steps = ArrayOps.steps(9, anchors, 0, 0.5, 2, 0.1, anchor_spacing_blend=0)
            # array([ 0.   ,  0.99 ,  0.924, -1.5  ,  1.   ,  4.5  ,  4.872,  4.025,  2.   ])

            # Combining uniform and normal randomness
            steps = ArrayOps.steps(9, anchors, 0.2, 0.3, 2, 0.1, anchor_spacing_blend=0.5)
            # array([ 0.   ,  0.069, -1.333,  0.468,  1.538,  3.835,  4.897,  4.267,  2.   ])
        """

        if (
            random_uniform_frac < 0
            or random_normal_frac < 0
            or random_uniform_frac + random_normal_frac > 1
        ):
            raise ValueError(
                f"Invalid fractions: {random_uniform_frac=}, {random_normal_frac=}."
                "Provide non-negative values summing to <=1."
            )

        if anchor_spacing_blend < 0 or anchor_spacing_blend > 1:
            raise ValueError(f"Invalid {anchor_spacing_blend=}. Expected 0<=blend<=1")

        if not isinstance(anchors, (np.ndarray, Tensor)):
            anchors = np.array(anchors)

        n_uniform_steps = int(n_steps * random_uniform_frac)
        n_normal_steps = (
            int(n_steps * random_normal_frac / n_clusters) if n_clusters > 0 else 0
        )
        n_linear_steps = n_steps - n_uniform_steps - n_normal_steps * n_clusters

        # calculate the distance between anchors and normalize it and then blend with equal anchor spacing
        anchor_gap = np.cumsum(np.abs(np.diff(anchors, axis=-1, prepend=anchors[0])))
        anchor_gap = anchor_gap / anchor_gap[-1] * (1 - anchor_spacing_blend)
        anchor_gap = anchor_gap + np.linspace(0, 1, len(anchors)) * anchor_spacing_blend

        cluster_std = cluster_std or float(np.std(anchor_gap)) / 10 * max(n_clusters, 1)

        new_x = np.concatenate(
            [
                np.linspace(0, 1, n_linear_steps),
                ArrayOps.random_uniform((n_uniform_steps,), 0, 1),
            ]
            + [
                ArrayOps.random_normal(
                    (n_normal_steps,), np.random.rand(), cluster_std, 0, 1
                )
                for _ in range(n_clusters)
            ]
        )
        new_x = np.sort(new_x)

        return linear_interpolation(anchors, new_x, old_x=anchor_gap, dim=-1)


@overload
def linear_interpolation(
    array: Union[NDArray[np.number], Sequence],
    new_x: Union[Sequence[Union[int, float]], NDArray[np.number], Tensor],
    old_x: Union[Sequence[Union[int, float]], NDArray[np.number], Tensor, None] = None,
    dim: int = 0,
) -> NDArray: ...


@overload
def linear_interpolation(
    array: Tensor,
    new_x: Union[Sequence[Union[int, float]], NDArray[np.number], Tensor],
    old_x: Union[Sequence[Union[int, float]], NDArray[np.number], Tensor, None] = None,
    dim: int = 0,
) -> Tensor: ...


def linear_interpolation(
    array: Union[NDArray[np.number], Tensor, Sequence],
    new_x: Union[Sequence[Union[int, float]], NDArray[np.number], Tensor],
    old_x: Union[Sequence[Union[int, float]], NDArray[np.number], Tensor, None] = None,
    dim: int = 0,
) -> Union[NDArray, Tensor]:
    """
    Perform linear interpolation on a multidimensional array or tensor along a dimension.

    This function essentially connects all consecutive values in a multidimensional array with
    straight lines along a specified dimension, so that intermediate values can be calculated.
    It takes the input array, a set of new indexes or alternatively new & old coordinate values,
    and a dimension along which to perform interpolation.

    Parameters:
        array (NDArray[np.number] | Tensor | List): The input array or tensor to interpolate.
        new_x (Sequence[int | float] | NDArray[np.number] | Tensor): The new index values or coordinate values at which to calculate the intermediate values from `array`. Must be 1D. Order of values does not matter. if `old_x` is not provided, these values are relative to the index of the data in `array` i.e. [0, 1, 2, ...] and negative indexes are allowed. If `old_x` is provided, all `new_x` values must be within it's bounds.
        old_x (Sequence[int | float] | NDArray[np.number] | Tensor | None, optional): The old *coordinate* values corresponding to the data in `array` along the `dim`. Must be 1D and strictly sorted ascending. Can contain negative numbers. If None, method assumes it to be a linear sequence starting at 0 and growing with step +1 i.e. `[0, 1, 2, ...]` like the index of `array`.
        dim (int, optional): The dimension along which to perform interpolation. Default is 0.

    Returns:
        NDArray | Tensor: The result of linear interpolation along the specified dimension.

    Raises:
        ValueError: If `new_x` or `old_x` is not 1 dimensional.

    Examples:

        .. code-block:: python

            data = np.array([1, 2, 3, 5])
            new_indexes = np.array([1.5, 0.5, 2.5])
            interpolated_data = linear_interpolation(data, new_indexes)
            print(interpolated_data)
            # array([2.5, 1.5, 4. ])

            old_x = [0, 4, 4.5, 5]
            new_x = [0, 1, 2, 2.5, 3, 4, 5]
            interpolated_data = linear_interpolation(data, new_x, old_x=old_x)
            print(interpolated_data)
            # array([1.   , 1.25 , 1.5  , 1.625, 1.75 , 2.   , 5.   ])

            positional_embedding_table = torch.randn(100, 768)  # (max_seq_len, embedding_dim)
            intermediate_positions = torch.linspace(0, 99, 500)
            new_embedding_table = linear_interpolation(positional_embedding_table, intermediate_positions, dim=0)
            # new_embedding_table.shape -> (500, 768) # (new_max_seq_len, embedding_dim)

    Note:
        This function supports both NumPy arrays and PyTorch tensors as input and preserves gradient.
    """
    if not isinstance(array, (np.ndarray, Tensor)):
        array = np.array(array)

    indexes, dim = __validate_lin_interp_args(array, new_x, old_x, dim)

    # interpolate (Magic!)
    floored_indexes = ArrayOps.floor(indexes)
    ceiled_indexes = ArrayOps.ceil(indexes)
    fraction = (indexes - floored_indexes).reshape(
        [len(indexes) if d == dim else 1 for d in range(array.ndim)]
    )

    interpolated = ArrayOps.take(array, floored_indexes, dim) * (1 - fraction)
    interpolated = interpolated + ArrayOps.take(array, ceiled_indexes, dim) * fraction

    return interpolated


def __validate_lin_interp_args(array: Union[NDArray, Tensor], new_x, old_x, dim: int):
    if dim < -1 * array.ndim or dim >= array.ndim:
        raise ValueError(f"Invalid dim: {dim} use -{array.ndim} <= d <= {array.ndim-1}")
    if dim < 0:
        dim = dim + array.ndim

    if old_x is not None:
        if len(old_x) != array.shape[dim]:
            raise ValueError(f"Invalid old_x: ({len(old_x)=}) != ({array.shape[dim]=})")

        old_x, new_x = np.array(old_x), np.array(new_x)
        if (out_of_bounds := (new_x < old_x.min()) | (new_x > old_x.max())).any():
            raise ValueError(
                f"Invalid values in new_x: {new_x[out_of_bounds].tolist()}. "
                f"Must be in range {old_x.min()} <= x <= {old_x.max()}."
            )

        differences = old_x[:-1] - old_x[1:]
        if not ((is_descending := (differences > 0).all()) or (differences < 0).all()):
            raise ValueError("Provide sorted `old_x` (otherwise rearrange the array)")

        indexes = (
            np.interp(new_x, old_x[::-1], np.arange(len(old_x))[::-1])
            if is_descending
            else np.interp(new_x, old_x, np.arange(len(old_x)))
        )

    else:
        if not isinstance(new_x, (np.ndarray, Tensor)):
            new_x = np.array(new_x)
        if (
            out_of_bounds := (new_x < -array.shape[dim]) | (new_x >= array.shape[dim])
        ).any():
            raise ValueError(
                f"Invalid new indexes: {new_x[out_of_bounds.tolist()].tolist()}. "
                f"Must be in range {-array.shape[dim]} <= x <= {array.shape[dim]-1}."
            )
        indexes = new_x

    indexes = ArrayOps.cast(indexes, type(array))
    if indexes.ndim != 1:
        raise ValueError(f"Invalid indexes shape: {indexes.shape}. Must be 1D.")

    return indexes, dim


@overload
def adjust_vector_angle(
    vector_1: Union[NDArray, Sequence[float]],
    vector_2: Union[NDArray, Sequence[float]],
    scaling_factor: float,
    post_normalize: bool = False,
) -> Tuple[NDArray, NDArray]: ...


@overload
def adjust_vector_angle(
    vector_1: Tensor,
    vector_2: Tensor,
    scaling_factor: float,
    post_normalize: bool = False,
) -> Tuple[Tensor, Tensor]: ...


def adjust_vector_angle(
    vector_1: Union[NDArray, Tensor, Sequence[float]],
    vector_2: Union[NDArray, Tensor, Sequence[float]],
    scaling_factor: float,
    post_normalize: bool = False,
) -> Tuple[Union[NDArray, Tensor], Union[NDArray, Tensor]]:
    """Move a pair of vectors away or towards each other in the same plane.

    Converge or Diverge a pair of vectors by increasing or decreasing their distance from each other.
    The norm or the length of the vectors is preserved.

    Args:
        vector_1 (NDArray | Tensor): A 1D array of size n representing a word in an n dimensional vector space.
        vector_2 (NDArray | Tensor): A 1D array of size n representing another word in an n dimensional vector space.
        scaling_factor (float): The scaling factor by which the vector difference should be enhanced or diminished. The fraction of distance between the vectors where new vector should land. (sf > 1 diverges the two vectors. sf = 1 leaves the two vectors unchanged. 0.5 < sf < 1 converges the two vectors. sf = 0.5 makes the two vectors equal to their mean. sf = 0 swaps the two vectors. sf < 0.5 move the vectors away from their mean but in opposite direction.)
        post_normalize (bool, optional): Make the magnitude of both output vectors equal to 1 after they have been rotated. Defaults to False.

    Returns:
        Tuple[NDArray | Tensor, NDArray | Tensor]: moved vectors.

    Notes:

    .. code-block:: python

        # sf > 1 diverges the two vectors
        # new_v1 = v2 + 2.00 * (v1 - v2) = 2 * v1 - v2     # more v1, less v2.
        # new_v2 = v1 - 2.00 * (v1 - v2) = 2 * v2 - v1     # more v2, less v1.

        # sf = 1 leaves the two vectors unchanged
        # new_v1 = v2 + 1.00 * (v1 - v2) = v1
        # new_v2 = v1 - 1.00 * (v1 - v2) = v2

        # 0.5 < sf < 1 converges the two vectors
        # new_v1 = v2 + 0.75 * (v1 - v2) = 0.75 * v1 + 0.25 * v2    # weighted average
        # new_v1 = v1 - 0.75 * (v1 - v2) = 0.75 * v2 + 0.25 * v1    # weighted average

        # sf = 0.5 makes the two vectors equal
        # new_v1 = v2 + 0.50 * (v1 - v2) = 0.5 * v1 + 0.5 * v2   # mean
        # new_v1 = v1 - 0.50 * (v1 - v2) = 0.5 * v2 + 0.5 * v1   # mean

        # sf = 0. swaps the two vectors
        # new_v1 = v2 + 0.00 * (v1 - v2) = v2
        # new_v2 = v1 + 0.00 * (v1 - v2) = v1

        # sf < 0.5 move the vectors away from their mean but in opposite direction
        # new_v1 = v2 + (-1) * (v1 - v2) = 2 * v2 - v1    # more v2, less v1.
        # new_v2 = v1 - (-1) * (v1 - v2) = 2 * v1 - v2    # more v1, less v2.
    """

    # TODO: make it work for batches of vectors or nd-arrays
    # TODO: Handle NaNs

    if not isinstance(vector_1, (np.ndarray, Tensor)):
        vector_1 = np.array(vector_1)
    if not isinstance(vector_2, (np.ndarray, Tensor)):
        vector_2 = np.array(vector_2)

    v1_norm = ArrayOps.norm(vector_1, dim=None)
    v2_norm = ArrayOps.norm(vector_2, dim=None)

    # make the magnitude of both vectors = 1
    vector_1 = vector_1 / (v1_norm or 1)
    vector_2 = vector_2 / (v2_norm or 1)

    # figure out the dimension of divergence
    v1_minus_v2 = vector_1 - vector_2

    # move each vector away or towards the other by equal amount
    new_v1 = vector_2 + scaling_factor * v1_minus_v2
    new_v2 = vector_1 - scaling_factor * v1_minus_v2

    # make the magnitude of the new vectors = 1
    new_v1 = new_v1 / (ArrayOps.norm(new_v1) or 1)
    new_v2 = new_v2 / (ArrayOps.norm(new_v2) or 1)

    if not post_normalize:
        # restore original magnitudes
        new_v1 = new_v1 * v1_norm
        new_v2 = new_v2 * v2_norm

    return new_v1, new_v2


@overload
def align_vectors(
    source_matrix: NDArray, target_matrix: NDArray, pre_normalize: bool = True
) -> NDArray: ...


@overload
def align_vectors(
    source_matrix: Tensor, target_matrix: Tensor, pre_normalize: bool = True
) -> Tensor: ...


def align_vectors(
    source_matrix: Union[NDArray, Tensor],
    target_matrix: Union[NDArray, Tensor],
    pre_normalize: bool = True,
) -> Union[NDArray, Tensor]:
    """
    Align the source matrix to the target matrix using the orthogonal transformation.

    Args:
        source_matrix (NDArray | Tensor): A 2D array of shape (dictionary_length, embedding_dimension) containing word vectors from source model (or language).
        target_matrix (NDArray | Tensor): A 2D array of shape (dictionary_length, embedding_dimension) containing word vectors from target model (or language).
        normalize_vectors (bool, optional): Whether to normalize the training vectors before SVD. Defaults to True.

    Returns:
        NDArray | Tensor: An orthogonal transformation which aligns the source language to the target language.

    Note:
        This function supports both NumPy arrays and PyTorch tensors as input.
        (Based on: https://github.com/babylonhealth/fastText_multilingual)
    """

    if not isinstance(source_matrix, (np.ndarray, Tensor)):
        source_matrix = np.array(source_matrix)
    if not isinstance(target_matrix, (np.ndarray, Tensor)):
        target_matrix = np.array(target_matrix)

    # optionally normalize the training vectors
    if pre_normalize:
        src_norm = ArrayOps.norm(source_matrix, dim=1, keepdim=True)
        src_norm[src_norm == 0] = 1
        source_matrix = source_matrix / src_norm

        tgt_norm = ArrayOps.norm(target_matrix, dim=1, keepdim=True)
        tgt_norm[tgt_norm == 0] = 1
        target_matrix = target_matrix / tgt_norm

    # perform the SVD
    product = source_matrix.T @ target_matrix
    U, _, V = ArrayOps.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return U @ V
