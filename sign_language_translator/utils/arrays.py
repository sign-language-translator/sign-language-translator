from typing import Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch import long as torch_long

__all__ = [
    "ArrayOps",
    "linear_interpolation",
    "adjust_vector_angle",
    "align_vectors",
]


class ArrayOps:
    @staticmethod
    def floor(array: Union[NDArray, Tensor]) -> Union[NDArray, Tensor]:
        if isinstance(array, np.ndarray):
            return np.floor(array)
        elif isinstance(array, Tensor):
            return array.floor()
        else:
            raise TypeError(f"Invalid type for flooring: {type(array)}")

    @staticmethod
    def ceil(array: Union[NDArray, Tensor]) -> Union[NDArray, Tensor]:
        if isinstance(array, np.ndarray):
            return np.ceil(array)
        elif isinstance(array, Tensor):
            return array.ceil()
        else:
            raise TypeError(f"Invalid type for ceiling: {type(array)}")

    @staticmethod
    def take(
        array: Union[NDArray, Tensor], index: Union[NDArray, Tensor, List], dim: int = 0
    ) -> Union[NDArray, Tensor]:
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
        x: Union[NDArray, Tensor, List, Iterable],
        data_type: Type[Union[np.ndarray, Tensor]],
    ) -> Union[NDArray, Tensor]:
        if data_type == np.ndarray:
            return np.array(x)
        elif data_type == Tensor:
            return Tensor(x)
        else:
            raise TypeError(f"Invalid type for array casting: {data_type}")

    @staticmethod
    def norm(
        x: Union[NDArray, Tensor, List, Iterable],
        dim: Optional[int] = None,
        keepdim=False,
    ) -> Union[NDArray, Tensor]:
        """
        Compute the norm of a given array or tensor along a specified dimension.

        Args:
            x (Union[NDArray, Tensor, List, Iterable]): The input array or tensor.
            dim (Optional[int]): The dimension along which to compute the norm. If None, the norm is computed over the entire array or tensor. Default is None.
            keepdim (bool): Whether to keep the dimension of the input array or tensor after computing the norm. Default is False.

        Returns:
            Union[NDArray, Tensor]: The norm of the input array or tensor.

        Raises:
            TypeError: If the input type is not supported.
        """

        if isinstance(x, np.ndarray):
            return np.linalg.norm(x, axis=dim, keepdims=keepdim)  # type: ignore
        elif isinstance(x, Tensor):
            return x.norm(dim=dim, keepdim=keepdim)
        else:
            raise TypeError(f"Invalid type for norm: {type(x)}")

    @staticmethod
    def svd(
        x: Union[NDArray, Tensor, List, Iterable]
    ) -> Tuple[Union[NDArray, Tensor], Union[NDArray, Tensor], Union[NDArray, Tensor]]:
        """
        Compute the singular value decomposition of a given array or tensor.

        Args:
            x (Union[NDArray, Tensor, List, Iterable]): The input array or tensor.

        Returns:
            Tuple[Union[NDArray, Tensor], Union[NDArray, Tensor], Union[NDArray, Tensor]]: The (Rotation, coordinate scaling, reflection) matrices of the input array or tensor.

        Raises:
            TypeError: If the input type is not supported.
        """

        if not isinstance(x, (List, Iterable)):
            x = np.array(x)

        if isinstance(x, np.ndarray):
            return np.linalg.svd(x)
        if isinstance(x, Tensor):
            U, S, V = x.svd()
            return U, S, V.T

        raise TypeError(f"Invalid type for svd: {type(x)}")

    @staticmethod
    def top_k(
        x: Union[NDArray, Tensor, List, Iterable], k: int, dim: int = -1, largest=True
    ) -> Tuple[Union[NDArray, Tensor], Union[NDArray, Tensor]]:
        """
        Compute the top k values and their indices along a specified dimension of a given array or tensor.

        Args:
            x (Union[NDArray, Tensor, List, Iterable]): The input array or tensor.
            k (int): The number of top values to return.
            dim (int, optional): The dimension along which to compute the top k values. Default is -1.
            largest (bool, optional): Whether to return the largest or smallest k values. Default is True.

        Returns:
            Tuple[Union[NDArray, Tensor], Union[NDArray, Tensor]]: The top k values and their indices along the specified dimension.

        Raises:
            TypeError: If the input type is not supported.
        """

        if isinstance(x, np.ndarray):
            indices = (
                np.argpartition(x, -k, axis=dim)[-k:]
                if largest
                else np.argpartition(x, k, axis=dim)[:k]
            )
            return x.take(indices, axis=dim), indices
        if isinstance(x, Tensor):
            values, indices = x.topk(k, dim=dim, largest=largest, sorted=True)
            return values, indices

        raise TypeError(f"Invalid type for topk: {type(x)}")


def linear_interpolation(
    array: Union[NDArray[np.float64], Tensor, List],
    new_indexes: Optional[Sequence[Union[int, float]]] = None,
    old_x: Optional[Sequence[Union[int, float]]] = None,
    new_x: Optional[Sequence[Union[int, float]]] = None,
    dim: int = 0,
) -> Union[NDArray, Tensor]:
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

    return new_v1, new_v2


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

    """
    # https://github.com/babylonhealth/fastText_multilingual

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
