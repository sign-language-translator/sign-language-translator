from random import choices
from typing import Any, Dict, Iterable, List, Set

import numpy as np
from torch import Tensor, long as torch_long
from numpy.typing import NDArray


def search_in_values_to_retrieve_key(
    code_name: str, class_to_codes: Dict[Any, Set[str]]
):
    # verify there is no repetition/reuse in language codes
    all_codes = [code for codes in class_to_codes.values() for code in codes]
    assert len(all_codes) == len(set(all_codes)), "code reused for multiple keys"

    for key, codes in class_to_codes.items():
        if code_name.lower() in codes:
            return key

    return None


def sample_one_index(weights: List[float], temperature: float = 1.0) -> int:
    """Select an item based on the given probability distribution.
    Returns the index of the selected item sampled from weighted random distribution.

    Args:
        weights (List[float]): the relative weights corresponding to each index.
        temperature (float): The temperature value for controlling the sampling behavior.
            High temperature means sampling probabilities are more uniform (says random things).
            Low temperature means that sampling probabilities are higher for bigger weights.
            Defaults to 1.0.

    Returns:
        int: The index of the chosen item.
    """

    return choices(
        range(len(weights)),
        weights=[w / temperature for w in weights],
        k=1,
    )[0]


def in_jupyter_notebook():
    """
    Checks if the code is running in a Jupyter notebook.

    Returns:
        bool: True if running in a Jupyter notebook, False otherwise.
    """

    try:
        from IPython import get_ipython  # type: ignore

        return "IPKernelApp" in get_ipython().config  # type: ignore
    except:  # pylint: disable = bare-except
        return False


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
        x: NDArray | Tensor | List | Iterable, data_type
    ) -> NDArray | Tensor:
        if data_type == np.ndarray:
            return np.array(x)
        elif data_type == Tensor:
            return Tensor(x)
        else:
            raise TypeError(f"Invalid type for array casting: {data_type(x)}")


__all__ = [
    "search_in_values_to_retrieve_key",
    "sample_one_index",
    "in_jupyter_notebook",
    "ArrayOps",
]
