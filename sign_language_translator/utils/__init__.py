"""This module provides utility functions for the sign language translator package.

Functions:
- download: A function for downloading files from urls.
- tree: A function for printing a directory tree.
- sample_one_index: Select an index based on the given probability distribution.
- search_in_values_to_retrieve_key: search inside every dict value and return the key on match.
- in_jupyter_notebook: Checks if the code is running in a Jupyter notebook.
- linear_interpolation: figure out intermediate values inside an array.
- threaded_map: Multi-threaded mapping of a function to an iterable.

Classes:
- ArrayOps: A class for array operations agnostic to numpy.ndarray and torch.Tensor.
"""

from sign_language_translator.utils.arrays import ArrayOps, linear_interpolation
from sign_language_translator.utils.download import download
from sign_language_translator.utils.parallel import threaded_map
from sign_language_translator.utils.tree import tree
from sign_language_translator.utils.utils import (
    in_jupyter_notebook,
    sample_one_index,
    search_in_values_to_retrieve_key,
)

__all__ = [
    # functions
    "tree",
    "download",
    "search_in_values_to_retrieve_key",
    "sample_one_index",
    "in_jupyter_notebook",
    "linear_interpolation",
    "threaded_map",
    # classes
    "ArrayOps",
]
