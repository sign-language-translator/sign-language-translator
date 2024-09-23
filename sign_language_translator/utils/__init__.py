"""
Utils
=====

This module provides utility functions for the sign language translator package.

Functions
---------

- download: A function for downloading files from urls.
- tree: A function for printing a directory tree.
- sample_one_index: Select an index based on the given probability distribution.
- search_in_values_to_retrieve_key: search inside every dict value and return the key on match.
- in_jupyter_notebook: Checks if the code is running in a Jupyter notebook.
- is_regex: Checks if the given string is a regex or a regular string.
- linear_interpolation: figure out intermediate values inside an array.
- threaded_map: Multi-threaded mapping of a function to an iterable.
- extract_recursive: Recursively extracts values associated with a specified key from a nested dictionary.

Classes
-------

- ArrayOps: A class for array operations agnostic to numpy.ndarray and torch.Tensor.
- Archive: A utility class for making, viewing and extracting archive files such as .zip files.
- PrintableEnumMeta: A metaclass for making enum classes printable with the class members.
- ProgressStatusCallback: A class for updating a tqdm progress bar inside a function.
"""

from sign_language_translator.utils.archive import Archive
from sign_language_translator.utils.arrays import (
    ArrayOps,
    adjust_vector_angle,
    align_vectors,
    linear_interpolation,
)
from sign_language_translator.utils.download import download
from sign_language_translator.utils.parallel import threaded_map
from sign_language_translator.utils.tree import tree
from sign_language_translator.utils.utils import (
    PrintableEnumMeta,
    ProgressStatusCallback,
    extract_recursive,
    in_jupyter_notebook,
    is_internet_available,
    is_regex,
    sample_one_index,
    search_in_values_to_retrieve_key,
    validate_path_exists,
)

__all__ = [
    # functions
    "tree",
    "download",
    "search_in_values_to_retrieve_key",
    "sample_one_index",
    "in_jupyter_notebook",
    "is_regex",
    "linear_interpolation",
    "threaded_map",
    "extract_recursive",
    "adjust_vector_angle",
    "align_vectors",
    "validate_path_exists",
    "is_internet_available",
    # classes
    "Archive",
    "ArrayOps",
    "PrintableEnumMeta",
    "ProgressStatusCallback",
]
