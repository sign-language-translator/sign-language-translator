"""This module contains various helper functions and utilities for configuration purposes.

Module Structure:
- set_dataset_dir(path: str) -> None: Sets the dataset directory path.
"""

import os
from .settings import Settings


def set_dataset_dir(path: str) -> None:
    """Set the sign-language-datasets directory path.

    Args:
        path (str): The path to the dataset directory.

    Raises:
        AssertionError: If the provided path is not a directory.
    """

    assert os.path.isdir(path), "the provided path is not a directory"

    Settings.DATASET_ROOT_DIRECTORY = path
    # ? trigger an event
