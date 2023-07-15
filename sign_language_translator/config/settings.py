"""defines settings constants for global context

Module Structure:
- Settings (class): contains config constants as class variables.
- set_dataset_dir(path: str) -> None: Sets the dataset directory path in Settings.
"""

from os.path import dirname, isdir, join

from sign_language_translator.config.helpers import (
    get_package_version,
    prepare_filename_url_dict,
)


class Settings:
    """Class containing settings and configuration parameters
    for the sign language translator library."""

    DATASET_ROOT_DIRECTORY: str = join(
        dirname(dirname(__file__)), "sign-language-datasets"
    )
    """The root directory path where the sign language datasets are stored."""

    FILENAME_SEPARATOR = "_"
    """The separator used in dataset filenames to separate different attributes."""

    FILENAME_CONNECTOR = "-"
    """The connector used in filenames to join parts of same attribute."""

    FILE_TO_URLS = prepare_filename_url_dict(
        get_package_version(), join(dirname(__file__), "urls.yaml")
    )
    """A dictionary mapping filenames to their corresponding URLs, based on the package version."""

    AUTO_DOWNLOAD = True
    """A flag indicating whether automatic downloading of missing dataset files is enabled."""


def set_dataset_dir(path: str) -> None:
    """Set the sign-language-datasets directory path.

    Args:
        path (str): The path to the dataset directory.

    Raises:
        AssertionError: If the provided path is not a directory.
    """

    assert isdir(path), f"the provided path is not a directory. Path: {path}"

    Settings.DATASET_ROOT_DIRECTORY = path
    # ? trigger an event


__all__ = [
    "Settings",
    "set_dataset_dir",
]
