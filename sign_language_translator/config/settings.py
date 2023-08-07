"""defines settings constants for global context

Module Structure:
- Settings (class): contains config constants as class variables.
- set_resources_dir(path: str) -> None: Sets the resources(dataset/models) directory path in Settings.
"""

from os.path import dirname, isdir, join

from sign_language_translator.config.helpers import prepare_filename_url_dict


class Settings:
    """Class containing settings and configuration parameters
    for the sign language translator library."""

    RESOURCES_ROOT_DIRECTORY: str = join(
        dirname(dirname(__file__)), "sign-language-resources"
    )
    """The root directory path where the sign language datasets & models are stored."""

    FILENAME_SEPARATOR = "_"
    """The separator used in dataset filenames to separate different attributes."""

    FILENAME_CONNECTOR = "-"
    """The connector used in filenames to join parts of same attribute."""

    FILE_TO_URLS = prepare_filename_url_dict(
        join(dirname(__file__), "urls.yaml")
    )
    """A dictionary mapping filenames to their corresponding URLs, based on the package version."""

    AUTO_DOWNLOAD = True
    """A flag indicating whether automatic downloading of missing dataset files is enabled."""


def set_resources_dir(path: str) -> None:
    """Set the SLT resources directory path.
    Helpful when sign-language-datasets from the cloud is mounted on disk.

    Args:
        path (str): The path to the resources/dataset/models directory.

    Raises:
        AssertionError: If the provided path is not a directory.
    """

    assert isdir(path), f"the provided path is not a directory. Path: {path}"

    Settings.RESOURCES_ROOT_DIRECTORY = path
    # ? trigger an event


__all__ = [
    "Settings",
    "set_resources_dir",
]
