"""defines settings constants for global context

Module Structure:
- Settings (class): contains config constants as class variables.
"""

__all__ = [
    "Settings",
]


class Settings:
    """Class containing settings and configuration parameters
    for the sign language translator library."""

    FILENAME_SEPARATOR = "_"
    """The separator used in dataset filenames to separate different attributes."""

    FILENAME_CONNECTOR = "-"
    """The connector used in filenames to join parts of same attribute."""

    AUTO_DOWNLOAD = True
    """A flag indicating whether automatic downloading of missing dataset files is enabled."""

    SHOW_DOWNLOAD_PROGRESS = True
    """A flag indicating whether the download progress should be shown by default."""
