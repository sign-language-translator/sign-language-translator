"""defines settings constants for global context
"""

from os.path import join, dirname


class Settings:
    """defines settings constants for global context"""

    DATASET_ROOT_DIRECTORY: str = join(
        dirname(dirname(__file__)), "sign-language-datasets"
    )

    FILENAME_SEPARATOR = "_"
    FILENAME_CONNECTOR = "-"
