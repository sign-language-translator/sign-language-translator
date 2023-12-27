"""This module contains various helper functions and utilities for configuration purposes.

Module Structure:
- get_package_version() -> str: Retrieves the version of the package.
- read_urls(file_path: str) -> Dict[str, str]:
    reads urls for package resources and prepares filename to url map.
"""

import json
from importlib.metadata import version
from typing import Dict

from sign_language_translator.utils.utils import extract_recursive


def get_package_version():
    """
    Retrieves the version of the 'sign-language-translator' package.

    Returns:
        str: The version of the package.
    """

    return version("sign_language_translator")


def read_urls(file_path: str, encoding="utf-8") -> Dict[str, str]:
    # refactor: extract key recursively
    """
    Prepares a dictionary mapping filenames to their corresponding URLs.

    Args:
        file_path (str): The path to the JSON file containing resources information.

    Returns:
        Dict[str, str]: A dictionary mapping filenames to their corresponding URLs.
    """

    with open(file_path, "r", encoding=encoding) as f:
        data = json.load(f)

    filename_url_dict = {
        file: url
        for file_to_url in extract_recursive(data, "file_to_url")
        for file, url in file_to_url.items()
    }

    return filename_url_dict
