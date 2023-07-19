"""This module contains various helper functions and utilities for configuration purposes.

Module Structure:
- prepare_filename_url_dict(yaml_file_path: str) -> Dict[str, str]:
    reads urls for package resources and prepares filename to url map.
- get_package_version() -> str: Retrieves the version of the package.
"""

from typing import Dict

from pkg_resources import get_distribution
from yaml import safe_load


def prepare_filename_url_dict(yaml_file_path: str) -> Dict[str, str]:
    """
    Prepares a dictionary mapping filenames to their corresponding URLs.

    Args:
        yaml_file_path (str): The path to the YAML file containing resources information.

    Returns:
        Dict[str, str]: A dictionary mapping filenames to their corresponding URLs.
    """

    # TODO: ? Download all and call without arg ?
    # move download() to a separate file to end circular import

    with open(yaml_file_path, "r", encoding="utf-8") as f:
        data = safe_load(f)

    filename_url_dict = {}

    # Find the files and update filename url dict
    for resource_key in ["datasets", "models", "others"]:
        for resource in data[resource_key]:
            for file in resource["files"]:
                filename = file["name"]
                url = file["url"]
                filename_url_dict[filename] = url

    # TODO: download extra_urls.yaml
    # when lookup videos in filename_url_dict
    # call this function and update Settings.FILE_TO_URLS

    return filename_url_dict


def get_package_version():
    """
    Retrieves the version of the 'sign-language-translator' package.

    Returns:
        str: The version of the package.
    """

    return get_distribution("sign-language-translator").version
