"""This module contains various helper functions and utilities for configuration purposes.

Module Structure:
- prepare_filename_url_dict(yaml_file_path: str) -> Dict[str, str]:
    reads urls for package resources and prepares filename to url map.
- get_package_version() -> str: Retrieves the version of the package.
"""

import json
from typing import Dict

from pkg_resources import get_distribution


def prepare_filename_url_dict(file_path: str, encoding="utf-8") -> Dict[str, str]:
    """
    Prepares a dictionary mapping filenames to their corresponding URLs.

    Args:
        yaml_file_path (str): The path to the YAML file containing resources information.

    Returns:
        Dict[str, str]: A dictionary mapping filenames to their corresponding URLs.
    """

    # TODO: ? Download all and call without arg ?
    # move download() to a separate file to end circular import

    with open(file_path, "r", encoding=encoding) as f:
        data = json.load(f)

    filename_url_dict = {}

    # Find the files and update filename url dict
    def extract_urls(data: Dict, results: Dict):
        for key in data:
            if key == "file_to_url":
                results.update(data[key])
            elif key == "files":
                for item in data[key]:
                    results[item["name"]] = item["url"]
            elif isinstance(data[key], dict):
                extract_urls(data[key], results)
            elif isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict):
                        extract_urls(item, results)

    extract_urls(data, filename_url_dict)

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
