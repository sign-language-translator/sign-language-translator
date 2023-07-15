"""This module contains various helper functions and utilities for configuration purposes.

Module Structure:
- prepare_filename_url_dict(package_version: str, yaml_file_path: str) -> Dict[str, str]:
    reads urls for matching dataset version and prepares filename to url map.
- get_package_version() -> str: Retrieves the version of the package.
"""

from typing import Dict

import yaml
from pkg_resources import get_distribution


def prepare_filename_url_dict(
    package_version: str, yaml_file_path: str
) -> Dict[str, str]:
    """
    Prepares a dictionary mapping filenames to their corresponding URLs based on the package version.

    Args:
        package_version (str): The version of the package.
        yaml_file_path (str): The path to the YAML file containing dataset information.

    Returns:
        Dict[str, str]: A dictionary mapping filenames to their corresponding URLs.

    Raises:
        ValueError: If no dataset version is found for the given package version.
    """

    with open(yaml_file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    filename_url_dict = {}
    dataset_version = None

    # Find the dataset version corresponding to the package version
    for version_mapping in data["package_versions"]:
        if version_mapping["version"] == package_version:
            dataset_version = version_mapping["dataset_version"]
            break

    if dataset_version is None:
        raise ValueError(
            f"No dataset version found for package version '{package_version}'"
        )

    # Find the files for the dataset version and build the filename to URL dictionary
    for dataset in data["datasets"]:
        if dataset["version"] == dataset_version:
            for file in dataset["files"]:
                filename = file["name"]
                url = file["url"]
                filename_url_dict[filename] = url
            break

    return filename_url_dict


def get_package_version():
    """
    Retrieves the version of the 'sign-language-translator' package.

    Returns:
        str: The version of the package.
    """

    return get_distribution("sign-language-translator").version
