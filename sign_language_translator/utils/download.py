"""
Module for downloading files from URLs and managing package resources.

This module provides functions for downloading files from specified URLs and saving them to the given file paths.
It also includes a function for downloading package resources matching a specified filename regex and saving them
to the appropriate file paths.

Functions:
- download(file_path, url, overwrite=False, progress_bar=False, timeout=20.0, chunk_size=65536):
    Downloads a file from the specified URL and saves it to the given file path.
- download_package_resource(filename_regex, overwrite=False, progress_bar=False, timeout=20.0):
    Downloads package resources matching the given filename regex and saves them to the appropriate file paths.
"""

import os
import re
from time import time

import requests
from tqdm.auto import tqdm

from sign_language_translator.config.settings import Settings


def download(
    file_path: str,
    url: str,
    overwrite=False,
    progress_bar=False,
    timeout: float = 20.0,
    leave=True,
    chunk_size=65536,
) -> bool:
    """
    Downloads a file from the specified URL and saves it to the given file path.

    Args:
        file_path (str): The path where the downloaded file will be saved.
        url (str): The URL of the file to be downloaded.
        overwrite (bool, optional): If False, skips downloading if the file already exists. Defaults to False.
        progress_bar (bool, optional): If True, displays a progress bar during the download. Defaults to False.
        timeout (int, optional): The maximum number of seconds to wait for a server response. Defaults to 20.0.

    Returns:
        bool: True if the file is downloaded successfully, False otherwise.

    Raises:
        FileExistsError: if overwrite is False and the destination path already contains a file.
    """

    # TODO: resume failed download

    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"There is already a file at {file_path}")

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as fp:
            stream = response.iter_content(chunk_size=chunk_size)

            if progress_bar:
                total_bytes = int(
                    response.headers.get("content-length", 0)
                )  # for some reason, some bars finish too early
                stream = tqdm(
                    stream,
                    total=(total_bytes // chunk_size) + 1,
                    desc=f"Downloading {os.path.split(file_path)[-1]}",
                    leave=leave,
                    unit="chunk",
                )

            start_time = time()
            speed = 0
            for chunk in stream:
                if chunk:
                    fp.write(chunk)

                # display download speed
                if progress_bar:
                    speed = (len(chunk)/(1024**2)/(time() - start_time) + speed)/2
                    stream.set_postfix_str(  # type:ignore
                        f"{speed:.3f}MB/s"
                    )
                    start_time = time()

        return True

    except requests.exceptions.RequestException:
        return False


def download_resource(
    filename_regex: str,
    overwrite=False,
    progress_bar=False,
    timeout: float = 20.0,
    leave=True,
    chunk_size=65536,
) -> bool:
    """
    Downloads package resources matching the given filename regex and saves them to the appropriate file paths.

    Args:
        filename_regex (str): Regular expression pattern to match the desired filenames.
        overwrite (bool, optional): If False, skips downloading if the resource file already exists. Defaults to False.
        progress_bar (bool, optional): If True, displays a progress bar during the download. Defaults to False.
        timeout (float, optional): The maximum number of seconds to wait for a server response. Defaults to 20.0.

    Returns:
        bool: True if all resources are downloaded successfully or already exist, False otherwise.
    """

    matching_filenames_to_url = {
        key: val
        for key, val in Settings.FILE_TO_URLS.items()
        if re.match(filename_regex, key)
    }
    statuses = []
    for filename, url in matching_filenames_to_url.items():
        # Make sure that the file/directory exists
        file_path = os.path.join(Settings.RESOURCES_ROOT_DIRECTORY, filename)
        if os.path.exists(file_path) and not overwrite:
            continue
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Download the file from the URL
        status = download(
            file_path,
            url,
            progress_bar=progress_bar,
            timeout=timeout,
            overwrite=overwrite,
            leave=leave,
            chunk_size=chunk_size,
        )
        statuses.append(status)

    return all(statuses or [False])


__all__ = [
    "download",
    "download_resource",
]
