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
from time import time
from warnings import warn

import requests
from tqdm.auto import tqdm

__all__ = [
    "download",
]


def download(
    file_path: str,
    url: str,
    overwrite=False,
    progress_bar=False,
    timeout: float = 20.0,
    leave=True,
    chunk_size=65536,
    status_callback=None,
) -> bool:
    """
    Downloads a file from the specified URL and saves it to the given file path.

    Args:
        file_path (str): The path where the downloaded file will be saved.
        url (str): The URL of the file to be downloaded.
        overwrite (bool, optional): If False, skips downloading if the file already exists. Defaults to False.
        progress_bar (bool, optional): If True, displays a progress bar during the download. Defaults to False.
        timeout (int, optional): The maximum number of seconds to wait for a server response. Defaults to 20.0.
        leave (bool, optional): Wether to leave the progress bar behind after the download. Defaults to True.
        chunk_size (int, optional): The number of bytes to fetch in each step. Defaults to 65536.

    Returns:
        bool: True if the file is downloaded successfully, False otherwise.

    Raises:
        FileExistsError: if overwrite is False and the destination path already contains a file.
    """

    # TODO: resume failed download (headers = {'Range': f'bytes={resume_byte_pos}-'})
    # TODO: separate threads for download and writing (implement queue(s))

    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"There is already a file at {file_path}")

    try:
        response = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as fp:
            stream = response.iter_content(chunk_size=chunk_size)

            total_bytes = int(response.headers.get("content-length", 0))
            if progress_bar:
                stream = tqdm(
                    stream,
                    total=(total_bytes // chunk_size) + 1,
                    desc=f"Downloading {os.path.split(file_path)[-1]}",
                    leave=leave,
                    unit="chunk",
                )

            start_time = time()
            speed = 0
            saved = 0
            # download & write
            for chunk in stream:
                if chunk:
                    fp.write(chunk)

                # Calculate download speed (exponential moving average)
                speed += len(chunk) / (1024**2) / (time() - start_time)
                speed /= 2
                start_time = time()

                # Update progress bar
                if isinstance(stream, tqdm):
                    stream.set_postfix_str(f"{speed:.3f}MB/s")

                if status_callback:
                    saved += len(chunk)
                    status = {
                        "file": f"{saved / total_bytes:.1%}" if total_bytes else "?%",
                        "down": f"{speed:.3f}MB/s",
                    }
                    status_callback(status)

        # TODO: if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:

        return True

    except requests.exceptions.RequestException as e:
        warn(f"Download of '{file_path}' failed: ({str(e)})")
        return False
