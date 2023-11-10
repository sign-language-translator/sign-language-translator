"""Module for managing assets required by to the sign language translator,
such as datasets and models."""

import json
import re
from datetime import datetime
from os import makedirs, remove
from os.path import abspath, dirname, exists, isdir, isfile, join, sep
from typing import Dict, List, Tuple

from sign_language_translator.config.settings import Settings
from sign_language_translator.config.utils import read_urls
from sign_language_translator.utils.download import download


class Assets:
    """
    Class for managing assets related to the sign language translator.

    Attributes:
        ROOT_DIR (str): The root directory path where the sign language datasets & models are stored.
        FILE_TO_URL (Dict[str, str]): A dictionary mapping asset filenames to their corresponding URLs.
        primary_urls_file (str): The name of the first URLs file that is loaded by default and must contain links to other url files.
        urls_file_dir (str): The directory path where the URLs files are stored.
        asset_regex_to_urls_file (List[Tuple[str, str]]): A list of tuples mapping regular expressions matching asset names to URLs files.
        _checksum_filename (str): The filename for the checksum file.
        _loaded_url_files (set): A set containing names of loaded URLs files.

    Methods:
        set_root_dir(path: str) -> None:
            Set the SLT resources directory path.

        get_id(filename_or_regex: str) -> List[str]:
            Get the IDs matching the given filename or regex.

        get_url(filename_or_regex: str) -> List[str]:
            Get the URLs for assets matching the given filename or regex.

        get_path(filename_or_regex: str) -> List[str]:
            Get the file paths for assets matching the given filename or regex.

        download(filename_or_regex: str, overwrite=False, progress_bar: bool | None = None, timeout: float = 20.0, leave=True, chunk_size=65536) -> bool:
            Download assets matching the given filename regex and save them to the appropriate file paths inside the assets root directory.

        delete_out_of_date_asset(asset_id: str, url: str, checksum: Dict[str, Dict[str, str]]) -> None:
            Delete assets that are out of date based on the checksum (change in URL).

        reload() -> None:
            Clear cache and read the URL files again.

        load_urls(filename: str) -> None:
            Load URLs from the specified file.
    """

    # =============== #
    #    CONFIGURE    #
    # =============== #

    ROOT_DIR: str = join(dirname(dirname(abspath(__file__))), "assets")
    """The root directory path where the sign language datasets & models are stored. Defaults is 'install_directory/assets'."""

    primary_urls_file: str = "urls.json"
    """The name of the first URLs file that is loaded by default and must contain links to other url files.
    Defaults is 'urls.json'. Note: all url filenames must end with 'urls.json'."""

    urls_file_dir: str = dirname(abspath(__file__))
    """The directory path where the URLs files are stored. Defaults is 'install_directory/sign_language_translator/config'."""

    FILE_TO_URL: Dict[str, str] = read_urls(join(urls_file_dir, primary_urls_file))
    """A dictionary mapping asset filenames to their corresponding URLs."""

    asset_regex_to_urls_file: List[Tuple[str, str]] = [
        # regex_for_assets, urls_file
        # (r"videos/pk-.*\.mp4, "pk-reference-clips-urls.json")
        (r".*", "extra_urls.json"),
    ]
    """A list of tuples that map regular expressions matching asset names to URLs files containing URLs to that asset group.
    The regex are tried sequentially so make sure list order is right."""

    _checksum_filename = ".checksum.json"
    _loaded_url_files = set()

    @classmethod
    def set_root_dir(cls, path: str) -> None:
        """Set the SLT resources directory path.
        Helpful when using custom datasets, or when data must be stored outside install directory.
        (e.g hosted on cloud and mounted on disk).
        By default, resources are downloaded to the 'install_directory/assets'.

        Args:
            path (str): The path to the resources/dataset/models directory.

        Raises:
            ValueError: If the provided path is not a directory.
        """

        path = abspath(path)
        if exists(path) and not isdir(path):
            raise ValueError(f"The provided path is not a directory. Path: {path}")
        makedirs(path, exist_ok=True)

        cls.ROOT_DIR = path

    # ============= #
    #    Getters    #
    # ============= #

    @classmethod
    def get_id(cls, filename_or_regex: str) -> List[str]:
        """
        Get the list of asset IDs that match the given filename or regex.

        Args:
            filename_or_regex (str): The filename or regex to match against asset IDs.

        Returns:
            List[str]: List of matching asset IDs.
        """

        # exact match
        if filename_or_regex in cls.FILE_TO_URL:
            return [filename_or_regex]

        # regex match
        try:
            regex = re.compile(filename_or_regex)
            ids = [file for file in cls.FILE_TO_URL if re.match(regex, file)]
            if ids:
                return ids
        except re.error:
            pass

        # load more urls
        for regex, urls_file in cls.asset_regex_to_urls_file:
            if urls_file in cls._loaded_url_files:
                continue

            if regex == filename_or_regex or re.match(regex, filename_or_regex):
                cls.load_urls(urls_file)

                # recursive call
                return cls.get_id(filename_or_regex)

        # not found
        return []

    @classmethod
    def get_url(cls, filename_or_regex: str) -> List[str]:
        """
        Get the list of URLs corresponding to the given filename or regex.

        Args:
            filename_or_regex (str): The filename or regex to match against asset IDs.

        Returns:
            List[str]: List of matching URLs.
        """

        return [cls.FILE_TO_URL[id] for id in cls.get_id(filename_or_regex)]

    @classmethod
    def get_path(cls, filename_or_regex: str) -> List[str]:
        """
        Get the list of file paths corresponding to the given filename or regex.

        Args:
            filename_or_regex (str): The filename or regex to match against asset IDs.

        Returns:
            List[str]: List of matching file paths.
        """

        return [
            join(
                (cls.urls_file_dir if asset_id.endswith("urls.json") else cls.ROOT_DIR),
                asset_id.replace("/", sep),
            )
            for asset_id in cls.get_id(filename_or_regex)
        ]

    # ================ #
    #    Management    #
    # ================ #

    @classmethod
    def download(
        cls,
        filename_or_regex: str,
        overwrite=False,
        progress_bar: bool | None = None,
        timeout: float = 20.0,
        leave=True,
        chunk_size=65536,
    ) -> bool:
        """
        Downloads package assets matching the given filename regex and saves them to the appropriate file paths.

        Args:
            filename_or_regex (str): Relative path or Regular expression to match the desired asset names.
            overwrite (bool, optional): If False, skips downloading if the resource file already exists. Defaults to False.
            progress_bar (bool, optional): If True, displays a progress bar during the download. If None, uses the value in slt.Settings.SHOW_DOWNLOAD_PROGRESS. Defaults to None.
            timeout (float, optional): The maximum number of seconds to wait for a server response. Defaults to 20.0.
            leave (bool, optional): Wether to leave the progress bar behind after the download. Defaults to True.
            chunk_size (int, optional): The number of bytes to fetch in each step. Defaults to 65536.

        Returns:
            bool: True if all resources are downloaded successfully or already exist, False otherwise.
        """

        if progress_bar is None:
            progress_bar = Settings.SHOW_DOWNLOAD_PROGRESS
            leave = False

        matching_ids = cls.get_id(filename_or_regex)
        urls = [cls.get_url(m_id)[0] for m_id in matching_ids]

        checksum = cls._read_checksum()

        statuses = []
        for asset_id, url in zip(matching_ids, urls):
            cls.delete_out_of_date_asset(asset_id, url, checksum)

            # Make sure that the file/directory exists
            file_path = cls.get_path(asset_id)[0]
            if exists(file_path) and not overwrite:
                continue
            makedirs(dirname(file_path), exist_ok=True)

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

            # update checksum file with date, url, filename, n_bytes
            if status:
                checksum[asset_id] = {
                    "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "url": url,
                }
                cls._save_checksum(checksum)

        return all(statuses or [False])

    @classmethod
    def delete_out_of_date_asset(
        cls, asset_id: str, url: str, checksum: Dict[str, Dict[str, str]]
    ) -> None:
        """
        Delete an asset if it is out of date. Determined by comparing the URL the asset
        was downloaded from differs from with the loaded URL.

        Args:
            asset_id (str): The ID of the asset to check.
            url (str): The URL associated with the asset.
            checksum (Dict[str, Dict[str, str]]): The checksum information for assets.
        """

        if (
            exists((path := cls.get_path(asset_id)[0]))
            and checksum.get(asset_id, {}).get("url", None) != url
        ):
            remove(path)
            checksum.pop(asset_id, None)

    # TODO: delete all out of date assets on import
    # @classmethod
    # def delete_all_out_of_date_assets(cls):
    #     """
    #     Delete all out of date assets determined by the checksum file and current URLs.
    #     """
    #     cls.reload()
    #     checksum = cls._read_checksum()
    #     downloaded_assets = list(checksum.keys())
    #     for asset_id in downloaded_assets:
    #         cls.delete_out_of_date_asset(asset_id, cls.get_url(asset_id)[0], checksum)
    #     cls._save_checksum(checksum) # bug: not saving checksum for urls file but downloads it

    @classmethod
    def reload(cls) -> None:
        """
        Clear the cache and read the URL files again.
        """

        cls.FILE_TO_URL = read_urls(join(cls.urls_file_dir, cls.primary_urls_file))
        cls._loaded_url_files = set()

    @classmethod
    def load_urls(cls, filename: str) -> None:
        """
        Load URLs from the specified file into the Assets.FILE_TO_URL dictionary.

        Args:
            filename (str): The name of the URLs file to load.
        """
        cls.download(filename)
        file_ids = cls.get_id(filename)
        urls_file_paths = [cls.get_path(f_id)[0] for f_id in file_ids]
        for file_id, path in zip(file_ids, urls_file_paths):
            cls.FILE_TO_URL.update(read_urls(path))
            cls._loaded_url_files.add(file_id)

    @classmethod
    def _read_checksum(cls) -> Dict[str, Dict[str, str]]:
        makedirs(cls.ROOT_DIR, exist_ok=True)
        checksum_file_path = join(cls.ROOT_DIR, cls._checksum_filename)
        if isfile(checksum_file_path):
            with open(checksum_file_path, "r", encoding="utf-8") as f:
                checksum: Dict[str, Dict[str, str]] = json.load(f)
        else:
            checksum = {}
            cls._save_checksum(checksum)

        return checksum

    @classmethod
    def _save_checksum(cls, checksum: Dict[str, Dict[str, str]]) -> None:
        checksum_file_path = join(cls.ROOT_DIR, cls._checksum_filename)
        with open(checksum_file_path, "w", encoding="utf-8") as f:
            json.dump(checksum, f, indent=2, ensure_ascii=False, sort_keys=True)
