"""Module for managing assets required by the sign language translator package,
such as datasets and models.
"""

import json
import os
import re
from datetime import datetime
from os.path import abspath, dirname, exists, isdir, isfile, join, sep
from typing import Dict, List, Tuple

from tqdm.auto import tqdm

from sign_language_translator.config.settings import Settings
from sign_language_translator.config.utils import read_urls
from sign_language_translator.utils import (
    Archive,
    ProgressStatusCallback,
    download,
    is_regex,
)


class Assets:
    """
    Static class for managing assets related to the sign language translator.
    It wraps around utility functions to automatically handle downloading, extracting, deleting and loading assets.

    Attributes:
        ROOT_DIR (str): The root directory path where the sign language datasets & models are stored.
        FILE_TO_URL (Dict[str, str]): A dictionary mapping asset filenames to their corresponding URLs.
        asset_regex_to_urls_file (List[Tuple[str, str]]): A list of tuples mapping regular expressions matching asset names to URLs files. The UR file must be a JSON containing a key "file_to_url" which maps to an object mapping relative paths of assets to their URLs.

    Methods:
        set_root_dir(path: str) -> None:
            Set the SLT resources directory path.

        get_ids(filename_or_regex: str) -> List[str]:
            Get the relative paths of assets matching the given filename_or_regex.

        download(filename_or_regex: str, overwrite=False, progress_bar: bool | None = None, timeout: float = 20.0, leave=True, chunk_size=65536) -> bool:
            Download assets matching the given filename regex and save them to the appropriate file paths inside the assets root directory.

        extract(filename_or_regex: str, archive_name_or_regex: str | None = None, overwrite=False, download_archive=True) -> List[str]:
            extract the files matching the argument from an archived dataset into the appropriate location.

        fetch(filename_or_regex: str, overwrite=False, archive_name_or_regex: str=None, download_archive=False) -> List[str]:
            extract the required files if an archive is pre-downloaded otherwise download the files if direct URL is available otherwise download the archive and extract.

        delete(filename_or_regex: str) -> None:
            remove the matching assets from storage and its records from the checksum file.

    Example:
    .. code-block:: python

        import sign_language_translator as slt
        # slt.Assets.set_root_dir("./custom-centralized-assets")

        # Archived datasets
        ids = slt.Assets.get_ids(r"datasets/.*\\.zip")
        paths = slt.Assets.download("datasets/pk-hfad-1_landmarks-mediapipe-pose-2-hand-1-csv.zip")
        files = slt.Assets.extract(paths[0], r".*\\.csv")

        # all dictionary videos for numbers
        urls = slt.Assets.get_url(r"videos/[a-z-]+_\\d+\\.mp4")

        # Automatically extract dictionary videos from an archive if its available else download from direct URL
        # try not to use vague regex anywhere and its best if you manually download needed dataset archive
        paths = slt.Assets.fetch(r"videos/pk-hfad-1_\\d+\\.mp4")

        # download a model
        paths = slt.Assets.download(r".*/names-stat-lm-w1\\.json")

        # add your own dataset
        slt.Assets.FILE_TO_URL.update({"relative/path/to/asset.mp4": "https://...",})
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
        (r"^videos/pk-.*mp4$", "pk-dictionary-urls.json"),
        (r".*zip$", "archive-urls.json"),
        (r".*", "extra-urls.json"),
    ]
    """A list of tuples that map regular expressions matching asset names to URLs files containing URLs to that asset group.
    The regex are tried sequentially so make sure list order is right."""

    _checksum_filename = "checksum.json"
    _loaded_url_files = set()

    @classmethod
    def set_root_dir(cls, path: str) -> None:
        """Set the SLT resources directory path.
        Helpful when using custom datasets, or when data must be stored outside install directory at a centralized location.
        (e.g hosted on cloud and mounted on disk).
        By default, resources are downloaded to the 'install_directory/assets'.

        Args:
            path (str): The path to the assets, datasets or models directory.

        Raises:
            ValueError: If the provided path is not a directory.
        """

        path = abspath(path)
        if exists(path) and not isdir(path):
            raise ValueError(f"The provided path is not a directory. Path: {path}")
        os.makedirs(path, exist_ok=True)

        cls.ROOT_DIR = path

    # ============= #
    #    Getters    #
    # ============= #

    @classmethod
    def get_ids(cls, filename_or_regex: str) -> List[str]:
        """
        Filters down the loaded Assets.FILE_TO_URLS dictionary and
        returns the list of asset IDs (relative paths) that match the given filename_or_regex.
        If no asset_id match the argument, appropriate urls file is loaded from the
        Assets.asset_regex_to_urls_file list and the function is called recursively.

        Args:
            filename_or_regex (str): The filename or regex to match against asset IDs.

        Returns:
            List[str]: List of matching asset IDs.
        """

        # exact match
        if filename_or_regex in cls.FILE_TO_URL:
            return [filename_or_regex]

        # regex match #* (but only with currently loaded url files)
        if is_regex(filename_or_regex):
            regex = re.compile(f"^{filename_or_regex}$")
            if ids := [file for file in cls.FILE_TO_URL if regex.match(file)]:
                return ids

        # load more urls
        for regex, urls_file in cls.asset_regex_to_urls_file:
            if urls_file in cls._loaded_url_files:
                continue

            if regex == filename_or_regex or re.match(regex, filename_or_regex):
                cls.load_urls(urls_file)

                # recursive call
                return cls.get_ids(filename_or_regex)

        # not found
        return []

    @classmethod
    def get_url(cls, filename_or_regex: str) -> List[str]:
        """
        Filters down the loaded Assets.FILE_TO_URLS dictionary and
        returns the list of URLs corresponding to the asset_ids matching given filename_or_regex.

        Args:
            filename_or_regex (str): The filename or regex to match against asset IDs.

        Returns:
            List[str]: List of matching URLs.
        """

        return [cls.FILE_TO_URL[id] for id in cls.get_ids(filename_or_regex)]

    @classmethod
    def get_path(cls, filename_or_regex: str) -> List[str]:
        """
        Filters down the loaded Assets.FILE_TO_URLS dictionary and
        returns the list of file paths corresponding to the asset_ids matching given filename_or_regex.

        Args:
            filename_or_regex (str): The filename or regex to match against asset IDs.

        Returns:
            List[str]: List of matching file paths.
        """

        return [cls._abs_path(asset_id) for asset_id in cls.get_ids(filename_or_regex)]

    # ============== #
    #    Fetching    #
    # ============== #

    @classmethod
    def extract(
        cls,
        filename_or_regex: str,
        archive_name_or_regex: str | None = None,
        overwrite=False,
        progress_bar=True,
        leave=True,
        download_archive=True,
    ) -> List[str]:
        """
        Extracts assets matching the given filename_or_regex from the archived datasets.
        The target location is inferred from the archive & asset name and resides inside assets root directory.
        If the archive name is not provided, it will be inferred from the filename_or_regex, so avoid using vague regex and follow the filename structure.
        If the archive is not downloaded, it will be downloaded first.

        Note:
            Please use the `slt.utils.Archive.extract()` function directly if you want deterministic behavior and avoid false guesses.

        Args:
            filename_or_regex (str): The filename or regex pattern to match the archive contents.
            archive_name_or_regex (str, optional): The name or regex pattern of the archive(s) that contains the assets. If None, the function tries guess it from the content name. Defaults to None.
            overwrite (bool, optional): Flag indicating whether to overwrite existing assets. Defaults to False.
            progress_bar (bool, optional): Flag indicating whether to display a progress bar during extraction. Defaults to True.
            leave (bool, optional): Flag indicating whether to leave the progress bar displayed after extraction. Defaults to True.
            download_archive (bool, optional): Flag indicating whether to download the archive if it is not already downloaded. Defaults to True.

        Returns:
            List[str]: A list of paths to the extracted assets.
        """
        cls.delete_out_of_date_assets()

        archive_regex = archive_name_or_regex or cls._infer_archive_name(
            filename_or_regex
        )
        if download_archive:
            cls.download(
                archive_regex, overwrite=False, progress_bar=progress_bar, leave=leave
            )

        content_name = filename_or_regex.split("/")[-1]
        extracted_assets = []
        for archive_id in cls.get_ids(archive_regex):
            asset_dir = archive_id.split("/")[-1].split("_")[-1].split("-")[0]
            extracted_assets += Archive.extract(
                archive_path=cls._abs_path(archive_id),
                regex=content_name,
                output_dir=cls._abs_path(asset_dir),
                overwrite=overwrite,
                progress_bar=progress_bar,
                leave=leave,
            )

        # TODO: checksum

        return extracted_assets

    @classmethod
    def download(
        cls,
        filename_or_regex: str,
        overwrite=False,
        timeout: float = 20.0,
        chunk_size=2**18,
        progress_bar: bool | None = None,
        leave=True,
    ) -> List[str]:
        """
        Downloads package assets matching the given filename regex and saves them to the appropriate file paths.

        Args:
            filename_or_regex (str): Relative path or Regular expression to match the desired asset names.
            overwrite (bool, optional): If False, skips downloading if the resource file already exists. Defaults to False.
            timeout (float, optional): The maximum number of seconds to wait for a server response. Defaults to 20.0.
            chunk_size (int, optional): The number of bytes to fetch in each step. Defaults to 256*1024.
            progress_bar (bool, optional): If True, displays a progress bar during the download. If None, uses the value in slt.Settings.SHOW_DOWNLOAD_PROGRESS. Defaults to None.
            leave (bool, optional): Wether to leave the progress bar behind after the download. Defaults to True.

        Returns:
            List[str]: List of paths to matching files that were downloaded or existed already.
        """
        cls.delete_out_of_date_assets()
        existing_paths = []

        # Select assets to download
        id_path_url = []
        for asset_id in cls.get_ids(filename_or_regex):
            path = cls._abs_path(asset_id)
            if exists(path) and not overwrite:
                existing_paths.append(path)
                continue
            id_path_url.append((asset_id, path, cls.FILE_TO_URL[asset_id]))

        # Configure progress bar
        callback = None
        if progress_bar is None:
            progress_bar = Settings.SHOW_DOWNLOAD_PROGRESS
            leave = False
        if progress_bar and len(id_path_url) > 1:
            id_path_url = tqdm(id_path_url, leave=leave)
            callback = ProgressStatusCallback(id_path_url)

        # Download assets
        for asset_id, file_path, url in id_path_url:
            # progress bar
            if isinstance(id_path_url, tqdm):
                id_path_url.set_description(f"Downloading {asset_id}")

            # Make sure that the file/directory exists
            os.makedirs(dirname(file_path), exist_ok=True)

            # Download the file from the URL
            success_in_download = download(
                file_path,
                url,
                progress_bar=(progress_bar and len(id_path_url) == 1),
                timeout=timeout,
                overwrite=overwrite,
                chunk_size=chunk_size,
                status_callback=callback,
            )

            # update checksum file with date, url, filename, n_bytes
            if success_in_download:
                cls._update_checksum(asset_id, {"url": url})

        return [path for _, path, _ in id_path_url] + existing_paths

    @classmethod
    def fetch(
        cls,
        filename_or_regex: str,
        overwrite=False,
        archive_name_or_regex: str | None = None,
        download_archive=False,
        progress_bar=True,
        leave=True,
    ) -> List[str]:
        """Tries to extract the asset from an archive if it is available, otherwise downloads the asset from its direct URL if available.
        If the archive name is not provided, it will be inferred from the filename_or_regex.

        Note:
            Only use for dictionary videos because only their direct URLs & archives both are available.
            Follow the filename structure to avoid false guesses (i.e. use all - and _ in the right places and insert wildcards as required).

        Args:
            filename_or_regex (str): The filename or regex to match against asset IDs.
            overwrite (bool, optional): Flag indicating whether to overwrite existing assets. Defaults to False.
            archive_name_or_regex (str, optional): The name or regex pattern of the archive(s) that contains the assets. If None, the function tries guess it from the content name. Defaults to None.
            download_archive (bool, optional): Flag indicating whether to download the archive if it is not already downloaded. Defaults to False.
            progress_bar (bool, optional): Flag indicating whether to show the progress bars or not. Defaults to True.
            leave (bool, optional): Flag indicating whether to leave the progress_bar behind after completion or not. Defaults to True.

        Returns:
            List[str]: List of paths to files that were downloaded or extracted.

        Example:
        .. code-block:: python

            import sign_language_translator as slt

            # extract/download all pakistan-sign-language dictionary videos for numbers
            paths = slt.Assets.fetch(r"videos/pk-[a-z]+-[0-9]+_\\d+\\.mp4")
        """

        extracted_assets = cls.extract(
            filename_or_regex,
            archive_name_or_regex=archive_name_or_regex,
            overwrite=overwrite,
            download_archive=download_archive,
            progress_bar=progress_bar,
            leave=leave,
        )
        downloaded_assets = cls.download(
            filename_or_regex,
            overwrite=False,
            progress_bar=progress_bar,
            leave=leave,
        )

        return list(set(extracted_assets + downloaded_assets))

    # ============ #
    #    Delete    #
    # ============ #

    @classmethod
    def delete_out_of_date_assets(cls) -> None:
        """
        Delete asset if it is out of date. Currently determined by comparing the download URL
        of the asset in checksum file with the loaded URL.
        Does not delete files not present in checksum.
        """

        checksum = cls._read_checksum()

        for asset_id, info in checksum.copy().items():
            path = cls._abs_path(asset_id)
            if not exists(path):
                checksum.pop(asset_id, None)
                continue
            if (
                "url" in info
                and asset_id in cls.FILE_TO_URL  # is loaded
                and info["url"] != cls.FILE_TO_URL[asset_id]
            ):
                os.remove(path)
                checksum.pop(asset_id, None)

            # TODO: check info["archive_id"] and delete archive & all extracted_contents

        cls._write_checksum(checksum)

    @classmethod
    def delete(cls, filename_or_regex: str):
        checksum = cls._read_checksum()
        matching_ids = cls.get_ids(filename_or_regex)
        for asset_id in matching_ids:
            if exists(file_path := cls._abs_path(asset_id)):
                os.remove(file_path)
                checksum.pop(asset_id, None)

        cls._write_checksum(checksum)

    # ================= #
    #    URL Loading    #
    # ================= #

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
        for file_id in cls.get_ids(filename):
            cls.FILE_TO_URL.update(read_urls(cls._abs_path(file_id)))
            cls._loaded_url_files.add(file_id)

    @classmethod
    def load_all_urls(cls) -> None:
        """
        Load all URL files into the Assets.FILE_TO_URL dictionary.
        """
        for _, urls_file in cls.asset_regex_to_urls_file:
            cls.load_urls(urls_file)
        cls.load_urls(r"([a-z-]+[-_])?urls.json")

    # ============= #
    #    helpers    #
    # ============= #

    @classmethod
    def _abs_path(cls, asset_id: str) -> str:
        return join(
            (cls.urls_file_dir if asset_id.endswith("urls.json") else cls.ROOT_DIR),
            asset_id.replace("/", sep),
        )

    @classmethod
    def _read_checksum(cls) -> Dict[str, Dict[str, str]]:
        os.makedirs(cls.ROOT_DIR, exist_ok=True)
        checksum_file_path = join(cls.ROOT_DIR, cls._checksum_filename)
        if isfile(checksum_file_path):
            with open(checksum_file_path, "r", encoding="utf-8") as f:
                checksum: Dict[str, Dict[str, str]] = json.load(f)
        else:
            checksum = {}
            cls._write_checksum(checksum)

        return checksum

    @classmethod
    def _write_checksum(cls, checksum: Dict[str, Dict[str, str]]) -> None:
        checksum_file_path = join(cls.ROOT_DIR, cls._checksum_filename)
        with open(checksum_file_path, "w", encoding="utf-8") as f:
            json.dump(checksum, f, indent=2, ensure_ascii=False, sort_keys=True)

    @classmethod
    def _update_checksum(cls, asset_id: str, info: Dict):
        checksum = cls._read_checksum()
        if asset_id not in checksum:
            checksum[asset_id] = {}
        checksum[asset_id].update(
            {"date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), **info}
        )
        cls._write_checksum(checksum)

    @classmethod
    def _infer_archive_name(cls, filename_or_regex: str) -> str:
        """
        Infers the archive name/regex that should contain the given asset based on the provided filename_or_regex argument.

        Args:
            filename_or_regex (str): The asset filename or regex from which its containing archive name must be inferred.

        Returns:
            str: A regex pattern that matches the archive name which should contain the given asset.
        """

        if filename_or_regex.endswith("zip"):
            return filename_or_regex

        base = filename_or_regex.split("/")[-1]
        collection = base.split("_", 1)[0] if "_" in base else ".*"
        extension = ext if (ext := base[-3:]) in ("csv", "npz", "npy", "mp4") else ".*"

        if "video" in filename_or_regex or extension == "mp4":
            category = "videos"
            model = "?"
            # todo: remove this hack
            # todo: make it work for dictionary & replications
            if extension == ".*":
                extension = "mp4"
        elif "landmarks" in filename_or_regex:
            category = "landmarks"
            model = ".*"
        else:
            model = ".^"
            category = ".^"

        return f"datasets/{collection}_{category}-{model}-{extension}\\.zip"
