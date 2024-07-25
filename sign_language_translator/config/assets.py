"""Module for managing assets required by the sign language translator package,
such as datasets and models.
"""

__all__ = [
    "Assets",
]

import json
import os
import re
from datetime import datetime
from os.path import abspath, dirname, exists, isdir, isfile, join, sep
from typing import Dict, List, Optional, Tuple, Union

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

        download(filename_or_regex: str, overwrite=False, progress_bar: bool = None, timeout: float = 20.0, leave=True, chunk_size=65536) -> bool:
            Download assets matching the given filename regex and save them to the appropriate file paths inside the assets root directory.

        extract(filename_or_regex: str, archive_name_or_regex: str = None, overwrite=False, download_archive=True) -> List[str]:
            extract the files matching the argument from an archived dataset into the appropriate location.

        delete(filename_or_regex: str) -> None:
            remove the matching assets from storage and its records from the checksum file.

    Example:

    .. code-block:: python

        import sign_language_translator as slt
        # slt.Assets.set_root_dir("~/centralized-slt-assets")

        # Archived datasets
        ids = slt.Assets.get_ids(r"datasets/.*\\.zip")
        paths = slt.Assets.download("datasets/pk-hfad-1.landmarks-mediapipe-world-csv.zip")
        files = slt.utils.Archive.extract(paths[0], "*.csv")

        # Specific file from archive
        path = slt.Assets.extract("pk-hfad-1_airplane.landmarks-mediapipe-world.csv", download_archive=True)

        # all dictionary videos for numbers
        urls = slt.Assets.get_url(r"videos/[a-z-]+_\\d+\\.mp4")

        # download a model
        paths = slt.Assets.download(r"models/names-stat-lm-w\\d\\.json")

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
        archive_name_or_regex: Optional[str] = None,
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
        checksum_asset_ids, checksum_infos = [], []

        arch = archive_name_or_regex or cls.infer_archive_name(filename_or_regex)
        if download_archive:
            cls.download(arch, overwrite=False, progress_bar=progress_bar, leave=leave)

        content_name = filename_or_regex.split("/")[-1]
        extracted_assets = []
        for archive_id in cls.get_ids(arch):
            asset_dir = archive_id.split("/")[-1].split(".")[-2].split("-")[0]
            extracted_assets += Archive.extract(
                archive_path=cls._abs_path(archive_id),
                regex=content_name,
                output_dir=cls._abs_path(asset_dir),
                overwrite=overwrite,
                progress_bar=progress_bar,
                leave=leave,
                verbose=False,
            )

            for asset_id in extracted_assets:
                # todo: skip those which were not extracted/overwritten but existed already
                checksum_asset_ids.append(asset_id)
                checksum_infos.append(
                    {
                        "archive_id": archive_id,
                        "archive_url": cls.get_url(archive_id)[0],
                    }
                )

        cls._update_checksum(checksum_asset_ids, checksum_infos)
        return extracted_assets

    @classmethod
    def download(
        cls,
        filename_or_regex: str,
        overwrite=False,
        timeout: float = 20.0,
        chunk_size=2**18,
        progress_bar: Optional[bool] = None,
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
            # false record
            if not exists(path):
                checksum.pop(asset_id, None)
                continue
            # outdated URL
            if (
                "url" in info
                and asset_id in cls.FILE_TO_URL  # is loaded
                and info["url"] != cls.FILE_TO_URL[asset_id]
            ):
                os.remove(path)
                checksum.pop(asset_id, None)

            # outdated source archive
            if (
                "archive_id" in info
                and "archive_url" in info
                and info["archive_id"] in cls.FILE_TO_URL
                and info["archive_url"] != cls.FILE_TO_URL[info["archive_id"]]
            ):
                os.remove(cls._abs_path(asset_id))
                checksum.pop(asset_id, None)

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
    def is_dictionary_video(cls, filename: str) -> bool:
        """Class method to check if the given filename is a dictionary video.
        Checks the folder name, extension & direct URL.

        Args:
            filename (str): The asset ID to check. (e.g. 'videos/pk-hfad-1_airplane.mp4')

        Returns:
            bool: True if the filename represents a dictionary video, False otherwise.
        """
        folder, basename = filename.split("/")
        if folder != "videos":
            return False

        label, extension = basename.rsplit(".", maxsplit=1)
        if extension != "mp4":
            return False

        chunks = label.split(Settings.FILENAME_SEPARATOR)
        if len(chunks) == 2 and len(cls.get_ids(filename)) == 1:
            return True

        return False

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
    def _update_checksum(
        cls, asset_id: Union[str, List[str]], info: Union[Dict, List[Dict]]
    ):
        if not isinstance(asset_id, list):
            asset_id = [asset_id]
        if not isinstance(info, list):
            info = [info]

        checksum = cls._read_checksum()
        for _asset_id, _info in zip(asset_id, info):
            checksum.setdefault(_asset_id, {}).update(
                {"date": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"), **_info}
            )
        cls._write_checksum(checksum)

    @classmethod
    def infer_archive_name(cls, filename_or_regex: str) -> str:
        """
        Infers the archive name/regex that should contain the given asset based on the provided filename_or_regex argument.
        Please follow the naming convention to avoid false guesses
        i.e. use all allowed special symbols `["/", "-", "_", ".", r"\\."]` in the right places.

        Args:
            filename_or_regex (str): The asset filename or regex from which its containing archive name must be inferred.

        Returns:
            str: A regex pattern that matches the archive name which should contain the given asset.
        """

        if filename_or_regex.endswith("zip"):
            return filename_or_regex

        base = filename_or_regex.split("/")[-1]
        collection = base.split("_", 1)[0] if "_" in base else r".*"
        category = r".^"
        model = r".^"
        extension = ext if (ext := base[-3:]) in ("csv", "npz", "npy", "mp4") else r".*"

        if "landmarks" in filename_or_regex:
            category = "landmarks" if extension != "mp4" else r".^"  # landmarks/x.mp4

            # assuming the filename structure follows the convention (select the part between two '.'s)
            sub_extension = re.split(r"(\\\.|\.(?![\*\+\{\?]))", filename_or_regex)[-3]
            model = sub_extension.split("-", maxsplit=1)[-1]
            if not model.startswith(("mediapipe", "testmodel")):  # validation
                model = r".*"

        elif "video" in filename_or_regex or extension == "mp4":
            category = "videos" if extension in ("mp4", r".*") else r".^"  # video/x.csv
            model = "?"  # todo: remove this hack
            # todo: make it work for dictionary & replications
            if extension == r".*":
                extension = "mp4"

        return f"datasets/{collection}\\.{category}-{model}-{extension}\\.zip$"
