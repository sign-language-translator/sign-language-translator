"""
Module for working with archives.

Classes:
    Archive: A utility class with static methods for creating, listing, and extracting files from ZIP archives.
"""

import fnmatch
import re
import zipfile
from glob import glob
from os.path import basename, exists, join
from typing import List, Optional, Union
from warnings import warn

from tqdm.auto import tqdm

from sign_language_translator.utils.utils import is_regex


class Archive:
    """This utility class provides static methods for creating, listing, and extracting files from ZIP archives.

    Methods:
    - create(filename_or_patterns: str | List[str], archive_path: str, compression=zipfile.ZIP_DEFLATED,
             progress_bar=True, overwrite=False)
        Create a ZIP archive from files matching the specified pattern.

    - list(archive_path: str, pattern="*", regex: str = r".*") -> List[str]
        List the files in a ZIP archive, optionally filtered by a glob pattern or regex.

    - extract(archive_path: str, pattern: str = "*", regex: str | re.Pattern = r".*", output_dir: str = ".",
              overwrite=False, progress_bar=True, leave=True, password: bytes = None, verbose=True) -> List[str]
        Extract files from a ZIP archive to the specified output directory, optionally
        filtered by file names, patterns, or regex.

    Example:

    .. code-block:: python

        from sign_language_translator.utils import Archive

        # Create a ZIP archive with files matching a pattern
        Archive.create("*.txt", "output_archive.zip", overwrite=True)

        # List files in a ZIP archive using a pattern and a regular expression
        files = Archive.list("input_archive.zip", pattern="file_*.txt", regex=r"file_\\d\\.txt")
        print(files)

        # Extract files from a ZIP archive to a specified directory
        extracted_files = Archive.extract("input_archive.zip", pattern="*.txt", output_dir="output_dir", overwrite=True)
        print(extracted_files)

    Note:
        - For file patterns, this class uses glob-style patterns e.g. "*.mp4".
        - When extracting files, warnings are issued for skipped files with the same base name.
    """

    @staticmethod
    def create(
        filename_or_patterns: Union[str, List[str]],
        archive_path: str,
        compression=zipfile.ZIP_DEFLATED,
        progress_bar=True,
        overwrite=False,
    ):
        """
        Create a zip archive from files matching the given pattern.

        Args:
            filename_or_patterns (str | List[str]): Files or Unix shell-style patterns matching the files to include in the archive.
            archive_path (str): Path to the output zip archive.
            compression (int, optional): Compression method (default is zipfile.ZIP_DEFLATED).
            progress_bar (bool, optional): Show a progress bar during creation (default is True).
            overwrite (bool, optional): Overwrite existing archive (default is False).

        Raises:
            FileExistsError: If the archive_path already exists and overwrite is False.
        """

        if exists(archive_path) and not overwrite:
            raise FileExistsError(
                f"'{archive_path = }' already exists. Use overwrite=True to replace it."
            )

        if isinstance(filename_or_patterns, str):
            filename_or_patterns = [filename_or_patterns]

        files = [f for pattern in filename_or_patterns for f in glob(pattern)]
        added = set()

        with zipfile.ZipFile(archive_path, "w", compression=compression) as zf:
            for file in tqdm(files) if progress_bar else files:
                if (base := basename(file)) not in added:
                    zf.write(file, base)
                    added.add(base)
                else:
                    warn(f"Skipping '{file}'. Already added a file of same base name.")

    @staticmethod
    def list(
        archive_path: str, pattern: str = "*", regex: Union[str, re.Pattern] = r".*"
    ) -> List[str]:
        """
        List files in the zip archive filtered by the specified pattern or regex.

        Args:
            archive_path (str): Path to the zip archive.
            pattern (str): Unix shell-style wildcard pattern to filter the contents (default is "*").
            regex (str | re.Pattern): Regular expression pattern to filter the contents (default is ".*").

        Returns:
            List[str]: List of file names in the archive that match the criteria.
        """

        # load
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = zf.namelist()

        # filter
        if pattern not in ["", "*"]:
            names = fnmatch.filter(names, pattern)

        if regex not in ["", r".*"]:
            names = [
                n
                for n in names
                if (re.match(regex, n) if is_regex(regex) else n == regex)
            ]

        return names

    @staticmethod
    def extract(
        archive_path: str,
        pattern: str = "*",
        regex: Union[str, re.Pattern] = r".*",
        output_dir: str = ".",
        overwrite=False,
        progress_bar=True,
        leave=True,
        password: Optional[bytes] = None,
        verbose=True,
    ) -> List[str]:
        """
        Extract specified files from a zip archive. Only those files are extracted that match the regex AND the pattern.

        Args:
            archive_path (str): Path to the zip archive.
            pattern (str): Unix shell-style wildcard pattern that specifies the files to extract (default is "*").
            regex (str | re.Pattern): Regular expression pattern that specifies the files to extract (default is ".*").
            output_dir (str, optional): Directory to extract files into (Default is ".").
            overwrite (bool, optional): Overwrite existing files during extraction (default is False).
            progress_bar (bool, optional): Show a progress bar during extraction (default is True).
            leave (bool, optional): Leave progress bar displayed upon completion (default is True).
            password (bytes, optional): Password for encrypted archives (default is None).
            verbose (bool, optional): Raise warnings for skipped existing files (default is True).

        Returns:
            List[str]: List of paths to the extracted files and the already extracted matching files.
        """
        files = Archive.list(archive_path, pattern, regex)

        # filter existing files
        extracted_files = []
        files_to_extract = []
        for file in files:
            if not overwrite and exists(output_path := join(output_dir, file)):
                extracted_files.append(output_path)
                if verbose:
                    warn(f"Skipping '{output_path}'.Use overwrite=True to extract.")
                continue
            files_to_extract.append(file)
        files = files_to_extract

        # extract
        with zipfile.ZipFile(archive_path, "r") as zf:
            files = tqdm(files, leave=leave) if progress_bar and files else files
            for file in files:
                if isinstance(files, tqdm):
                    files.set_description(
                        f"Extracting {basename(archive_path)}: '{file}'"
                    )

                zf.extract(file, output_dir, pwd=password)
                extracted_files.append(join(output_dir, file))

        return extracted_files

    # TODO: add(), remove()
