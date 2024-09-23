import os
import re
import socket
from enum import EnumMeta
from random import choices
from typing import Any, Dict, List, Set, Union

from tqdm.auto import tqdm

__all__ = [
    "PrintableEnumMeta",
    "ProgressStatusCallback",
    "extract_recursive",
    "in_jupyter_notebook",
    "is_regex",
    "sample_one_index",
    "search_in_values_to_retrieve_key",
    "validate_path_exists",
    "is_internet_available",
]


def search_in_values_to_retrieve_key(
    code_name: str, class_to_codes: Dict[Any, Set[str]]
):
    # verify there is no repetition/reuse in language codes
    all_codes = [code for codes in class_to_codes.values() for code in codes]
    assert len(all_codes) == len(set(all_codes)), "code reused for multiple keys"

    for key, codes in class_to_codes.items():
        if code_name.lower() in codes:
            return key

    return None


def sample_one_index(weights: List[float], temperature: float = 1.0) -> int:
    """Select an item based on the given probability distribution.
    Returns the index of the selected item sampled from weighted random distribution.

    Args:
        weights (List[float]): the relative weights corresponding to each index.
        temperature (float): The temperature value for controlling the sampling behavior.
            High temperature means sampling probabilities are more uniform (says random things).
            Low temperature means that sampling probabilities are higher for bigger weights.
            Defaults to 1.0.

    Returns:
        int: The index of the chosen item.
    """

    return choices(
        range(len(weights)),
        weights=[w ** (1 / temperature) for w in weights],
        k=1,
    )[0]


def in_jupyter_notebook():
    """
    Checks if the code is running in a Jupyter notebook.

    Returns:
        bool: True if running in a Jupyter notebook, False otherwise.
    """

    try:
        from IPython import get_ipython  # type: ignore

        return "IPKernelApp" in get_ipython().config  # type: ignore
    except:  # pylint: disable = bare-except
        return False


def extract_recursive(data: Dict[str, Any], key: str) -> List[Any]:
    """
    Recursively extracts values associated with a specified key from a nested dictionary.

    Args:
        data (Dict[str, Any]): The input dictionary containing nested structures.
        key (str): The key for which values need to be extracted.

    Returns:
        List[Any]: A list containing all values associated with the specified key, extracted
                   recursively from the input dictionary.

    Examples:

    .. code-block:: python

        data = {'a': 1, 'b': {'c': 2, 'd': {'e': 3, 'f': 4}}, 'g': [5, {'h': 6, 'e': 7}]}
        extract_recursive(data, 'e')
        # [3, 7]
        extract_recursive(data, 'h')
        # [6]
        extract_recursive(data, 'x')
        # []  # Key not found, returns an empty list.
    """

    extracted_values = []

    def extract(data: Dict, results: List):
        for k in data:
            if k == key:
                results.append(data[k])
            elif isinstance(data[k], dict):
                extract(data[k], results)
            elif isinstance(data[k], list):
                for item in data[k]:
                    if isinstance(item, dict):
                        extract(item, results)

    extract(data, extracted_values)

    return extracted_values


class PrintableEnumMeta(EnumMeta):
    """
    Metaclass for customizing the string representation of Enum classes.

    This metaclass overrides the __str__ & __repr__ method to provide a human-readable
    representation of Enum classes when they are printed. The generated string
    includes the class name and a formatted list of Enum members and their values.

    Example:

    .. code-block:: python

        class MyEnumClass(enum.Enum, metaclass=PrintableEnumMeta):
            MEMBER1 = "value_1"
            MEMBER2 = "value_2"

        print(MyEnumClass)

        # "MyEnumClass" enum class. Available values:
        # 1. MEMBER1 = value_1
        # 2. MEMBER2 = value_2
    """

    def __str__(cls):
        members = "\n".join(
            [
                f"{i + 1}. {member.name} = {member.value}"  # type:ignore
                for i, member in enumerate(cls)
            ]
        )
        return f'"{cls.__name__}" enum class. Available values:\n{members}'

    def __repr__(cls) -> str:
        return str(cls)

    def __contains__(cls, item) -> bool:
        return item in cls._value2member_map_


class ProgressStatusCallback:
    """
    A callback class to update a tqdm progress bar with custom status information.

    Args:
        tqdm_bar (tqdm): The tqdm progress bar to be updated.

    Attributes:
        tqdm_bar (tqdm): The tqdm progress bar associated with the callback.

    Methods:
        __call__(self, status: Dict[str, Any]):
            Update the tqdm progress bar with the provided status information.

    Example:

    .. code-block:: python

        # Instantiate a tqdm progress bar & callback
        progress_bar = tqdm(total=100, desc='Processing')
        callback = ProgressStatusCallback(tqdm_bar=progress_bar)

        # Update the progress bar inside some other function
        status_info = {'Epoch': 1, 'Loss': 0.123, 'Accuracy': 0.95}
        callback(status_info)
    """

    def __init__(self, tqdm_bar: tqdm):
        """
        Initialize the ProgressStatusCallback with a tqdm progress bar.

        Args:
            tqdm_bar (tqdm): The tqdm progress bar to be associated with the callback.
        """
        self.tqdm_bar = tqdm_bar

    def __call__(self, status: Dict[str, Any]):
        """
        Update the tqdm progress bar with the provided status information.

        Args:
            status (Dict[str, Any]): A dictionary containing custom status information.
                This information will be displayed as postfix on the tqdm progress bar.
        """
        self.tqdm_bar.set_postfix(status, refresh=True)


def is_regex(string: Union[str, re.Pattern]) -> bool:
    """Tests whether the argument is a regex or a regular string.

    Args:
        string (str | Pattern): The string to be tested.

    Returns:
        bool: whether the argument is a regex (True) or a regular string (False).
    """
    if isinstance(string, re.Pattern):
        return True

    if set("+*?|[]{}^$").intersection(set(string)):
        try:
            re.compile(string)
            return True
        except re.error:
            return False

    return False


def validate_path_exists(path: str, overwrite: bool = False) -> str:
    """
    Validates the existence of a given file path and optionally creates necessary directories.

    This function checks if a file already exists at the specified path. If the file exists
    and `overwrite` is set to `False`, a `FileExistsError` is raised. If `overwrite` is set
    to `True`, or if the file does not exist, the function returns the absolute path after
    ensuring that all necessary directories are created.

    Args:
        path (str): The file path to be validated.
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.

    Raises:
        FileExistsError: If the file already exists at the specified path and `overwrite` is set to `False`.

    Returns:
        str: The absolute path of the validated file.

    Examples:
        >>> validate_path_exists('/path/to/file.txt', overwrite=False)
        '/absolute/path/to/file.txt'
    """

    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"File already exists: '{path}' (Use overwrite=True)")

    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    return path


def is_internet_available() -> bool:
    """Hit a well-known server (Google DNS) to check for internet availability.

    Returns:
        bool: True if internet is available, False otherwise.
    """
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False
