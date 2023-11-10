from random import choices
from typing import Any, Dict, List, Set

__all__ = [
    "search_in_values_to_retrieve_key",
    "sample_one_index",
    "in_jupyter_notebook",
    "extract_recursive",
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
        weights=[w / temperature for w in weights],
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
        >>> data = {'a': 1, 'b': {'c': 2, 'd': {'e': 3, 'f': 4}}, 'g': [5, {'h': 6, 'e': 7}]}
        >>> extract_recursive(data, 'e')
        [3, 7]
        >>> extract_recursive(data, 'h')
        [6]
        >>> extract_recursive(data, 'x')
        []  # Key not found, returns an empty list.
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
