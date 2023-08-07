from random import choices
from typing import Any, Dict, List, Set


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


__all__ = [
    "search_in_values_to_retrieve_key",
    "sample_one_index",
]
