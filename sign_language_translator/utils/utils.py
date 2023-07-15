from typing import Any, Dict, Set


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


__all__ = [
    "search_in_values_to_retrieve_key",
]
