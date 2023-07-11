"""text preprocessing module
"""

import re
from typing import Dict


def replace_words(text: str, word_map: Dict[str, str], word_regex: str = r"\w+") -> str:
    def get_replacement(match: re.Match) -> str:
        word = match.group()
        replacement = word_map.get(word, word)
        return replacement

    text = re.sub(word_regex, get_replacement, text)
    return text


def remove_space_before_punctuation(text: str, punctuation={".", ",", "?", "!"}):
    regex = r"\s+[" + "".join([re.escape(punc) for punc in punctuation]) + r"]"

    def get_replacement(match: re.Match) -> str:
        matched_string: str = match.group(0)
        replacement_string = matched_string.lstrip()

        return replacement_string

    fixed_text = re.sub(regex, get_replacement, text)

    return fixed_text


__all__ = [
    "replace_words",
    "remove_space_before_punctuation",
]
