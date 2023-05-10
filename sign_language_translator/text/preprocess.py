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
