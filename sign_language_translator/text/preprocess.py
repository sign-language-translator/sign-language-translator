"""text preprocessing module
"""

import re
from typing import Dict

from urduhack.normalization import normalize_characters, normalize_combine_characters

from .vocab import (
    BAD_CHARACTERS_REGEX,
    REKHTA_MISSPELLED_TO_CORRECT,
    URDU_WORD_REGEX,
    TextualLanguage,
    preprocessing_map,
)


def replace_words(text: str, word_map: Dict[str, str], word_regex: str = r"\w+") -> str:
    def get_replacement(match: re.Match) -> str:
        word = match.group()
        replacement = word_map.get(word, word)
        return replacement

    text = re.sub(word_regex, get_replacement, text)
    return text


def urdu_text_normalization(text: str):
    text = normalize_characters(text)
    text = normalize_combine_characters(text)
    text = replace_words(
        text,
        word_map=preprocessing_map["misspelled_to_correct"][TextualLanguage.URDU.value],
        word_regex=URDU_WORD_REGEX,
    )
    text = re.sub(BAD_CHARACTERS_REGEX, " ", text)
    text = re.sub(r"[۔\.][۔\. ]+[\.۔]", "۔۔۔", text)

    return text


def urdu_poetry_preprocessor(text: str) -> str:
    text = ("؛ ").join(
        [
            line.strip("() '\"\t")
            for line in text.splitlines()
            if len(re.findall(URDU_WORD_REGEX, line)) > 1
        ]
    )
    text = replace_words(
        text,
        word_map=REKHTA_MISSPELLED_TO_CORRECT,
        word_regex=URDU_WORD_REGEX,
    )

    return text


def urdu_passage_preprocessor(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def urdu_wikipedia_preprocessor(text: str) -> str:
    text = text.strip(". !\"'\n\t")
    return text
