"""hashmaps for written words to videos.
"""

import json
import os
import re
from typing import Dict, List, Set, Union

from sign_language_translator import (
    DATASET_ROOT_DIRECTORY,
    SIGN_RECORDINGS_DATASET_DIRECTORY,
    Country,
    Organization,
    SignCollection,
    TextualLanguage,
)

with open(
    os.path.join(
        SIGN_RECORDINGS_DATASET_DIRECTORY,
        "collection_to_label_to_language_to_words.json",
    ),
    "r",
) as f:
    label_2_words: Dict[str, Dict[str, Dict[str, List[str]]]] = json.load(f)

with open(
    os.path.join(
        SIGN_RECORDINGS_DATASET_DIRECTORY,
        "organization_to_language_to_constructable_words.json",
    ),
    "r",
) as f:
    constructable_words: Dict = json.load(f)

with open(
    os.path.join(
        DATASET_ROOT_DIRECTORY,
        "text_preprocessing.json",
    ),
    "r",
) as f:
    preprocessing_map: Dict[str, Dict[str, Union[Dict[str, str], List[str]]]] = json.load(f)


SIGN_COLLECTION_TO_TEXTUAL_LANGUAGE = {
    SignCollection.PK_HFAD_1: [TextualLanguage.ENGLISH, TextualLanguage.URDU],
}

END_OF_SENTENCE_MARKS: List[str] = [".", "?", "!", "۔", "؟"] + ["۔۔", "۔۔۔", "..", "..."]
PUNCTUATION: List[str] = END_OF_SENTENCE_MARKS + [",", "،"]
SYMBOLS: List[str] = (
    PUNCTUATION + [" ", "-"] + ["؛"] #+ ["'", '"'] + ["(", ")", "[", "]"] + ["“", "”", "‘", "’"]
)

PUNCTUATION_REGEX = r"\s+[" + r"|".join([re.escape(punc) for punc in PUNCTUATION]) + r"]"
URDU_DIACRITICS = "ًٍَُِٰ"
URDU_WORD_REGEX = r"[\w" + URDU_DIACRITICS + r"]+"

REKHTA_MISSPELLED_TO_CORRECT = {
    "مرا": "میرا",
    "مری": "میری",
    "مري": "میری",
    "مرے": "میرے",
}

BAD_CHARACTERS_REGEX = r"|".join(["\u200c", "\u200f", " ", "‎"])

CONTEXT_REGEX = r"\([^\(\)]*\)"  # delete every thing wraped in a pair of parenthesis


def remove_context(word: str):
    without_context = re.sub(CONTEXT_REGEX, "", word)
    return without_context


# initalize maps
AMBIGUOUS_TO_CONTEXTED: Dict[str, Dict[str, Dict[str, List[str]]]] = dict()
SUPPORTED_WORD_TO_LABEL: Dict[str, Dict[str, Dict[str, str]]] = dict()
SUPPORTED_WORD_TO_LABEL_SEQUENCE: Dict[str, Dict[str, Dict[str, List[str]]]] = dict()

for sign_collection in label_2_words.keys():
    # improve this (keep only country and org code, drop collection):
    country_org = "-".join(sign_collection.split("-")[:2])

    AMBIGUOUS_TO_CONTEXTED[country_org] = dict()
    SUPPORTED_WORD_TO_LABEL[country_org] = dict()
    SUPPORTED_WORD_TO_LABEL_SEQUENCE[country_org] = dict()

    for label, language_to_words in list(label_2_words[sign_collection].items()) + [
        (None, lang_2_wds) for lang_2_wds in constructable_words.get(country_org, [])
    ]:
        for language, words in language_to_words.items():
            if language == "components":
                continue

            AMBIGUOUS_TO_CONTEXTED[country_org].setdefault(language, dict())
            SUPPORTED_WORD_TO_LABEL[country_org].setdefault(language, dict())
            SUPPORTED_WORD_TO_LABEL_SEQUENCE[country_org].setdefault(language, dict())

            for word in words:

                # word to video(s) map
                if label is not None:
                    SUPPORTED_WORD_TO_LABEL[country_org][language][
                        word
                    ] = f"{sign_collection}_{label}"
                if "components" in language_to_words.keys():
                    SUPPORTED_WORD_TO_LABEL_SEQUENCE[country_org][language][
                        word
                    ] = language_to_words["components"]

                # word to unambigous map
                without_context = remove_context(word)
                if without_context != word:
                    if (
                        without_context
                        not in AMBIGUOUS_TO_CONTEXTED[country_org][language]
                    ):
                        AMBIGUOUS_TO_CONTEXTED[country_org][language][
                            without_context
                        ] = []
                    AMBIGUOUS_TO_CONTEXTED[country_org][language][
                        without_context
                    ].append(word)

# {"language": ["words", ...], ...}
SPELLED_WORDS: Dict[str, List[str]] = {
    lang.value: [
        word
        for word in [
            # unpack your lists here
            *preprocessing_map["person_names"][lang.value],
        ]
    ]
    for lang in TextualLanguage
}

# {"language": {"one": 1, ...}, ...}
WORDS_TO_NUMBERS: Dict[str, Dict[str, int]] = {
    lang.value: {
        word: number
        for word, number in {
            # unpack your word to number maps here (beware overwriting)
            **preprocessing_map["words_to_numbers"][lang.value],
        }.items()
    }
    for lang in TextualLanguage
}


CONTEXTED_VOCAB: Dict[str, Set[str]] = {
    lang.value: {
        *preprocessing_map["joint_word_to_split_words"][lang.value],
        *SPELLED_WORDS[lang.value],
        *WORDS_TO_NUMBERS[lang.value],
        *[
            word
            for country_org_code in AMBIGUOUS_TO_CONTEXTED
            for contexteds in AMBIGUOUS_TO_CONTEXTED[country_org_code][
                lang.value
            ].values()
            for word in contexteds
        ],
        *[
            word
            for country_org_code in SUPPORTED_WORD_TO_LABEL
            for word in SUPPORTED_WORD_TO_LABEL[country_org_code][lang.value]
        ],
        *[
            word
            for country_org_code in SUPPORTED_WORD_TO_LABEL_SEQUENCE
            for word in SUPPORTED_WORD_TO_LABEL_SEQUENCE[country_org_code][lang.value]
        ],
    }
    for lang in TextualLanguage
}

UNCONTEXTED_VOCAB: Dict[str, Set[str]] = {
    lang: {remove_context(word) for word in CONTEXTED_VOCAB[lang]}
    for lang in CONTEXTED_VOCAB
}