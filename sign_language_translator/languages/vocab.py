"""File-handling to create word maps.
"""

import json
import os
import re
from typing import Dict, List, Set, Union, Tuple

from ..config.settings import Settings


class Vocab:
    """loads data from files for a specific language and specified sign_collections"""

    def __init__(
        self,
        language: str,
        sign_collections: List[str],
    ) -> None:
        self.language = language
        self.sign_collections = sign_collections
        self.sign_labels = self._load_sign_labels()
        self.preprocessing_map = self._load_preprocessing(self.language)
        self.label_to_words = self._load_label_to_words(language, sign_collections)
        self.label_sequence_to_words = self._load_constructable_words(
            self.language, self.sign_collections
        )
        self.context_regex = r"\([^\(\)]*\)"  # delete all occurrences of everything wrapped in a pair of parenthesis
        self.supported_words_with_context = self._make_supported_words()
        self.supported_words = {
            self.remove_context(word) for word in self.supported_words_with_context
        }
        self.ambiguous_to_context = self._make_ambiguous_map(
            self.supported_words_with_context
        )

        self.NON_SENTENCE_END_TOKENS = {
            w
            for wc in self.supported_words_with_context
            for w in [self.remove_context(wc)]
            if (("double-handed-letter)" in wc) and (not w.isascii()))
            or (len(w) == 1 and w.isalnum())
        }

    def remove_context(self, word: str):
        without_context = re.sub(self.context_regex, "", word)
        return without_context

    def _load_label_to_words(
        self, language: str, sign_collections: str
    ) -> Dict[str, List[str]]:
        with open(
            os.path.join(
                Settings.DATASET_ROOT_DIRECTORY,
                "sign_recordings",
                "collection_to_label_to_language_to_words.json",
            ),
            "r",
        ) as f:
            label_2_words = {
                f"{sc}_{label}": v[language]
                for sc in sign_collections
                for label, v in json.load(f)[sc].items()
            }

        return label_2_words

    def _load_constructable_words(
        self, language: str, sign_collections: str
    ) -> Dict[Tuple[str], List[str]]:
        with open(
            os.path.join(
                Settings.DATASET_ROOT_DIRECTORY,
                "sign_recordings",
                "collection_to_label_to_language_to_words.json",
            ),
            "r",
        ) as f:
            collection_to_label_to_language_to_words = json.load(f)

        with open(
            os.path.join(
                Settings.DATASET_ROOT_DIRECTORY,
                "sign_recordings",
                "organization_to_language_to_constructable_words.json",
            ),
            "r",
        ) as f:
            label_sequence_2_words = {
                tuple(d["components"]): d[language]
                for sc in sign_collections
                for d in json.load(f)["-".join(sc.split("-")[:2])]
                + [
                    v
                    for sc in sign_collections
                    for label, v in collection_to_label_to_language_to_words[sc].items()
                    if "components" in v
                ]
            }

        return label_sequence_2_words

    def _load_preprocessing(self, language: str) -> Dict:
        with open(
            os.path.join(
                Settings.DATASET_ROOT_DIRECTORY,
                "text_preprocessing.json",
            ),
            "r",
        ) as f:
            preprocessing_map = {k: v[language] for k, v in json.load(f).items()}

        return preprocessing_map

    def _load_sign_labels(self) -> Dict[str, List[str]]:
        with open(
            os.path.join(
                Settings.DATASET_ROOT_DIRECTORY,
                "sign_recordings",
                "recordings_labels.json",
            ),
            "r",
        ) as f:
            sign_labels = json.load(f)

        return sign_labels

    def _make_supported_words(self) -> Set[str]:
        vocab = set()
        vocab |= {w for words in self.label_sequence_to_words.values() for w in words}
        vocab |= {w for words in self.label_to_words.values() for w in words}

        # self.preprocessing_map["joint_word_to_split_words"],
        vocab |= set(self.preprocessing_map["person_names"])
        # self.preprocessing_map["named_entities"],
        vocab |= {
            w
            for w, n in self.preprocessing_map["words_to_numbers"].items()
            # Should be in TextLanguage. figure something out! delete from json?
            if "0" not in str(n)
        }
        # self.preprocessing_map["number_suffixes_to_zeros"],
        return vocab

    def _make_ambiguous_map(self, words: List[str]) -> Dict[str, List[str]]:
        ambiguous_2_context = dict()
        for word in words:
            without_context = self.remove_context(word)
            if without_context != word:
                if without_context not in ambiguous_2_context:
                    ambiguous_2_context[without_context] = []
                ambiguous_2_context[without_context].append(word)

        return ambiguous_2_context

    # load sign video/features

BAD_CHARACTERS_REGEX = r"|".join(
    {"\ufeff", "\u200b", "\u2005", "\u2009", "\u200b", "\u200c"}
    | {"\u0601", "\u2002", "\u2061", "\u202c", "\u3000", "\u200d"}
    | {"\u2003", "\u0602", "\xad", " ", "‎", "\u200f"}
)

