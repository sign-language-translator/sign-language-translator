"""File-handling to create word maps.
"""

import json
import os
import re
from typing import Any, Dict, List, Set, Tuple, Union

from sign_language_translator.config.settings import Settings


class Vocab:
    """loads data from files for a specific language and specified sign_collections"""

    def __init__(
        self,
        language: str,
        sign_collections: List[str] | None = None,
        data_root_dir: str = Settings.DATASET_ROOT_DIRECTORY,
    ) -> None:
        # save arguments
        self.language = language
        self.sign_collections = sign_collections
        self.data_root_dir = data_root_dir
        # :TODO: regex matching of sign_collections e.g. pk-*-*

        # read files
        self.sign_labels = self._load_sign_labels(sign_collections)
        self.preprocessing_map = self._load_preprocessing(self.language)
        self.label_to_words = self._load_label_to_words(language, sign_collections)
        self.label_sequence_to_words = self._load_constructable_words(
            language, sign_collections
        )

        # create vocab
        self.WORD_SENSE_REGEX = r"\([^\(\)]*\)"  # all occurrences of everything wrapped in a pair of parenthesis
        self.supported_words_with_word_sense = self._make_supported_words()
        self.supported_words = {
            self.remove_word_sense(word) for word in self.supported_words_with_word_sense
        }
        self.ambiguous_to_unambiguous = self._make_disambiguation_map(
            self.supported_words_with_word_sense
        )
        self.person_names = self.preprocessing_map.get('person_names',[])
        self.words_to_numbers = self.preprocessing_map.get('words_to_numbers', {})

    def get_word_sense(self, word: str) -> str:
        word_sense = re.match(self.WORD_SENSE_REGEX, word)
        word_sense = word_sense.group() if word_sense else ""

        return word_sense

    def remove_word_sense(self, word: str) -> str:
        without_word_sense = re.sub(self.WORD_SENSE_REGEX, "", word)
        return without_word_sense

    def _load_label_to_words(
        self, language: str, sign_collections: List[str]
    ) -> Dict[str, List[str]]:
        with open(
            os.path.join(
                self.data_root_dir,
                "sign_recordings",
                "collection_to_label_to_language_to_words.json",
            ),
            "r",
        ) as f:
            raw_data: Dict[str, Dict[str, Dict[str, List[str]]]] = json.load(f)
            label_2_words = {
                f"{sc}{Settings.FILENAME_SEPARATOR}{label}": lang_to_word_list[language]
                for sc in sign_collections or raw_data.keys()
                for label, lang_to_word_list in raw_data.get(sc, {}).items()
                if language in lang_to_word_list
            }

        return label_2_words

    def _load_constructable_words(
        self, language: str, sign_collections: str
    ) -> Dict[Tuple[str], List[str]]:
        with open(
            os.path.join(
                self.data_root_dir,
                "sign_recordings",
                "collection_to_label_to_language_to_words.json",
            ),
            "r",
        ) as f:
            collection_to_label_to_language_to_words: Dict[
                str, Dict[str, Dict[str, List[str]]]
            ] = json.load(f)

        with open(
            os.path.join(
                self.data_root_dir,
                "sign_recordings",
                "organization_to_language_to_constructable_words.json",
            ),
            "r",
        ) as f:
            raw_data: Dict[str, Dict[str, List[str]]] = json.load(f)
            label_sequence_2_words = {
                tuple(sign_dict["components"]): sign_dict[language]
                for sc in sign_collections or raw_data.keys()
                for sign_dict in raw_data.get(
                    Settings.FILENAME_CONNECTOR.join(
                        sc.split(Settings.FILENAME_CONNECTOR)[:2]
                    ),
                    [],
                )
                + [
                    v
                    for sc in sign_collections
                    or collection_to_label_to_language_to_words.keys()
                    for label, v in collection_to_label_to_language_to_words.get(
                        sc, {}
                    ).items()
                    if "components" in v
                ]
                if language in sign_dict
                and (
                    all(
                        [
                            comp.split(Settings.FILENAME_SEPARATOR)[0] in sc
                            for comp in sign_dict["components"]
                        ]
                    )
                    if sign_collections
                    else True
                )
            }

        return label_sequence_2_words

    def _load_preprocessing(self, language: str) -> Dict[str, Any]:
        with open(
            os.path.join(
                self.data_root_dir,
                "text_preprocessing.json",
            ),
            "r",
        ) as f:
            raw_data: Dict[str, Dict[str, Any]] = json.load(f)
            preprocessing_map: Dict[str, Any] = {
                k: v.get(language, type(list(v.values())[0])())
                for k, v in raw_data.items()
            }

        return preprocessing_map

    def _load_sign_labels(
        self, sign_collections: List[str] | None = None
    ) -> Dict[str, List[str]]:
        with open(
            os.path.join(
                self.data_root_dir,
                "sign_recordings",
                "recordings_labels.json",
            ),
            "r",
        ) as f:
            sign_labels: Dict[str, List[str]] = json.load(f)
            sign_labels = [
                f"{sign_collection}{Settings.FILENAME_SEPARATOR}{label}"
                for sign_collection, label_list in sign_labels.items()
                for label in label_list
                if sign_collection in (sign_collections or sign_labels.keys())
            ]

        return sign_labels

    def _make_supported_words(self) -> Set[str]:
        vocab = set()
        vocab |= {w for words in self.label_sequence_to_words.values() for w in words}
        vocab |= {w for words in self.label_to_words.values() for w in words}

        # self.preprocessing_map["joint_word_to_split_words"],
        vocab |= set(self.preprocessing_map.get("person_names", {}))
        # self.preprocessing_map["named_entities"],
        vocab |= {
            w
            for w, n in self.preprocessing_map.get("words_to_numbers", {}).items()
            # :TODO: Should be in TextLanguage. figure something out! delete from json?
            if "0" not in str(n)
        }
        # self.preprocessing_map["number_suffixes_to_zeros"],
        return vocab

    def _make_disambiguation_map(self, words: List[str]) -> Dict[str, List[str]]:
        ambiguous_2_unambiguous = dict()
        for word in words:
            without_word_sense = self.remove_word_sense(word)
            if without_word_sense != word:
                if without_word_sense not in ambiguous_2_unambiguous:
                    ambiguous_2_unambiguous[without_word_sense] = []
                ambiguous_2_unambiguous[without_word_sense].append(word)

        return ambiguous_2_unambiguous

    # load sign video/features


BAD_CHARACTERS_REGEX = r"|".join(
    {"\ufeff", "\u200b", "\u2005", "\u2009", "\u200b", "\u200c"}
    | {"\u0601", "\u2002", "\u2061", "\u202c", "\u3000", "\u200d"}
    | {"\u2003", "\u0602", "\xad", " ", "‎", "\u200f"}
)
