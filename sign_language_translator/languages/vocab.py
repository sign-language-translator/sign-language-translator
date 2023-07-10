"""load word datasets to create word maps etc."""

import json
import os
import re
from typing import Any, Dict, Iterable, List, Set

from sign_language_translator.config.settings import Settings


class Vocab:
    """loads data from files for a specific language and specified sign_collections"""

    def __init__(
        self,
        language: str = r"$.",
        sign_collections: Iterable[str] = (r"$.",),
        data_root_dir: str = Settings.DATASET_ROOT_DIRECTORY,
        arg_is_regex: bool = True,
    ) -> None:
        # save arguments
        self.language = language
        self.sign_collections = sign_collections
        self.data_root_dir = data_root_dir
        self.arg_is_regex = arg_is_regex

        # source file paths
        self.label_to_words_path = os.path.join(
            self.data_root_dir,
            "sign_recordings",
            "collection_to_label_to_language_to_words.json",
        )
        self.constructed_words_path = os.path.join(
            self.data_root_dir,
            "sign_recordings",
            "organization_to_language_to_constructable_words.json",
        )
        self.preprocessing_path = os.path.join(
            self.data_root_dir,
            "text_preprocessing.json",
        )
        self.token_to_id_path = os.path.join(
            self.data_root_dir,
            "language_to_token_to_id.json",
        )

        # defaults
        self.word_sense_regex = r"\([^\(\)]*\)"  # everything in parenthesis
        self.word_to_labels: Dict[str, List[List[str]]] = {}
        self.token_to_id: Dict[str, int] = {}
        self.supported_words_with_word_sense: Set[str] = set()
        self.supported_words: Set[str] = set()
        self.ambiguous_to_unambiguous: Dict[str, List[str]] = {}
        self.ambiguous_to_unambiguous: Dict[str, List[str]] = {}

        # load jsons
        self.word_to_labels.update(
            self._make_word_to_labels(
                language=language,
                sign_collections=sign_collections,
                collection_to_label_to_language_to_words=self.__load_label_to_words(),
                organization_to_language_to_constructable_words=self.__load_constructable_words(),
                regex=arg_is_regex,
            )
        )
        self.token_to_id.update(self.__load_token_to_id(language, arg_is_regex))
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}

        # populate data from files
        self.supported_words_with_word_sense = set(self.word_to_labels)
        self.supported_words = {
            self.remove_word_sense(w) for w in self.supported_words_with_word_sense
        }
        self.ambiguous_to_unambiguous = self._make_disambiguation_map(
            self.supported_words_with_word_sense
        )
        self.labels: Set[str] = {
            label
            for label_sequences in self.word_to_labels.values()
            for label_sequence in label_sequences
            for label in label_sequence
        }

        preprocessing_map = self.__load_preprocessing(language)
        self.person_names: List[str] = preprocessing_map.get("person_names", [])
        self.words_to_numbers: Dict[str, int] = preprocessing_map.get(
            "words_to_numbers", {}
        )
        self.misspelled_to_correct: Dict[str, str] = preprocessing_map.get(
            "misspelled_to_correct", {}
        )
        self.number_suffix_to_zeros: Dict[str, str] = preprocessing_map.get(
            "number_suffixes_to_zeros", {}
        )
        self.joint_word_to_split_words: Dict[str, str] = preprocessing_map.get(
            "joint_word_to_split_words", {}
        )
        # TODO: improve coverage of key.isnumeric()
        self.numeric_keys: Set[str] = {
            key for key in self.word_to_labels if key.isnumeric()
        }

    def remove_word_sense(self, text: str) -> str:
        """Remove the word sense or disambiguation information from given text.

        Args:
            text (str): The text from which the word sense needs to be removed.

        Returns:
            str: The word without the word sense or disambiguation information.

        Example:
            word = "this is a spring(metal-coil). those are glasses(water-containers)."
            without_word_sense = remove_word_sense(word)
            print(without_word_sense)  # Output: "this is a spring. those are glasses."
        """

        without_word_sense = re.sub(self.word_sense_regex, "", text)
        return without_word_sense

    def __load_label_to_words(self):
        with open(
            self.label_to_words_path,
            "r",
            encoding="utf-8",
        ) as file_pointer:
            collection_to_label_to_language_to_words: Dict[
                str, Dict[str, Dict[str, List[str]]]
            ] = json.load(file_pointer)

        return collection_to_label_to_language_to_words

    def __load_constructable_words(self):
        with open(
            self.constructed_words_path,
            "r",
            encoding="utf-8",
        ) as file_pointer:
            organization_to_language_to_constructable_words: Dict[
                str, List[Dict[str, List[str]]]
            ] = json.load(file_pointer)

        return organization_to_language_to_constructable_words

    def __load_preprocessing(self, language: str, regex: bool = True) -> Dict[str, Any]:
        with open(
            self.preprocessing_path,
            "r",
            encoding="utf-8",
        ) as file_pointer:
            raw_data: Dict[str, Dict[str, Any]] = json.load(file_pointer)

        preprocessing_map: Dict[str, Any] = {
            key: lang_to_data.get(lang, self.__get_default_value(lang_to_data))
            for key, lang_to_data in raw_data.items()
            for lang in lang_to_data
            if self.__check_text_in_list(lang, [language], regex=regex)
        }

        return preprocessing_map

    def __load_token_to_id(self, language: str, regex: bool = True) -> Dict[str, int]:
        with open(
            self.token_to_id_path,
            "r",
            encoding="utf-8",
        ) as file_pointer:
            raw_data: Dict[str, Dict[str, int]] = json.load(file_pointer)

        token_to_id: Dict[str, int] = {
            key: val
            for lang, data in raw_data.items()
            for key, val in data.items()
            if self.__check_text_in_list(lang, [language], regex=regex)
        }

        return token_to_id

    def _make_word_to_labels(
        self,
        language: str,
        sign_collections: Iterable[str],
        collection_to_label_to_language_to_words: Dict[
            str, Dict[str, Dict[str, List[str]]]
        ],
        organization_to_language_to_constructable_words: Dict[
            str, List[Dict[str, List[str]]]
        ],
        regex: bool = True,
    ):
        """takes json word mapping datasets and creates a dict mapping text tokens to sign_labels

        Args:
            language (str | None, optional): language code used in json or a regex matching it whose data should be extracted. Defaults to None.
            sign_collections (Iterable[str] | None, optional): all sign collections to extract from json or regex matching them. Defaults to None.
            regex (bool, optional): treat the provided language code and sign_collections as regex when comparing against json keys. Defaults to True.

        Returns:
            Dict[str, List[List[str]]]: maps each text token to a list containing sequences of signs
        """
        word_to_labels: Dict[str, List[List[str]]] = {}

        # simple word map
        for (
            sign_coll,
            label_to_language_to_words,
        ) in collection_to_label_to_language_to_words.items():
            if not self.__check_text_in_list(sign_coll, sign_collections, regex=regex):
                continue
            for label, language_to_words in label_to_language_to_words.items():
                for lang, words in language_to_words.items():
                    if lang == "components" or not self.__check_text_in_list(
                        lang, [language], regex=regex
                    ):
                        continue
                    for word in words:
                        if word not in word_to_labels:
                            word_to_labels[word] = []

                        word_to_labels[word].append(
                            [f"{sign_coll}{Settings.FILENAME_SEPARATOR}{label}"]
                        )
                        if "components" in language_to_words:
                            # all components are from specified sign collections only
                            if all(
                                self.__check_text_in_list(
                                    comp.split(Settings.FILENAME_SEPARATOR)[0],
                                    sign_collections,
                                    regex=regex,
                                )
                                for comp in language_to_words["components"]
                            ):
                                word_to_labels[word].append(
                                    language_to_words["components"]
                                )

        # constructable words
        for (
            sign_coll,
            list_of_language_to_words,
        ) in organization_to_language_to_constructable_words.items():
            if not self.__check_text_in_list(sign_coll, sign_collections, regex=regex):
                continue
            for language_to_words in list_of_language_to_words:
                for lang, words in language_to_words.items():
                    if lang == "components" or not (
                        self.__check_text_in_list(lang, [language], regex=regex)
                        and "components" in language_to_words
                    ):
                        continue

                    for word in words:
                        if word not in word_to_labels:
                            word_to_labels[word] = []
                        word_to_labels[word].append(language_to_words["components"])

        # drop duplicates (when language = r".*", digits etc are repeated)
        word_to_labels = {
            word: list(list(seq) for seq in set(tuple(seq) for seq in label_seqs))
            for word, label_seqs in word_to_labels.items()
        }

        return word_to_labels

    def __check_text_in_list(
        self,
        text: str,
        list_of_texts: Iterable[str],
        regex: bool = True,
    ):
        if not (list_of_texts and all(list_of_texts)):
            return False

        return (
            any(re.match(pattern, text) for pattern in list_of_texts)
            if regex
            else text in list_of_texts
        )

    def __get_default_value(self, dictionary: Dict) -> Any:
        default_obj = None

        if dictionary:
            # get an empty instance of first value's class
            first_value = list(dictionary.values())[0]
            value_class = type(first_value)
            default_obj = value_class()

        return default_obj

    def _make_disambiguation_map(self, words: Iterable[str]) -> Dict[str, List[str]]:
        ambiguous_2_unambiguous = {}
        for word in words:
            without_word_sense = self.remove_word_sense(word)
            if without_word_sense != word:
                if without_word_sense not in ambiguous_2_unambiguous:
                    ambiguous_2_unambiguous[without_word_sense] = []
                ambiguous_2_unambiguous[without_word_sense].append(word)

        return ambiguous_2_unambiguous

    # load sign video/features


__all__ = [
    "Vocab",
]
