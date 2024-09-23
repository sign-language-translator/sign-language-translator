"""load word datasets to create word maps etc."""

import json
import os
import re
from typing import Any, Dict, Iterable, List, Set, TypedDict

from sign_language_translator.config.assets import Assets
from sign_language_translator.config.settings import Settings

__all__ = [
    "Vocab",
    "MappingDataset",
]


# =================================== #
#    Mapping Dataset Static Typing    #
# =================================== #


class Mapping(TypedDict):
    components: List[str]
    """list of sign videos that can make up the video identified by label"""
    label: str
    """the filename of the sign video"""
    token: Dict[str, List[str]]
    """maps language codes to list of tokens that correspond to the label"""
    gloss: Dict[str, List[str]]
    """maps language codes to list of glosses that correspond to the video. A gloss is a word-for-word translation of the signs in the video."""
    translation: Dict[str, List[str]]
    """maps language codes to list of translations that correspond to the video. A translation is a grammatically correct text that has the same meaning as the sign sequence in the video."""


class MappingDataset(TypedDict):
    country: str
    description: str
    mapping: List[Mapping]
    organization: str
    url: str


# ================= #
#    Data Reader    #
# ================= #


class Vocab:
    """Loads text datasets for a specific language, country and organization.

    Note:
        Our mapping datasets will only be downloaded automatically if the `data_root_dir` arg is the same as `Assets.ROOT_DIR`.
    """

    def __init__(
        self,
        language: str = r".^",
        country: str = r".^",
        organization: str = r".^",
        part_number: str = r".^",
        data_root_dir: str = Assets.ROOT_DIR,
        arg_is_regex: bool = True,
        word_sense_regex: str = r"\([^\(\)]*\)",
    ) -> None:
        # save arguments
        self.language = language
        self.country = country
        self.organization = organization
        self.part_number = part_number
        self.data_root_dir = os.path.abspath(data_root_dir)
        self.arg_is_regex = arg_is_regex
        self.word_sense_regex = word_sense_regex

        # initialize properties with defaults
        self.word_to_labels: Dict[str, List[List[str]]] = {}
        self.supported_tokens: Set[str] = set()
        self.ambiguous_to_unambiguous: Dict[str, List[str]] = {}
        self.person_names: List[str] = []
        self.words_to_numbers: Dict[str, int] = {}
        self.misspelled_to_correct: Dict[str, str] = {}
        self.number_suffix_to_zeros: Dict[str, str] = {}
        self.joint_word_to_split_words: Dict[str, str] = {}
        self.numeric_keys: Set[str] = set()

        # load data
        self.__load_mapping_datasets()
        self.__load_preprocessing()

    def remove_word_sense(self, text: str) -> str:
        """Remove the word sense or disambiguation information from given text.

        Args:
            text (str): The text from which the word sense needs to be removed.

        Returns:
            str: The word without the word sense or disambiguation information.

        Example:

        .. code-block:: python

            word = "this is a spring(metal-coil). those are glasses(water-containers)."
            without_word_sense = remove_word_sense(word)
            print(without_word_sense)  # Output: "this is a spring. those are glasses."
        """

        without_word_sense = re.sub(self.word_sense_regex, "", text)
        return without_word_sense

    def __load_preprocessing(self) -> None:
        self.__download_resource(fname := "text-preprocessing.json")
        with open(os.path.join(self.data_root_dir, fname), "r", encoding="utf-8") as f:
            raw_data: Dict[str, Dict[str, Any]] = json.load(f)

        data: Dict[str, Any] = {
            key: lang_to_data.get(lang, self.__default_value(lang_to_data))
            for key, lang_to_data in raw_data.items()
            for lang in lang_to_data
            if self.__match(self.language, lang, self.arg_is_regex)
        }

        self.person_names: List[str] = data.get("person_names", [])
        self.words_to_numbers: Dict[str, int] = data.get("words_to_numbers", {})
        self.misspelled_to_correct: Dict[str, str] = data.get(
            "misspelled_to_correct", {}
        )
        self.number_suffix_to_zeros: Dict[str, str] = data.get(
            "number_suffixes_to_zeros", {}
        )
        self.joint_word_to_split_words: Dict[str, str] = data.get(
            "joint_word_to_split_words", {}
        )
        # TODO: improve coverage of key.isnumeric()
        self.numeric_keys: Set[str] = {
            key for key in self.word_to_labels if key.isnumeric()
        }

    def __load_mapping_datasets(self) -> None:
        # download conditionally
        self.__download_resource(
            filename := f"{self.country.rstrip('$')}-dictionary-mapping.json"
        )

        # load existing
        mapping_filepaths = [
            os.path.join(self.data_root_dir, file)
            for file in os.listdir(self.data_root_dir)
            if re.match(filename, file)
        ]
        for filepath in mapping_filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                self.word_to_labels.update(
                    self._make_word_to_labels(
                        self.language,
                        self.country,
                        self.organization,
                        self.part_number,
                        json.load(f),
                    )
                )

        self.supported_tokens = set(self.word_to_labels)
        self.ambiguous_to_unambiguous = self._make_disambiguation_map(
            self.supported_tokens
        )
        self.labels: Set[str] = {
            label
            for label_sequences in self.word_to_labels.values()
            for label_sequence in label_sequences
            for label in label_sequence
        }

    def _make_word_to_labels(
        self,
        language: str,
        country: str,
        organization: str,
        part_number: str,
        mapping_datasets: List[MappingDataset],
        is_regex: bool = True,
    ) -> Dict[str, List[List[str]]]:
        """
        Takes JSON word mapping datasets and creates a dictionary mapping text tokens to sign labels.

        Args:
            language (str, optional): Language code used in JSON or a regex matching it whose data should be extracted. Defaults to None.
            country (str): Country code used to filter the mapping datasets.
            organization (str): Organization code used to filter the mapping datasets.
            part_number (str): Part number used to filter the mapping datasets.
            mapping_datasets (List[MappingDataset]): List of mapping datasets.
            is_regex (bool, optional): Treat the provided language code, country, organization and part_number as regex when comparing against JSON keys. Defaults to True.

        Returns:
            Dict[str, List[List[str]]]: A dictionary that maps each text token to a list containing sequences of signs.
        """

        word_to_labels: Dict[str, List[List[str]]] = {}

        for dataset in mapping_datasets:
            # different country or organization
            if not (
                self.__match(country, dataset["country"], is_regex)
                and self.__match(organization, dataset["organization"], is_regex)
            ):
                continue

            for mapping in dataset["mapping"]:
                # different part number
                if not all(
                    self.__match(
                        part_number, label.split("_")[0].split("-")[-1], is_regex
                    )
                    for label in ([mapping["label"]] if "label" in mapping else [])
                    + mapping.get("components", [])
                ):
                    continue

                for lang, token_list in mapping.get("token", {}).items():
                    # different language
                    if not self.__match(language, lang, is_regex):
                        continue

                    for token in token_list:
                        if token not in word_to_labels:
                            word_to_labels[token] = []
                        if "label" in mapping:
                            word_to_labels[token].append([mapping["label"]])
                        if "components" in mapping:
                            # ???? Why filter down the components??????
                            if all(
                                self.__match(country, x[0], is_regex)
                                and self.__match(organization, x[1], is_regex)
                                for comp in mapping["components"]
                                if (
                                    x := comp.split(Settings.FILENAME_SEPARATOR)[
                                        0
                                    ].split(Settings.FILENAME_CONNECTOR)
                                )
                            ):
                                word_to_labels[token].append(mapping["components"])

        # Drop duplicates (when language = r".*", digits, etc. are repeated)
        word_to_labels = {
            word: list(list(seq) for seq in set(tuple(seq) for seq in label_seqs))
            for word, label_seqs in word_to_labels.items()
        }

        return word_to_labels

    def _make_disambiguation_map(self, words: Iterable[str]) -> Dict[str, List[str]]:
        """create a mapping from ambiguous words to possible unambiguous words

        Args:
            words (Iterable[str]): A list of disambiguated words. A disambiguated word is a word that has a word-sense in it. e.g. ["spring(season)", "spring(bouncy-coil)"].

        Returns:
            Dict[str, List[str]]: A dictionary mapping ambiguous words to possible unambiguous words. e.g. {"spring": ["spring(season)", "spring(bouncy-coil)"]}.
        """
        ambiguous_2_unambiguous = {}
        for word in words:
            without_word_sense = self.remove_word_sense(word)
            if without_word_sense != word:
                if without_word_sense not in ambiguous_2_unambiguous:
                    ambiguous_2_unambiguous[without_word_sense] = []
                ambiguous_2_unambiguous[without_word_sense].append(word)

        return ambiguous_2_unambiguous

    def __match(self, text_1: str, text_2: str, text_1_is_regex: bool) -> bool:
        return bool(re.match(text_1, text_2)) if text_1_is_regex else text_1 == text_2

    def __default_value(self, dictionary: Dict) -> Any:
        """get an empty instance of dict's first value's class"""
        first_value = next(iter(dictionary.values()), None)
        value_class = type(first_value)
        default_obj = value_class()

        return default_obj

    def __download_resource(self, file_name: str):
        if (
            Settings.AUTO_DOWNLOAD
            and not os.path.exists(os.path.join(self.data_root_dir, file_name))
            and self.data_root_dir == Assets.ROOT_DIR
        ):
            Assets.download(file_name, overwrite=False)
