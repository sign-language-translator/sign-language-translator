import json
import os
import re
import warnings
from glob import glob
from typing import Dict, List

from sign_language_translator import Settings


def load_data(data_filename):
    # :TODO: clean up this commented code:
    # from sign_language_translator import set_dataset_dir

    # set_dataset_dir(
    #     "/Users/mudassar.iqbal/Library/CloudStorage/GoogleDrive-mdsriqb@gmail.com/My Drive/sign-language-translator/sign-language-datasets"
    # )

    data = None
    data_path = os.path.join(
        Settings.DATASET_ROOT_DIRECTORY,
        data_filename,
    )
    data_path = os.path.realpath(data_path)
    if os.path.exists(data_path):
        with open(data_path) as f:
            data = json.load(f)

    return data


def load_recordings_labels() -> List[str]:
    json_data: Dict[str, List[str]] = load_data(
        os.path.join(
            "sign_recordings",
            "recordings_labels.json",
        )
    )
    if not json_data:
        return []

    json_data = [
        Settings.FILENAME_SEPARATOR.join((sign_collection, label))
        for sign_collection, label_list in json_data.items()
        for label in label_list
    ]

    return json_data


def validate_sign_label_pattern(sign_label: str):
    assert re.match(
        (
            r"^\w+"
            + re.escape(Settings.FILENAME_CONNECTOR)
            + r"\w+"
            + re.escape(Settings.FILENAME_CONNECTOR)
            + r"[0-9]+"
            + re.escape(Settings.FILENAME_SEPARATOR)
            + r"[^ "
            + re.escape(Settings.FILENAME_SEPARATOR)
            + r"]+$"
        ),
        sign_label,
    ), "sign label is not correct pattern"


def validate_word_dict(word_dict: Dict[str, List[str]], sign_labels=set()):
    assert isinstance(word_dict, dict)
    assert len(word_dict) > 0, "no languages in the word_dict"
    for lang in word_dict:
        # type checking
        assert isinstance(lang, str)
        assert isinstance(word_dict[lang], list)
        assert len(word_dict[lang]) > 0, "no words in the word_list"
        assert all(
            [isinstance(word, str) for word in word_dict[lang]]
        ), "word list contains non-strings"

        # value checking
        # assert lang in list(Languages) + ["components"]
        if lang == "components":
            assert (
                len(word_dict) > 1
            ), "gotta have some 'text-language' beside 'components' key"
            for label in word_dict[lang]:
                assert (
                    label in sign_labels
                ), "component sign labels was not found in recordings_labels.json"
            assert (
                len(word_dict[lang]) > 1
            ), "gotta have more than 1 component otherwise its just repetition"


def test_recordings_labels():
    json_data = load_recordings_labels()
    if not json_data:
        warnings.warn("'recordings_labels' json file from dataset could not be loaded")
        return

    filepaths = glob(
        os.path.join(
            Settings.DATASET_ROOT_DIRECTORY,
            "sign_recordings",
            "reference_clips",
            "**",
            "*.mp4",
        )
    )
    # :TODO: [-1][:-4] after files renamed to sign-collection_label_person_camera.mp4
    file_labels = [
        (Settings.FILENAME_SEPARATOR.join(p.split(os.sep)[-2:]))[:-4] for p in filepaths
    ]

    assert len(file_labels) == len(
        set(file_labels)
    ), "repetition in reference clip filenames"
    assert len(json_data) == len(set(json_data)), "repetition in reference_labels.json"
    assert set(json_data) == set(
        file_labels
    ), f"reference clips filenames not equal reference_labels.json {set(file_labels).symmetric_difference(set(json_data))}"


def test_constructable_words():
    data = load_data(
        os.path.join(
            "sign_recordings",
            "organization_to_language_to_constructable_words.json",
        )
    )
    if not data:
        warnings.warn(
            "'constructable_words' json file from dataset could not be loaded"
        )
        return
    sign_labels = set(load_recordings_labels())
    assert len(sign_labels) > 0, "check recordings_labels.json"

    for sign_collection in data:
        assert bool(
            re.match(
                r"^\w+" + re.escape(Settings.FILENAME_CONNECTOR) + r"\w+$",
                sign_collection,
            )
        ), "bad 'country-organization' key"

        for word_dict in data[sign_collection]:
            validate_word_dict(word_dict, sign_labels=sign_labels)


def test_label_to_words():
    data = load_data(
        os.path.join(
            "sign_recordings",
            "collection_to_label_to_language_to_words.json",
        )
    )
    if not data:
        warnings.warn("'label_to_words' json file from dataset could not be loaded")
        return
    sign_labels = set(load_recordings_labels())
    assert len(sign_labels) > 0, "check recordings_labels.json"

    for sign_collection in data:
        if sign_collection != "wordless":
            assert bool(
                re.match(
                    r"^\w+"
                    + re.escape(Settings.FILENAME_CONNECTOR)
                    + r"\w+"
                    + re.escape(Settings.FILENAME_CONNECTOR)
                    + r"[0-9]+$",
                    sign_collection,
                )
            ), "bad 'country-organization-number' key"

        for sign_label, word_dict in data[sign_collection].items():
            assert (
                Settings.FILENAME_SEPARATOR.join((sign_collection, sign_label))
                in sign_labels
            ), ""
            validate_word_dict(word_dict, sign_labels=sign_labels)


def test_repetition():
    pass


def test_ambiguous_words():
    pass


def test_word_senses():
    pass


def test_test_data():
    pass
