import json
import os
import re
import warnings
from collections import Counter
from typing import Dict, List

from sign_language_translator import Settings
from sign_language_translator.config.assets import Assets


def load_data(file_id):
    Assets.download(file_id, overwrite=False)

    data_path = Assets.get_path(file_id)[0]
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
    else:
        data = None

    return data


def load_recordings_labels() -> List[str]:
    json_data: Dict[str, List[str]] | None = load_data(
        "recordings_labels.json"
    )

    flattened_data = (
        [
            Settings.FILENAME_SEPARATOR.join((sign_collection, label))
            for sign_collection, label_list in json_data.items()
            for label in label_list
        ]
        if json_data
        else []
    )

    return flattened_data


def validate_sign_label_pattern(sign_label: str):
    assert re.match(
        (
            r"^[a-z]+"
            + re.escape(Settings.FILENAME_CONNECTOR)
            + r"[a-z]+"
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
    labels = load_recordings_labels()
    if not labels:
        warnings.warn("'recordings_labels' json file from dataset could not be loaded")
        return

    repeated = {lab: count for lab, count in Counter(labels).most_common() if count > 1}
    assert not repeated, f"repetition in reference_labels.json, {repeated}"

    # Assets.get_url(f".*/{re.escape(label)}.mp4"))
    not_linked = [lab for lab in labels if not Assets.get_url(f"videos/{lab}.mp4")]
    assert not not_linked, f"{len(not_linked)} labels not linked: {not_linked}"


def test_constructable_words():
    data = load_data("organization_to_language_to_constructable_words.json")
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
    data = load_data("collection_to_label_to_language_to_words.json")
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


# def test_repetition():
#     pass


# def test_ambiguous_words():
#     pass


# def test_word_senses():
#     pass


# def test_test_data():
#     pass
