import json
import re
from collections import Counter
from typing import Dict, List

from sign_language_translator.config.assets import Assets
from sign_language_translator.config.settings import Settings
from sign_language_translator.languages.vocab import MappingDataset


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


def test_mapping_datasets():
    default_assets_dir = Assets.ROOT_DIR
    try:
        Assets.set_root_dir("./temp")
        paths = Assets.download(r".*-mapping.json", overwrite=True)
    except Exception as exc:
        Assets.set_root_dir(default_assets_dir)
        raise exc

    # assumes the dataset has correct format because otherwise it would fail during iteration hence no schema validation
    # schema_path = Assets.get_path("mapping-schema.json")[0]

    COUNTRY_PATTERN = r"^[a-z]{2}$"
    ORGANIZATION_PATTERN = r"^[a-z]{2,}$"
    WORD_SENSE_REGEX = r"\([^\(\)]*\)"
    VALID_WORD_SENSE_REGEX = r"^\([^ ]+\)$"  # TODO: improve regex ([\w-diacritics]+)
    all_labels = []
    all_components = []
    ambiguous_tokens: Dict[str, List[str]] = {}
    all_tokens: Dict[str, List[str]] = {}

    for path in paths:
        # load
        with open(path, "r", encoding="utf-8") as f:
            data: List[MappingDataset] = json.load(f)

        # validate
        for group in data:
            # check format: country & organization
            assert re.match(
                COUNTRY_PATTERN, group["country"]
            ), f"country: '{group['country']}' is not correct pattern in file: '{path}'"
            assert re.match(
                ORGANIZATION_PATTERN, group["organization"]
            ), f"organization: '{group['organization']}' is not correct pattern in file: '{path}'"

            for mapping in group["mapping"]:
                if path.endswith("-dictionary-mapping.json"):
                    if "label" in mapping:
                        # check direct link
                        assert Assets.get_url(
                            f"videos/{mapping['label']}.mp4"
                        ), f"no URL for label: {mapping['label']}"
                    # check presence: token
                    assert (
                        "token" in mapping
                    ), f"'token' key is not in mapping: '{mapping}' in dictionary dataset file: '{path}'"

                    # check ambiguous words
                    for lang, tokens in mapping["token"].items():
                        all_tokens.setdefault(lang, []).extend(tokens)
                        for token in tokens:
                            word_senses = re.findall(WORD_SENSE_REGEX, token)
                            for word_sense in word_senses:
                                # check format: word sense
                                assert re.match(
                                    VALID_WORD_SENSE_REGEX, word_sense
                                ), f"word sense: '{word_sense}' is not correct pattern, in mapping: '{mapping}' in file: '{path}'"
                            if word_senses:
                                ambiguous_tokens.setdefault(lang, []).append(token)
                else:
                    # check absence: token
                    assert (
                        "token" not in mapping
                    ), f"'token' key is in mapping: '{mapping}' in non-dictionary dataset file: '{path}'"
                    # check presence: gloss or translation
                    assert (
                        "gloss" in mapping or "translation" in mapping
                    ), f"'gloss' or 'translation' key is not in mapping: '{mapping}' in file: '{path}'"
                    # TODO: check archive link
                    # assert Assets.get_archive_url(f"videos/{mapping['label']}.mp4"), f"no URL for label: {mapping['label']}"

                if "label" in mapping:
                    # check format: label
                    validate_sign_label_pattern(mapping["label"])
                    all_labels.append(mapping["label"])
                elif "components" in mapping:
                    # check format: components
                    for component in mapping["components"]:
                        validate_sign_label_pattern(component)
                    all_components.append(tuple(mapping["components"]))
                else:
                    raise AssertionError(
                        f"'label' or 'components' key is not in mapping: '{mapping}' in file: '{path}'"
                    )

    # check for duplicates
    duplicate_labels = {k: v for k, v in Counter(all_labels).items() if v > 1}
    assert not duplicate_labels, f"duplicate labels: {duplicate_labels}"

    duplicate_components = {k: v for k, v in Counter(all_components).items() if v > 1}
    assert not duplicate_components, f"duplicate components: {duplicate_components}"

    # check for unknown components
    all_component_labels = {c for components in all_components for c in components}
    unknown_components = all_component_labels - set(all_labels)
    assert not unknown_components, f"unknown components: {unknown_components}"

    # check for unlabeled ambiguous tokens
    for (lang), tokens in ambiguous_tokens.items():
        ambiguous = {re.sub(WORD_SENSE_REGEX, "", tok) for tok in tokens}
        assert not (
            unlabeled := set(all_tokens[lang]) & ambiguous
        ), f"unlabeled ambiguous tokens: '{unlabeled}' in language: '{lang}'"

    # reset ROOT_DIR for remaining test-cases
    Assets.set_root_dir(default_assets_dir)


# def test_sample_data():
#     pass
