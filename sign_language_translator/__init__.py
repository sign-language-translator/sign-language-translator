"""Main __init__ file. Defines global variables & constants.
"""

import enum
import os

DATASET_ROOT_DIRECTORY = "/"

def set_dataset_dir(path: str) -> None:
    assert os.path.isdir(path)
    global DATASET_ROOT_DIRECTORY
    DATASET_ROOT_DIRECTORY = path

SIGN_RECORDINGS_DATASET_DIRECTORY = os.path.join(
    DATASET_ROOT_DIRECTORY, "sign_recordings"
)
TEXT_CORPORA_DATASET_DIRECTORY = os.path.join(DATASET_ROOT_DIRECTORY, "text_corpora")


class Country(enum.Enum):
    PAKISTAN = "pk"


class Organization(enum.Enum):
    HFAD = "hfad"
    # NICE = "nice"
    # FESF = "fesf"


class TextualLanguage(enum.Enum):
    ENGLISH = "english"
    URDU = "urdu"


class SignCollection(enum.Enum):
    PK_HFAD_1 = "pk-hfad-1"


# INTERNATIONAL_LANGUAGE = TextualLanguage.ENGLISH
# NATIVE_LANGUAGE = TextualLanguage.URDU


# def set_native_language(language: TextualLanguage):
#     """the regional spoken/textual language to be used throughout the project."""
#     global NATIVE_LANGUAGE
#     NATIVE_LANGUAGE = language

# def set_international_language(language: TextualLanguage):
#     """the international spoken/textual language to be used throughout the project."""
#     global INTERNATIONAL_LANGUAGE
#     INTERNATIONAL_LANGUAGE = language