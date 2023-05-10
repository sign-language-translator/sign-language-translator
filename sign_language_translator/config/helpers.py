import os
from .settings import Settings
from typing import Union, List


def set_dataset_dir(path: str) -> None:
    assert os.path.isdir(path), "the provided path is not a directory"

    Settings.DATASET_ROOT_DIRECTORY = path
    # trigger an event


def get_model(task: str, sign_language, text_language, video_feature_model: str, approach):
    
    if task in ["tts", "text-to-sign", "text_to_sign"]:
        pass
    elif task in ["stt", "sign-to-text", "sign_to_text"]:
        pass
    else:
        return None
