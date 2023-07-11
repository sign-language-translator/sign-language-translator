"""# [sign_language_translator](https://github.com/sign-language-translator/sign-language-translator)
This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.
The goal is to provide a user friendly API to novel Sign Language Translation solutions that can easily adapt to any regional sign language.

## Usage

```python
import sign_language_translator as slt

# download dataset (by default, dataset is downloaded within the install directory)
# slt.set_dataset_dir("path/to/sign-language-datasets") # optional
# slt.download("id")

# ------------------------------------------

# Load text-to-sign model
# deep_t2s_model = slt.get_model("generative_t2s_base-01") # pytorch
# rule-based model (concatenates clips of each word)
t2s_model = slt.get_model(
    model_code = "ConcatenativeSynthesis", # slt.enums.ModelCodes.CONCATENATIVE_SYNTHESIS.value
    text_language = "English", # or object of any child of slt.languages.text.text_language.TextLanguage class
    sign_language = "PakistanSignLanguage", # or object of any child of slt.languages.sign.sign_language.SignLanguage class
    sign_feature_model = "mediapipe_pose_v2_hand_v1",
)

text = "hello world!"
sign_language_sentence = t2s_model(text)

# moviepy_video_object = sign_language_sentence.video()
# moviepy_video_object.ipython_display()
# moviepy_video_object.write_videofile(f"sentences/{text}.mp4")

# ------------------------------------------

# load sign
video = slt.read_video("video.mp4")
# features = slt.extract_features(video, "mediapipe_pose_v2_hand_v1")

# Load sign-to-text model
deep_s2t_model = slt.get_model("gesture_mp_base-01") # pytorch
features = deep_s2t_model.extract_features(video)

# translate
text = deep_s2t_model(features)
print(text)
```
"""

from sign_language_translator import (
    config,
    data_collection,
    languages,
    models,
    text,
    utils,
    vision,
)
from sign_language_translator.config import enums
from sign_language_translator.config.settings import Settings
from sign_language_translator.config.helpers import set_dataset_dir
from sign_language_translator.languages import get_sign_language, get_text_language
from sign_language_translator.models import get_model

__all__ = [
    "set_dataset_dir",
    "Settings",
    "vision",
    "text",
    "data_collection",
    "models",
    "languages",
    "utils",
    "config",
    "enums",
    # object loaders
    "get_sign_language",
    "get_text_language",
    "get_model"
    # "get_feature_model"
]
