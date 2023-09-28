"""# [sign_language_translator](https://github.com/sign-language-translator/sign-language-translator)
This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.
The goal is to provide a user friendly API to novel Sign Language Translation solutions that can easily adapt to any regional sign language.

## Usage

```python
import sign_language_translator as slt

# download dataset or models (if you need them for personal use)
# (by default, resources are auto-downloaded within the install directory)
# slt.set_resource_dir("path/to/folder")  # Helps preventing duplication across environments or using cloud synced data
# slt.utils.download_resource(".*.json")  # downloads into resource_dir
# print(slt.Settings.FILE_TO_URL.keys())  # All downloadable resources

print("All available models:")
print(list(slt.ModelCodes))  # slt.ModelCodeGroups
# print(list(slt.TextLanguageCodes))
# print(list(slt.SignLanguageCodes))
# print(list(slt.SignFormatCodes))

# -------------------------- TRANSLATE: text to sign --------------------------

# Load text-to-sign model
# deep_t2s_model = slt.get_model("t2s-flan-T5-base-01.pt") # pytorch
# rule-based model (concatenates clips of each word)
t2s_model = slt.get_model(
    model_code = "concatenative-synthesis", # slt.ModelCodes.CONCATENATIVE_SYNTHESIS
    text_language = "urdu", # or object of any child of slt.languages.text.text_language.TextLanguage class
    sign_language = "PakistanSignLanguage", # or object of any child of slt.languages.sign.sign_language.SignLanguage class
    sign_feature_model = "mediapipe_pose_v2_hand_v1",
)

text = "HELLO دنیا!" # HELLO treated as an acronym
sign_language_sentence = t2s_model(text)

# slt_video_object = sign_language_sentence.video()
# slt_video_object.ipython_display()
# slt_video_object.save(f"sentences/{text}.mp4")

# -------------------------- TRANSLATE: sign to text --------------------------

# load sign
video = slt.Video("video.mp4")
# features = slt.extract_features(video, "mediapipe_pose_v2_hand_v1")

# Load sign-to-text model
deep_s2t_model = slt.get_model("gesture_mp_base-01") # pytorch

# translate via single call to pipeline
# text = deep_s2t_model.translate(video)

# translate via individual steps
features = deep_s2t_model.extract_features(video.iter_frames())
logits = deep_s2t_model(features)
tokens = deep_s2t_model.decode(logits)
text = deep_s2t_model.detokenize(tokens)

print(features.shape)
print(logits.shape)
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
from sign_language_translator.config.enums import (
    ModelCodes,
    ModelCodeGroups,
    TextLanguages as TextLanguageCodes,
    SignFormats as SignFormatCodes,
    SignLanguages as SignLanguageCodes,
)
from sign_language_translator.config.settings import Settings, set_resources_dir
from sign_language_translator.languages import get_sign_language, get_text_language
from sign_language_translator.models import get_model

__version__ = config.helpers.get_package_version()

__all__ = [
    # config
    "set_resources_dir",
    "Settings",
    "__version__",
    # modules
    "vision",
    "text",
    "data_collection",
    "models",
    "languages",
    "utils",
    "config",
    "enums",
    # classes (enum)
    "ModelCodes",
    "TextLanguageCodes",
    "SignLanguageCodes",
    "SignFormatCodes",
    "ModelCodeGroups",
    # object loaders
    "get_sign_language",
    "get_text_language",
    "get_model"
]
