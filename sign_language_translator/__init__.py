"""# sign_language_translator

## Usage
```python
import sign_language_translator as slt

# download dataset
# slt.download_data("path/to")
slt.set_dataset_dir("path/to/sign-language-datasets")

# download translation models
# model = slt.download_model("...")
t2s_model = slt.get_model(
    task = "text-to-sign",
    approach = "concatenative", # for rule-based synthesis. "generative" for Deep Learning based synthesis.
    text_language = "Urdu",
    sign_language = "PakistanSignLanguage",
    sign_feature_model = "mediapipe_pose_v2_hand_v1",
)

sign_language_sentence = t2s_model.translate("hello world!")

# video = sign_language_sentence.video() # moviepy
# frame = sign_language_sentence.get_frame(index=10) # PIL Image
# slice = sign_language_sentence[10:11]
# features = sign_language_sentence.numpy()
# features = sign_language_sentence.torch()
sign_language_sentence.show()
```
"""

from .config.helpers import set_dataset_dir
from .config.settings import Settings
from . import vision, text, data_collection, models, utils, config

__all__ = [
    "set_dataset_dir",
    "Settings",
    "vision",
    "text",
    "data_collection",
    "models",
    "utils",
    "config",
]
