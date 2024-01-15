"""
sign_language_translator
========================
Code: https://github.com/sign-language-translator/sign-language-translator
Help: https://sign-language-translator.readthedocs.io

This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.
The goal is to provide a user friendly API to novel Sign Language Translation solutions that can easily adapt to any regional sign language.

Usage
-----

.. code-block:: python

    import sign_language_translator as slt

    # download dataset or models (if you need them for personal use)
    # (by default, resources are auto-downloaded within the install directory)
    # slt.Assets.set_root_dir("path/to/folder")  # Helps preventing duplication across environments or using cloud synced data
    # slt.Assets.download(".*.json")  # downloads into resource_dir
    # print(slt.Settings.FILE_TO_URL.keys())  # All downloadable resources

    print("All available models:")
    print(list(slt.ModelCodes))  # slt.ModelCodeGroups
    # print(list(slt.TextLanguageCodes))
    # print(list(slt.SignLanguageCodes))
    # print(list(slt.SignFormatCodes))

    # -------------------------- TRANSLATE: text to sign --------------------------

    import sign_language_translator as slt

    # Load text-to-sign model
    # deep_t2s_model = slt.get_model("t2s-flan-T5-base-01.pt") # pytorch

    # rule-based model (concatenates clips of each word)
    t2s_model = slt.models.ConcatenativeSynthesis(
        text_language = "urdu", # or object of any child of slt.languages.text.text_language.TextLanguage class
        sign_language = "pakistan-sign-language", # or object of any child of slt.languages.sign.sign_language.SignLanguage class
        sign_format = "video", # or object of any child of slt.vision.sign.Sign class
    )

    text = "HELLO دنیا!" # HELLO treated as an acronym
    sign_language_sentence = t2s_model(text)

    # sign_language_sentence.show() # class: slt.vision.sign.Sign or its child
    # sign_language_sentence.save(f"sentences/{text}.mp4")

    # -------------------------- TRANSLATE: sign to text --------------------------

    import sign_language_translator as slt

    # # Load sign-to-text model (pytorch) (COMING SOON!)
    # translation_model = slt.get_model(slt.ModelCodes.Gesture)
    embedding_model = slt.models.MediaPipeLandmarksModel()

    sign = slt.Video("video.mp4")
    embedding = embedding_model.embed(sign.iter_frames())
    # text = translation_model.translate(embedding)

    # print(text)
    sign.show()
    # slt.Landmarks(embedding, connections="mediapipe-world").show()
"""

from sign_language_translator import config, languages, models, text, utils, vision
from sign_language_translator.config import enums
from sign_language_translator.config.assets import Assets
from sign_language_translator.config.enums import ModelCodeGroups, ModelCodes
from sign_language_translator.config.enums import SignFormats as SignFormatCodes
from sign_language_translator.config.enums import SignLanguages as SignLanguageCodes
from sign_language_translator.config.enums import TextLanguages as TextLanguageCodes
from sign_language_translator.config.settings import Settings
from sign_language_translator.languages import get_sign_language, get_text_language
from sign_language_translator.models import get_model
from sign_language_translator.vision._utils import get_sign_wrapper_class
from sign_language_translator.vision.video.video import Video

__version__ = config.utils.get_package_version()

__all__ = [
    # config
    "Assets",
    "Settings",
    "__version__",
    # modules
    "vision",
    "text",
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
    # classes (wrappers)
    "Video",
    # object loaders
    "get_sign_language",
    "get_text_language",
    "get_model",
    "get_sign_wrapper_class",
]
