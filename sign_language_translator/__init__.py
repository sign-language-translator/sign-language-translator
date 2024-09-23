"""
sign_language_translator
========================

Code: https://github.com/sign-language-translator/sign-language-translator
Help: https://slt.readthedocs.io
Demo: https://huggingface.co/sltAI

This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.
The goal is to provide a user friendly API to novel Sign Language Translation solutions that can easily adapt to any regional sign language.

Usage
-----

.. code-block:: python

    import sign_language_translator as slt

    # download dataset or models (if you need them for personal use)
    # (by default, resources are auto-downloaded within the install directory)
    # slt.Assets.set_root_dir("path/to/folder")  # Helps preventing duplication across environments or using cloud synced data
    # slt.Assets.download(r".*.json")  # downloads into asset_dir
    # print(slt.Assets.FILE_TO_URL.keys())  # All downloadable resources

    print("All available models:")
    print(list(slt.ModelCodes))  # slt.ModelCodeGroups
    # print(list(slt.TextLanguageCodes))
    # print(list(slt.SignLanguageCodes))
    # print(list(slt.SignFormatCodes))

    # -------------------------- TRANSLATE: text to sign --------------------------

    # The core model of the project (rule-based text-to-sign translator)
    # which enables us to generate synthetic training datasets
    model = slt.models.ConcatenativeSynthesis(
    text_language="urdu", sign_language="pk-sl", sign_format="video" )

    text = "یہ بہت اچھا ہے۔" # "This very good is."
    sign = model.translate(text) # tokenize, map, download & concatenate
    sign.show()


    model.text_language = slt.TextLanguageCodes.HINDI     # slt.languages.text.English()
    model.sign_format = slt.SignFormatCodes.LANDMARKS
    model.sign_embedding_model = "mediapipe-world"

    sign_2 = model.translate("कैसे हैं आप?") # "How are you?"
    sign_2.save("how-are-you.csv", overwrite=True)
    sign_2.save_animation("how-are-you.gif", overwrite=True)

    # -------------------------- TRANSLATE: sign to text --------------------------

    # sign = slt.Video("path/to/video.mp4")
    sign = slt.Video.load_asset("pk-hfad-1_aap-ka-nam-kya(what)-hy")  # your name what is? (auto-downloaded)
    sign.show_frames_grid()

    # Extract Pose Vector for feature reduction
    embedding_model = slt.models.MediaPipeLandmarksModel()      # pip install "sign_language_translator[mediapipe]"  # (or [all])
    embedding = embedding_model.embed(sign.iter_frames())

    slt.Landmarks(embedding.reshape((-1, 75, 5)),
                connections="mediapipe-world").show()

    # # Load sign-to-text model (pytorch) (COMING SOON!)
    # translation_model = slt.get_model(slt.ModelCodes.Gesture)
    # text = translation_model.translate(embedding)
    # print(text)
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
from sign_language_translator.vision.landmarks.landmarks import Landmarks
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
    "Landmarks",
    "Video",
    # object loaders / factory functions
    "get_sign_language",
    "get_text_language",
    "get_model",
    "get_sign_wrapper_class",
]
