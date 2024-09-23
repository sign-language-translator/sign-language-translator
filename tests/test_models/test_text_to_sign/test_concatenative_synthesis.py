import os

import pytest

from sign_language_translator.languages.sign.pakistan_sign_language import (
    PakistanSignLanguage,
)
from sign_language_translator.models.text_to_sign import ConcatenativeSynthesis
from sign_language_translator.vision.video.video import Video


def test_concatenative_synthesis_model():
    text_lang = "urdu"
    sign_lang = "pk-sl"
    sign_format = Video

    model = ConcatenativeSynthesis(
        text_language=text_lang, sign_language=sign_lang, sign_format=sign_format
    )
    assert model.text_language.name() == "ur"
    assert model.sign_language.name() == "pakistan-sign-language"
    assert model.sign_format.name() == "video"

    text = "ایک سیب اچھا ہے"
    sign_language_sentence = model.translate(text)

    # ubuntu container does not have h264 codec so OpenCV video writer will not work
    sign_language_sentence.save(path := "temp/sign.mp4", overwrite=True, codec="mp4v")
    assert os.path.exists(path), os.listdir("temp")

    text = "گھنٹے گھنٹہ"
    sign = model.translate(text)

    sign_language_sentence.save(path := "temp/sign_2.mp4", overwrite=True, codec="mp4v")
    assert os.path.exists(path), os.listdir("temp")

    model.sign_format = "landmarks"
    assert model.sign_format.name() == "landmarks"
    model.sign_embedding_model = "mediapipe-world"
    assert model.sign_embedding_model == "mediapipe-world"
    model.sign_language = PakistanSignLanguage()
    assert model.sign_language.name() == "pakistan-sign-language"

    # ==== Hindi ==== #
    model.text_language = "hindi"
    assert model.text_language.name() == "hi"

    text = "पाँच घंटे।"
    sign = model.translate(text)

    sign.save((path := f"temp/{text}.csv"), overwrite=True)
    with open(path, "r", encoding="utf-8") as f:
        assert len(f.read().splitlines()) > 1

    # ==== English ==== #
    model.text_language = "english"
    assert model.text_language.name() == "en"

    sign = model.translate("The Door closed with Noise.")

    sign.save((path := f"temp/{text}.csv"), overwrite=True)
    with open(path, "r", encoding="utf-8") as f:
        assert len(f.read().splitlines()) > 1


def test_concatenative_synthesis_validation():
    with pytest.raises(TypeError):
        _ = ConcatenativeSynthesis(
            text_language=1, sign_language="pk-sl", sign_format="video"  # type: ignore
        )

    with pytest.raises(TypeError):
        _ = ConcatenativeSynthesis(
            text_language="urdu", sign_language={}, sign_format="video"  # type: ignore
        )

    with pytest.raises(TypeError):
        _ = ConcatenativeSynthesis(
            text_language="urdu", sign_language="pk-sl", sign_format=list  # type: ignore
        )

    model = ConcatenativeSynthesis(
        text_language="urdu", sign_language="pk-sl", sign_format="video"
    )
    assert model.sign_embedding_model is None
    model.sign_embedding_model = None
    with pytest.raises(ValueError):
        model.sign_embedding_model = "something"
    model.sign_format = "landmarks"
    model.sign_embedding_model = "mediapipe-world"
    with pytest.raises(ValueError):
        model.sign_embedding_model = "something"
    with pytest.raises(ValueError):
        model.sign_embedding_model = None
    model._sign_embedding_model = None
    with pytest.raises(ValueError):
        _ = model.sign_embedding_model

    model._sign_language = None
    with pytest.raises(ValueError):
        _ = model.sign_language
    model._text_language = None
    with pytest.raises(ValueError):
        _ = model.text_language
    model._sign_format = None
    with pytest.raises(ValueError):
        _ = model.sign_format
