from sign_language_translator.models import ConcatenativeSynthesis


def test_concatenative_synthesis_model():
    text_lang = "urdu"
    sign_lang = "psl"
    sign_format = "mediapipe"

    model = ConcatenativeSynthesis(text_language=text_lang, sign_language=sign_lang, sign_format=sign_format)
    assert model.text_language.name() == text_lang
    assert model.sign_language is not None
    assert model.sign_features is not None

    model.translate("سیب اچھا ہے")
