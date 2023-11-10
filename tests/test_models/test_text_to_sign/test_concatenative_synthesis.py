from sign_language_translator.models import ConcatenativeSynthesis


def test_concatenative_synthesis_model():
    text_lang = "urdu"
    sign_lang = "psl"
    sign_format = "video"

    model = ConcatenativeSynthesis(
        text_language=text_lang, sign_language=sign_lang, sign_format=sign_format
    )
    assert model.text_language.name() == text_lang
    assert model.sign_language.name() == "pakistan-sign-language"
    assert model.sign_format.name() == "video"

    text = "ایک سیب اچھا ہے"
    sign_language_sentence = model.translate(text)
    sign_language_sentence.save(f"temp/{text}.mp4", overwrite=True)  # type: ignore
    # TODO: assert os.path.exists(f"temp/{text}.mp4"), os.listdir("temp")
