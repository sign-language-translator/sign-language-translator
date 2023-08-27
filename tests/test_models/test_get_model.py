from sign_language_translator.models._utils import get_model
from sign_language_translator import ModelCodes


def test_get_model():
    assert get_model("mediapipe") is not None
    assert get_model(ModelCodes.MIXER_LM_NGRAM_URDU.value) is not None
    assert (
        get_model(ModelCodes.CONCATENATIVE_SYNTHESIS, "urdu", "psl", "mediapipe")
        is not None
    )
