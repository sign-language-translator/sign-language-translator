from sign_language_translator.models._utils import get_model
from sign_language_translator import ModelCodes


def test_get_model():
    # translators
    assert (
        get_model(
            ModelCodes.CONCATENATIVE_SYNTHESIS,
            text_language="urdu",
            sign_language="psl",
            sign_format="mediapipe",
        )
        is not None
    )

    # language models
    assert get_model(ModelCodes.MIXER_LM_NGRAM_URDU.value) is not None
    assert get_model(ModelCodes.TRANSFORMER_LM_UR_SUPPORTED.value) is not None
    assert get_model(ModelCodes.NGRAM_LM_BIGRAM_NAMES.value) is not None

    # video embedding models
    assert get_model("mediapipe") is not None

    # non-existent model
    assert get_model("non-existent-model-code-should-return-None") is None
    