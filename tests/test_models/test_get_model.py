import sys

from sign_language_translator import ModelCodes
from sign_language_translator.models import (
    ConcatenativeSynthesis,
    MediaPipeLandmarksModel,
    MixerLM,
    NgramLanguageModel,
    TransformerLanguageModel,
)
from sign_language_translator.models._utils import get_model


def test_get_model():
    # translators
    assert isinstance(
        get_model(
            ModelCodes.CONCATENATIVE_SYNTHESIS,
            text_language="urdu",
            sign_language="psl",
            sign_format="video",
        ),
        ConcatenativeSynthesis,
    )

    # language models
    assert isinstance(get_model(ModelCodes.MIXER_LM_NGRAM_URDU.value), MixerLM)
    assert isinstance(
        get_model(ModelCodes.TRANSFORMER_LM_UR_SUPPORTED.value),
        TransformerLanguageModel,
    )
    assert isinstance(
        get_model(ModelCodes.NGRAM_LM_BIGRAM_NAMES.value), NgramLanguageModel
    )

    # video embedding models
    if (3, 8) <= sys.version_info <= (3, 11):
        assert isinstance(get_model("mediapipe"), MediaPipeLandmarksModel)

    # non-existent model
    assert get_model("non-existent-model-code-should-return-None") is None
