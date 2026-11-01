import sys

from sign_language_translator import ModelCodes
from sign_language_translator.models import (
    ConcatenativeSynthesis,
    MediaPipeHolisticModel,
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
    if (3, 8) <= sys.version_info < (3, 12):
        model = get_model(ModelCodes.MEDIAPIPE_HOLISTIC_V2.value)
        assert isinstance(model, MediaPipeHolisticModel)
        assert model.model_complexity == 2
        assert get_model("mediapipe-holistic-0").model_complexity == 0
        assert isinstance(get_model("holistic"), MediaPipeHolisticModel)

    # non-existent model
    assert get_model("non-existent-model-code-should-return-None") is None
