from re import match

from sign_language_translator.models import BeamSampling, MixerLM, NgramLanguageModel


def test_ngram_language_model():
    lm1 = NgramLanguageModel(window_size=1, unknown_token=".")
    lm1.fit(["[abbb]", "[abc]", "[abbc]"])

    lm2 = NgramLanguageModel(window_size=1, unknown_token=".")
    lm2.fit(["[abbb]", "[abc]", "[abbc]"])
    lm2.finetune(["[abbbbbbbbb]", "[abc]", "[abbbc]"], weightage=0.2)

    mix = MixerLM([lm1, lm2], unknown_token=".", model_selection_strategy="merge")

    sampler = BeamSampling(
        mix,
        beam_width=1,
        start_of_sequence_token="[",
        end_of_sequence_token="]",
        max_length=20,
    )

    for _ in range(5):
        generation, _ = sampler.complete("[")
        assert match(r"\[ab+c?\]?", generation)  # type: ignore
