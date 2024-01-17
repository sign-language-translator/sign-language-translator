import os
import random
from re import match

from sign_language_translator.models import BeamSampling, MixerLM, NgramLanguageModel


def test_ngram_language_model():
    random.seed(0)

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

    for _ in range(10):
        generation, _ = sampler.complete("[")
        assert match(r"\[ab+c?\]?", generation)  # type: ignore

    mix.strategy = "choose"
    for _ in range(20):
        tokens, probs = mix.next_all("[")
        assert abs(sum(probs) - 1) < 0.0001
        assert set(tokens) <= {"a", "b", "c", "[", "]"}

    assert str(mix) is not None

    sampler.return_log_of_probability = False
    generation, prob = sampler()
    assert match(r"\[ab+c?\]?", "".join(generation))
    assert 0 <= prob <= 1

    generation, prob = sampler.complete("x")
    assert generation == "x"  # unknown token is not appended
    assert prob == 1  # 2**0

    # save model
    os.makedirs("temp", exist_ok=True)
    lm1.save("temp/lm1.json", overwrite=True)

    assert os.path.exists("temp/lm1.json")
    assert NgramLanguageModel.load("temp/lm1.json").model == lm1.model
