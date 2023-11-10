import os
import warnings
from copy import deepcopy

from sign_language_translator.config.assets import Assets
from sign_language_translator.languages.sign.pakistan_sign_language import (
    PakistanSignLanguage,
)
from sign_language_translator.text.tagger import Tags


def get_pakistan_sl_object():
    return PakistanSignLanguage() if os.path.exists(Assets.ROOT_DIR) else None


def test_pakistan_sentence_restructure():
    psl = get_pakistan_sl_object()

    if psl is None:
        warnings.warn(
            "Pakistan Sign Language object could not be initialized (check slt.Assets.ROOT_DIR)"
        )
        return

    urdu_tokens = ["میں(i)", " ", "وزیراعظم", " ", "عمران", " ", "خان", " "]
    urdu_tokens += ["کے(of)", " ", "گھر", " ", "گیا", "۔"]

    tags = [Tags.SUPPORTED_WORD, Tags.SPACE, Tags.SUPPORTED_WORD, Tags.SPACE, Tags.NAME]
    tags += [Tags.SPACE, Tags.NAME, Tags.SPACE, Tags.SUPPORTED_WORD, Tags.SPACE]
    tags += [Tags.SUPPORTED_WORD, Tags.SPACE, Tags.SUPPORTED_WORD, Tags.PUNCTUATION]

    expected_restructured = [
        "میں(i)",
        "وزیراعظم",
        "عمران",
        "خان",
        "کے(of)",
        "گھر",
        "گیا",
    ]

    restructured, _, _ = psl.restructure_sentence(urdu_tokens, tags=tags)
    assert restructured == expected_restructured


def test_pakistan_token_to_sign():
    psl = get_pakistan_sl_object()

    if psl is None:
        warnings.warn(
            "Pakistan Sign Language object could not be initialized (check slt.Assets.ROOT_DIR)"
        )
        return

    tokens = ["میں(متکلم)", "وزیراعظم", "عمران", "خان", "کے(حرف-اضافت)", "گھر", "گیا"]

    tags = [Tags.SUPPORTED_WORD, Tags.SUPPORTED_WORD, Tags.NAME, Tags.NAME]
    tags += [Tags.SUPPORTED_WORD, Tags.SUPPORTED_WORD, Tags.SUPPORTED_WORD]

    signs = psl.tokens_to_sign_dicts(tokens, tags=tags)
    expected_signs_1 = [
        {"signs": [["pk-hfad-1_میں(i)"]], "weights": [1.0]},
        {
            "signs": [
                [
                    "pk-hfad-1_a(double-handed-letter)",
                    "pk-hfad-1_a(double-handed-letter)",
                    "pk-hfad-1_a(double-handed-letter)",
                ],
                ["pk-hfad-1_وزیراعظم"],
            ],
            "weights": [0.5, 0.5],
        },
        {"signs": [["pk-hfad-1_ع"]], "weights": [1.0]},
        {"signs": [["pk-hfad-1_m(single-handed-letter)"]], "weights": [1.0]},
        {"signs": [["pk-hfad-1_r(single-handed-letter)"]], "weights": [1.0]},
        {"signs": [["pk-hfad-1_a(single-handed-letter)"]], "weights": [1.0]},
        {"signs": [["pk-hfad-1_n(single-handed-letter)"]], "weights": [1.0]},
        {"signs": [["pk-hfad-1_خ"]], "weights": [1.0]},
        {"signs": [["pk-hfad-1_a(single-handed-letter)"]], "weights": [1.0]},
        {"signs": [["pk-hfad-1_n(single-handed-letter)"]], "weights": [1.0]},
        {"signs": [["pk-hfad-1_کے(of)"]], "weights": [1.0]},
        {"signs": [["pk-hfad-1_گھر"]], "weights": [1.0]},
        {"signs": [["pk-hfad-1_گیا"]], "weights": [1.0]},
    ]
    # the other order
    expected_signs_2 = deepcopy(expected_signs_1)
    expected_signs_2[1]["signs"] = expected_signs_2[1]["signs"][::-1]

    assert signs in [expected_signs_1, expected_signs_2]

    assert [
        {"signs": [["pk-hfad-1_10"]], "weights": [1.0]}
    ] == psl.tokens_to_sign_dicts("10")

    try:
        psl.tokens_to_sign_dicts("2u14ujvdfhrfhvbh12")
    except ValueError:
        pass
    try:
        psl.tokens_to_sign_dicts(["i"], [Tags.AMBIGUOUS])
    except ValueError:
        pass


def test_psl():
    psl = get_pakistan_sl_object()

    if psl is None:
        warnings.warn(
            "Pakistan Sign Language object could not be initialized (check slt.Assets.ROOT_DIR)"
        )
        return

    psl.STOPWORDS.add("the")
    signs = psl(["1,010", "the", "102"], tags=[Tags.NUMBER, Tags.WORD, Tags.NUMBER])  # type: ignore
    assert signs == [
        {"signs": [["pk-hfad-1_10"]], "weights": [1.0]},
        {"signs": [["pk-hfad-1_10"]], "weights": [1.0]},
        {"signs": [["pk-hfad-1_10"]], "weights": [1.0]},
        {"signs": [["pk-hfad-1_v(single-handed-letter)"]], "weights": [1.0]},
    ]
