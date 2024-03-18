import os

import torch
from click.testing import CliRunner

from sign_language_translator.cli import slt
from sign_language_translator.config.assets import Assets
from sign_language_translator.config.enums import ModelCodeGroups


def test_slt():
    runner = CliRunner()
    result = runner.invoke(slt)
    # should show help message

    assert result.exit_code == 0
    assert "Usage: slt [OPTIONS] COMMAND [ARGS]" in result.output


# def test_slt_translate():
#     # BUG: ffmpeg error while writing video (only via CliRunner on ubuntu workflow).
#     # global cap_ffmpeg_impl.hpp:3018 open Could not find encoder for codec_id=27, error: Encoder not found
#     # [ERROR:0@6.347] global cap_ffmpeg_impl.hpp:3093 open VIDEOIO/FFMPEG: Failed to initialize VideoWriter

#     runner = CliRunner()

#     # concatenative synthesis model
#     text = "سیب اچھا ہے۔"
#     result = runner.invoke(
#         slt,
#         [
#             "translate",
#             text,
#             "--text-lang",
#             "urdu",
#             "--sign-lang",
#             "psl",
#             "--model-code",
#             "concatenative-synthesis",
#             "--sign-format",
#             "video",
#             "--overwrite",
#             "true",
#             "--output-dir",
#             "temp",
#             "--display",
#             "false",
#             "--save-format",
#             "mkv",
#         ],
#     )
#     assert result.exit_code == 0
#     assert os.path.exists(f"temp/{text}.mkv")


def test_slt_embed_video():
    runner = CliRunner()

    video_id = "videos/wordless_wordless.mp4"
    Assets.download(video_id, overwrite=False)
    source_filepath = Assets.get_path(video_id)[0]

    result = runner.invoke(
        slt,
        [
            "embed",
            source_filepath,
            "--model-code",
            "mediapipe-pose-2-hand-1",
            "--output-dir",
            "temp",
            "--save-format",
            "npy",
            "--overwrite",
            "true",
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists("temp/wordless_wordless.mp4.npy")


def test_slt_embed_text():
    test_model_code = "lookup-test-model.pt"
    ModelCodeGroups.ALL_TEXT_EMBEDDING_MODELS.value.add(test_model_code)
    ModelCodeGroups.ALL_VECTOR_LOOKUP_MODELS.value.add(test_model_code)

    runner = CliRunner()

    with open(token_path := os.path.join("temp", "tokens.txt"), "w") as f:
        f.write("sign hello\nworld hello")

    result = runner.invoke(
        slt,
        [
            "embed",
            "hello",
            token_path,
            "world",
            "--model-code",
            test_model_code,
            "--embedding-type",
            "normalized",
            "--output-dir",
            "temp",
            "--overwrite",
            "true",
        ],
    )
    assert result.exit_code == 0
    output = "temp" + os.sep + result.output.splitlines()[-1].split("temp" + os.sep)[-1]
    assert os.path.exists(output)

    checkpoint = torch.load(output)
    assert checkpoint["tokens"] == ["hello", "sign hello", "world hello", "world"]
    assert checkpoint["vectors"].shape[0] == 4


def test_slt_complete():
    runner = CliRunner()

    result = runner.invoke(slt, ["complete", "[m", "--model-code", "bigram-names"])
    assert result.exit_code == 0
    assert result.output.split()[-1].startswith("[m")  # ignore progress bar


def test_slt_assets_download():
    runner = CliRunner()

    result = runner.invoke(
        slt, ["assets", "download", "text-preprocessing.json", "--overwrite", "true"]
    )
    assert result.exit_code == 0
    assert os.path.exists(Assets.get_path("text-preprocessing.json")[0])


def test_slt_assets_tree():
    runner = CliRunner()

    result = runner.invoke(slt, ["assets", "tree"])
    assert result.exit_code == 0
    assert result.output.strip().startswith("assets")


def test_slt_assets_path():
    runner = CliRunner()

    result = runner.invoke(slt, ["assets", "path"])
    assert result.exit_code == 0
    assert result.output.strip() == Assets.ROOT_DIR
