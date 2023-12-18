import os

from click.testing import CliRunner

from sign_language_translator.cli import slt
from sign_language_translator.config.assets import Assets


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


def test_slt_embed():
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


def test_slt_complete():
    runner = CliRunner()

    result = runner.invoke(slt, ["complete", "[m", "--model-code", "bigram-names"])
    assert result.exit_code == 0
    assert result.output.split()[-1].startswith("[m")  # ignore progress bar


def test_slt_assets_download():
    runner = CliRunner()

    result = runner.invoke(
        slt, ["assets", "download", "text_preprocessing.json", "--overwrite", "true"]
    )
    assert result.exit_code == 0
    assert os.path.exists(Assets.get_path("text_preprocessing.json")[0])


def test_slt_assets_tree():
    runner = CliRunner()

    result = runner.invoke(
        slt, ["assets", "tree"]
    )
    assert result.exit_code == 0
    assert result.output.strip().startswith("assets")

def test_slt_assets_path():
    runner = CliRunner()

    result = runner.invoke(
        slt, ["assets", "path"]
    )
    assert result.exit_code == 0
    assert result.output.strip() == Assets.ROOT_DIR
