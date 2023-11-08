import os

from click.testing import CliRunner

from sign_language_translator.cli import slt
from sign_language_translator.config.settings import Settings


def test_slt():
    runner = CliRunner()
    result = runner.invoke(slt)
    # should show help message

    assert result.exit_code == 0
    assert "Usage: slt [OPTIONS] COMMAND [ARGS]" in result.output


def test_slt_translate():
    runner = CliRunner()

    # concatenative synthesis model
    text = "سیب اچھا ہے۔"
    result = runner.invoke(
        slt,
        [
            "translate",
            text,
            "--text-lang",
            "urdu",
            "--sign-lang",
            "psl",
            "--model-code",
            "concatenative-synthesis",
            "--sign-format",
            "video",
            "--overwrite",
            "true",
            "--output-dir",
            "temp",
            "--display",
            "false",
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(f"temp/{text}.mp4")


def test_slt_embed():
    runner = CliRunner()

    source_filepath = os.path.join(
        Settings.RESOURCES_ROOT_DIRECTORY, "videos", "wordless_wordless.mp4"
    )

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
    assert result.output.startswith("[m")


def test_slt_download():
    runner = CliRunner()

    result = runner.invoke(
        slt, ["download", "text_preprocessing.json", "--overwrite", "true"]
    )
    assert result.exit_code == 0
    assert os.path.exists(
        os.path.join(Settings.RESOURCES_ROOT_DIRECTORY, "text_preprocessing.json")
    )
