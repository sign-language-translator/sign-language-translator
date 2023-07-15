"""
Sign Language Translator (SLT) Command Line Interface

This module provides a command line interface (CLI) for the Sign Language Translator (SLT) library.
It allows you to perform various operations, such as downloading resource files and translating text to sign language or vice versa.

Usage:
    slt [OPTIONS] COMMAND [ARGS]...

Options:
    --help  Show this message and exit.

Commands:
    download   Download resource files.
    translate  Translate text into sign language or vice versa.
"""

import click

# TODO: Dockerize the CLI.. but model persistance issue


@click.group()
def slt():
    """Sign Language Translator (SLT) command line interface."""


# @slt.command()
# @click.option(
#     "--dataset-dir",
#     help="save the dataset directory in an environment variable so that all other subcommands use that value.",
# )
# def configure(dataset_dir):
#     os.environ["SLT_DATASET_DIR"] = dataset_dir


# Download resource files
@slt.command()
@click.argument("filenames", nargs=-1, required=True)
@click.option("--overwrite", default=False, help="Overwrite existing files.")
@click.option(
    "--progress-bar",
    default=True,
    help="Show progress bar for kilobytes downloaded.",
)
@click.option("--timeout", default=20, help="Timeout duration for download requests.")
def download(filenames, overwrite, progress_bar, timeout):
    """
    Download resource files with regex.

    Downloads package resources matching the given filename regex and saves them to the appropriate file paths.
    """

    from sign_language_translator.utils import download_resource

    # Download the specified files
    for filename in filenames:
        success = download_resource(
            filename,
            overwrite=overwrite,
            progress_bar=progress_bar,
            timeout=timeout,
        )
        if success:
            click.echo(f"Downloaded package resource(s) matching: '{filename}'")
        else:
            click.echo(f"Couldn't download package resources matching: '{filename}'")


# Translate
@slt.command()
@click.argument("inputs", nargs=-1, required=True)
@click.option(
    "--model-code", required=True, help="Short code to identify the translation model."
)
@click.option("--text-lang", help="Name of the text language.")
@click.option("--sign-lang", help="Name of the sign language.")
@click.option(
    "--sign-features",
    help="the sign features to be used e.g. pixels, mediapipe-landmarks etc.",
)
@click.option(
    "--target-path", default=".", help="Output directory for generated translations."
)
def translate(inputs, model_code, text_lang, sign_lang, sign_features, target_path):
    """
    Translate text into sign language or vice versa.

    Translates the given inputs between text and sign language using the specified model and features.
    """
    from sign_language_translator import get_model

    model = get_model(
        model_code=model_code,
        sign_language=sign_lang,
        text_language=text_lang,
        video_feature_model=sign_features,
    )
    if model:
        for inp in inputs:
            click.echo(model.translate(inp))
    else:
        click.echo("This type of translation is not yet supported!")


# TODO: $ slt complete --model en-char-lm-1 --eos " " 'auto-complete is gre'


if __name__ == "__main__":
    slt()
