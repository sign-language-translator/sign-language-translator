"""
Sign Language Translator (SLT) Command Line Interface

This module provides a command line interface (CLI) for the Sign Language Translator (SLT) library.
It allows you to perform various operations such as translating text to sign language or vice versa,
downloading resource files, completing text sequences using Language Models & embedding videos into
sequences of vectors.

.. code-block:: console

    $ slt
    Usage:
        slt [OPTIONS] COMMAND [ARGS]...

    Options:
        --help  Show this message and exit.

    Commands:
        assets     Assets manager to download & display Datasets & Models.
        complete   Complete a sequence using Language Models.
        translate  Translate text into sign language or vice versa.
        embed      Embed Videos Using Selected Model.
"""

import os

import click

from sign_language_translator import __version__

# TODO: Dockerize the CLI.. but model persistance issue


@click.group()
@click.version_option(__version__)
def slt():
    """
    Sign Language Translator (SLT) command line interface.
    Documentation: https://sign-language-translator.readthedocs.io
    """


@slt.group(no_args_is_help=True)
def assets():
    """
    Assets manager to download & display Datasets & Models.
    """


# Display the assets root directory
@assets.command()
def path():
    """
    Display the assets root directory.
    """

    from sign_language_translator.config.assets import Assets

    click.echo(Assets.ROOT_DIR)


# Display a tree of downloaded resource files
@assets.command()
@click.option(
    "--files",
    "-f",
    default=True,
    show_default=True,
    help="Include files as leaf nodes in the directory tree.",
)
@click.option(
    "--ignore",
    "-i",
    default=[],
    multiple=True,
    show_default=True,
    help="List of regular expressions of file and directory names that should not be displayed.",
)
@click.option(
    "--directory",
    "-d",
    default=None,
    help="Where the asset files are stored. Defaults to 'installation-directory/assets'.",
)
def tree(files, ignore, directory):
    """
    Display a hierarchy of files in the SLT Assets folder.

    Examples:\n
        $ slt assets tree\n
        $ slt assets tree -f false\n
        $ slt assets tree -i ".*\\.mp4" -i ".*\\.csv"
    """

    from sign_language_translator.config.assets import Assets
    from sign_language_translator.utils import tree as display_tree

    if directory:
        Assets.set_root_dir(directory)

    display_tree(Assets.ROOT_DIR, directory_only=not files, ignore=ignore, regex=True)


# Download resource files
@assets.command(no_args_is_help=True)
@click.argument("filenames", nargs=-1, required=True)
@click.option(
    "--overwrite",
    "-o",
    default=False,
    help="Overwrite existing files. Defaults to False.",
)
@click.option(
    "--progress-bar",
    "-p",
    default=True,
    help="Show progress bar for kilobytes downloaded. Defaults to True.",
)
@click.option(
    "--timeout",
    default=20,
    help="Timeout duration for download requests. Defaults to 20 sec.",
)
@click.option(
    "--chunk-size",
    default=131072,
    help="number of bytes to download in each step. Defaults to 131072 or 128K.",
)
@click.option(
    "--directory",
    "-d",
    default=None,
    help="Where to save the downloaded files. Defaults to 'installation-directory/assets'.",
)
def download(filenames, overwrite, progress_bar, timeout, chunk_size, directory):
    """
    Download resource files with regex.

    Downloads package resources matching the given filename regex
    and saves them to the resources folder in installation directory.\n
    Examples:\n
        $ slt download '.*.json' --overwrite true
    """

    from sign_language_translator.config.assets import Assets

    if directory:
        Assets.set_root_dir(directory)

    # Download the specified files
    for filename in filenames:
        success = Assets.download(
            filename,
            overwrite=overwrite,
            progress_bar=progress_bar,
            timeout=timeout,
            chunk_size=chunk_size,
        )
        if success:
            click.echo(f"Downloaded package resource(s) matching: '{filename}'")
        else:
            click.echo(f"Couldn't download package resources matching: '{filename}'")


# Translate
@slt.command(no_args_is_help=True)
@click.argument("inputs", nargs=-1, required=True)
@click.option(
    "--model-code",
    "-m",
    required=True,
    help="Short code to identify the translation model.",
)
@click.option("--text-lang", help="Name of the text language.")
@click.option("--sign-lang", help="Name of the sign language.")
@click.option(
    "--sign-format",
    help="the sign features to be used e.g. video, mediapipe-landmarks etc.",
)
@click.option(
    "--output-dir", default=".", help="Output directory for generated translations."
)
@click.option(
    "--overwrite",
    "-o",
    default=False,
    help="Whether to overwrite the target file if it already exists. Defaults to False.",
)
@click.option(
    "--display",
    "-d",
    default=True,
    help="Whether to show the output video. Defaults to True.",
)
@click.option(
    "--save-format",
    default="mp4",
    help="The output file extension. Defaults to 'mp4'.",
)
def translate(
    inputs,
    model_code,
    text_lang,
    sign_lang,
    sign_format,
    output_dir,
    overwrite,
    display,
    save_format,
):
    """
    Translate text into sign language or vice versa.

    Translates the given inputs between text and sign language using the specified model and features.
    Currently following model-codes are supported:\n
    1. "concatenative-synthesis" or "rule-based"\n
    Examples:\n
        $ slt translate --model-code rule-based --text-lang urdu --sign-lang psl --sign-format video "ایک سیب اچھا ہے۔"
    """

    from sign_language_translator import get_model, get_sign_wrapper_class
    from sign_language_translator.models import TextToSignModel

    model = get_model(
        model_code=model_code,
        sign_language=sign_lang,
        text_language=text_lang,
        sign_format=sign_format,
    )
    if model and isinstance(model, TextToSignModel):  # TODO: , SignToTextModel):
        for text in inputs:
            sign = model.translate(text)
            path = os.path.join(output_dir, f"{text}.{save_format}")
            sign.save(path, overwrite=overwrite, leave=True)
            if display:
                get_sign_wrapper_class(sign.name())(path).show()

    else:
        click.echo("This type of translation is not yet supported!")


# Complete
@slt.command(no_args_is_help=True)
@click.argument("inputs", nargs=-1, required=True)
@click.option(
    "--model-code",
    "-m",
    required=True,
    help="Short code to identify the language model. You can use multiple models (but of same input type).",
    multiple=True,
)
@click.option(
    "--end-token",
    default=">",
    help="Keep generating until this token.",
    show_default=True,
)
@click.option(
    "--max-length",
    default=20,
    help="Maximum number of tokens to generate.",
    show_default=True,
)
@click.option(
    "--beam-width",
    default=3.0,
    help="Number of possible branches to explore during generation.",
    show_default=True,
)
@click.option(
    "--model-weight",
    "-w",
    default=[1.0],
    multiple=True,
    help="Likelihood of this model getting selected in case multiple models are passed. Defaults to equal probabilities.",
)
@click.option(
    "--selection-strategy",
    default="choose",
    help="In case multiple models are used, should one model be selected at a time for inference ('choose') or should all models be inferred and their output probabilities combined before sampling ('merge').",
    show_default=True,
)
@click.option(
    "--join",
    default=None,
    help="Join the tokens by placing this string inbetween.",
)
def complete(
    inputs,
    model_code,
    end_token,
    max_length,
    beam_width,
    model_weight,
    selection_strategy,
    join,
):
    """
    Complete a sequence using Language Models.

    Predicts next tokens in the given sequence and completes it using the specified model and beam search.
    You can also use multiple models and also pass along model selection weights for each model (using --model-weights).
    Currently you can use the following model-codes:\n
    1. urdu-mixed-ngram (token | start: "<")\n
    2. ur-supported-gpt (token | start: "<")\n
    3. unigram-names (char | start: "[")\n
    4. bigram-names  (char | start: "[a")\n
    5. trigram-names (char | start: "[ab")\n
    Examples:\n
        $ slt complete "<" --model-code urdu-mixed-ngram -w 1.0 --model-code ur-supported-gpt -w 1.5
    """

    from sign_language_translator import get_model
    from sign_language_translator.models import BeamSampling, LanguageModel, MixerLM

    if model_weight == (1,):
        model_weight = [1.0] * len(model_code)
    assert len(model_code) == len(
        model_weight
    ), f"provide 1 weight for every model. Received {model_weight}. Size mismatch: {len(model_code)-len(model_weight)}."

    models = [get_model(model_code=code) for code in model_code]

    if all(isinstance(model, LanguageModel) for model in models):
        sampler = BeamSampling(
            MixerLM(
                models,  # type: ignore
                selection_probabilities=model_weight,
                model_selection_strategy=selection_strategy,
            ),
            beam_width=beam_width,
            end_of_sequence_token=end_token,
            max_length=max_length,
        )

        if models[0].name and "character" in models[0].name:  # type: ignore
            for inp in inputs:
                completion, _ = sampler.complete(inp)
                if isinstance(join, str):
                    completion = join.join(completion)
                click.echo(completion)
        else:
            # assume that all inputs are tokens of same sequence
            completion, _ = sampler.complete(inputs)
            if isinstance(join, str):
                completion = join.join(completion)
            click.echo(completion)

    else:
        click.echo("Model loading failed!")


# Embed
@slt.command(no_args_is_help=True)
@click.argument("inputs", nargs=-1, required=True)
@click.option(
    "--model-code",
    "-m",
    required=True,
    help="Short code to identify the embedding model.",
)
@click.option(
    "--processes",
    default=1,
    help="Number of processes to launch to embed using multiprocessing.",
)
@click.option(
    "--embedding-type",
    default="all",
    help="Optional parameter to filter down embeddings if model can return various types.",
)
@click.option(
    "--output-dir",
    default=".",
    help="where to save the generated embeddings.",
)
@click.option(
    "--save-format",
    default="pt",
    help="What format to save the generated embeddings in. (e.g. pt, csv, npy, npz). Defaults to 'pt'.",
)
@click.option(
    "--overwrite",
    default=False,
    help="Flag to indicate whether to overwrite existing files. Defaults to False.",
)
# TODO: Precision/format
# @click.option(
#     "--precision",
#     default=4,
#     help="Number of decimal places to save.",
#     show_default=True,
# )
def embed(
    inputs,
    model_code,
    processes: int,
    embedding_type,
    output_dir,
    save_format,
    overwrite,
):
    """
    Embed Videos Using Selected Model.

    This function processes input videos using a selected embedding model to generate video embeddings.
    It supports multiple input videos using path patterns and multiprocessing for efficiency.
    Currently you can use the following model-codes:\n
    1. mediapipe-pose-2-hand-1\n
    2. mediapipe-pose-1-hand-1\n
    3. mediapipe-pose-0-hand-1\n
    Example:\n
        $ slt embed dataset/*.mp4 --model-code mediapipe-pose-2-hand-1 --embedding-type all --processes 4 --save-format csv
    """

    from sign_language_translator import get_model
    from sign_language_translator.models.utils import VideoEmbeddingPipeline

    model = get_model(model_code)
    pipeline = VideoEmbeddingPipeline(model)  # type: ignore

    pipeline.process_videos_parallel(
        inputs,
        n_processes=processes,
        output_dir=output_dir,
        save_format=save_format,
        landmark_type=embedding_type,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    slt()
