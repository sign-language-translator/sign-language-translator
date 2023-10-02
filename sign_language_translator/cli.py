"""
Sign Language Translator (SLT) Command Line Interface

This module provides a command line interface (CLI) for the Sign Language Translator (SLT) library.
It allows you to perform various operations such as translating text to sign language or vice versa,
downloading resource files & completing text sequences using Language Models.

Usage:
    slt [OPTIONS] COMMAND [ARGS]...

Options:
    --help  Show this message and exit.

Commands:
    complete   Complete a sequence using Language Models.
    download   Download resource files.
    translate  Translate text into sign language or vice versa.
    embed      Embed Videos Using Selected Model.
"""

import click

from sign_language_translator import __version__

# TODO: Dockerize the CLI.. but model persistance issue


@click.group()
@click.version_option(__version__)
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
@slt.command(no_args_is_help=True)
@click.argument("filenames", nargs=-1, required=True)
@click.option(
    "--overwrite",
    default=False,
    help="Overwrite existing files. Defaults to False.",
)
@click.option(
    "--progress-bar",
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
def download(filenames, overwrite, progress_bar, timeout, chunk_size):
    """
    Download resource files with regex.

    Downloads package resources matching the given filename regex
    and saves them to the resources folder in installation directory.\n
    Examples:\n
        $ slt download '.*.json' --overwrite true
    """

    from sign_language_translator.utils import download_resource

    # Download the specified files
    for filename in filenames:
        success = download_resource(
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
    "--model-code", required=True, help="Short code to identify the translation model."
)
@click.option("--text-lang", help="Name of the text language.")
@click.option("--sign-lang", help="Name of the sign language.")
@click.option(
    "--sign-format",
    help="the sign features to be used e.g. pixels, mediapipe-landmarks etc.",
)
@click.option(
    "--target-path", default=".", help="Output directory for generated translations."
)
def translate(inputs, model_code, text_lang, sign_lang, sign_format, target_path):
    """
    Translate text into sign language or vice versa.

    Translates the given inputs between text and sign language using the specified model and features.
    Currently following model-codes are supported:\n
    1. "concatenative-synthesis" or "rule-based"\n
    Examples:\n
        $ slt translate --model-code rule-based --text-lang urdu --sign-lang pakistan-sign-language --sign-format mediapipe-landmarks "وہ سکول گیا تھا۔"
    """

    from sign_language_translator import get_model
    from sign_language_translator.models import TextToSignModel

    model = get_model(
        model_code=model_code,
        sign_language=sign_lang,
        text_language=text_lang,
        sign_format=sign_format,
    )
    if model and isinstance(model, TextToSignModel):  # TODO: , SignToTextModel):
        for inp in inputs:
            click.echo(model.translate(inp))
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
    help="Keep generating until this token. Defaults to '>'.",
)
@click.option(
    "--max-length",
    default=20,
    help="Maximum number of tokens to generate. Defaults to 20.",
)
@click.option(
    "--beam-width",
    default=3.0,
    help="Number of possible branches to explore during generation. Defaults to 3.",
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
    help="In case multiple models are used, should one model be selected at a time for inference ('choose') or should all models be inferred and their output probabilities combined before sampling ('merge'). Defaults to 'choose'.",
)
def complete(
    inputs,
    model_code,
    end_token,
    max_length,
    beam_width,
    model_weight,
    selection_strategy,
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
                click.echo(completion)
        else:
            # assume that all inputs are tokens of same sequence
            completion, _ = sampler.complete(inputs)
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
