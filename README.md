# Sign Language Translator ⠎⠇⠞

[![python](https://img.shields.io/pypi/pyversions/sign-language-translator)](https://pypi.org/project/sign-language-translator/)
[![PyPi](https://img.shields.io/pypi/v/sign-language-translator)](https://pypi.org/project/sign-language-translator/)
[![Downloads](https://static.pepy.tech/personalized-badge/sign-language-translator?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/sign-language-translator/)

![Release Workflow Status](https://img.shields.io/github/actions/workflow/status/sign-language-translator/sign-language-translator/release.yml?branch=main)
[![codecov](https://codecov.io/gh/sign-language-translator/sign-language-translator/branch/main/graph/badge.svg)](https://codecov.io/gh/sign-language-translator/sign-language-translator)
[![Documentation Status](https://readthedocs.org/projects/sign-language-translator/badge/?version=latest)](https://sign-language-translator.readthedocs.io/)

1. [Overview](#overview)
   1. [Solution](#solution)
   2. [Major Components and Goals](#major-components-and-goals)
2. [How to install the package](#how-to-install-the-package)
3. [**Usage**](#usage)
4. [Languages](#languages)
5. [Models](#models)
6. [How to Build a Translator for your Sign Language](#how-to-build-a-translator-for-sign-language)
7. [Directory Tree](#directory-tree)
8. [How to Contribute](#how-to-contribute)
9. [Research Papers & Citation](#research-papers--citation)
10. [Upcoming/Roadmap](#upcomingroadmap)
11. [Credits and Gratitude](#credits-and-gratitude)
12. [Bonus](#bonus)
    1. Number of lines of code
    2. :)

## Overview

Sign language consists of gestures and expressions used mainly by the hearing-impaired to talk. This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.

The goal is to provide a user friendly API to novel Sign Language Translation solutions that can easily adapt to any regional sign language. Unlike most other projects, this python library can translate full sentences and not just the alphabet.
<!-- This is the package that powers the [slt_ai website](https://github.com/mdsrqbl/slt_ai). -->

A bigger hurdle is the lack of datasets and frameworks that deep learning engineers and software developers can use to build useful products for the target community. This project aims to empower sign language translation by providing robust components, tools and models for both sign language to text and text to sign language conversion. It seeks to advance the development of sign language translators for various regions while providing a way towards sign language standardization.

### Solution

I've have built an *extensible rule-based* text-to-sign translation system that can be used to generate training data for *Deep Learning* models for both sign to text & text to sign translation.

To create a rule-based translation system for your regional language, you can inherit the TextLanguage and SignLanguage classes and pass them as arguments to the ConcatenativeSynthesis class. To write sample texts of supported words, you can use our language models. Then, you can use that system to fine-tune our AI models. See the [documentation](https://sign-language-translator.readthedocs.io) for more.

### Major Components and Goals

<ol>
<li>
<details>
<summary>
Sign language to Text
</summary>

- Extract pose vectors (2D or 3D) from videos and map them to corresponding text representations of the performed signs.

- Fine-tuned a neural network, such as a state-of-the-art speech-to-text model, with gradual unfreezing starting from the input layers to convert pose vectors to text.

</details>
</li>

<li>
<details>
<summary>
Text to Sign Language
</summary>

- This is a relatively easier task if you parse input text and play appropriate video clips for each word.

1. Motion Transfer
    - Concatenate pose vectors in the time dimension and transfer the movements onto any given image of a person. This ensures smooth transitions between video clips.
2. Sign Feature Synthesis
    - Condition a pose sequence generation model on a pre-trained text encoder (e.g., fine-tune decoder of a multilingual T5) to output pose vectors instead of text tokens. This solves challenges related to unknown synonyms or hard-to-tokenize/process words or phrases.

</details>

<li>
<details>
<summary>
Language Processing Utilities
</summary>

 1. Sign Processing
    - 3D world landmarks extraction with Mediapipe.
    - Pose Visualization with matplotlib and moviepy.
    - Pose transformations (data augmentation) with scipy.
 2. Text Processing
    - Normalize text input by substituting unknown characters/spellings with supported words.
    - Disambiguate context-dependent words to ensure accurate translation.
         "spring" -> ["spring(water-spring)", "spring(metal-coil)"]
    - Tokenize text (word & sentence level).
    - Classify tokens and mark them with Tags.

 </details>

<li>
<details>
<summary>
Data Collection and Creation
</summary>

- Capture variations in signs in a scalable and diversity accommodating way and enable advancing sign language standardization efforts.

   1. Clip extraction from long videos using timestamps
   2. Multithreaded Web scraping
   3. Language Models to generate sentences composed of supported word

</details>

<li>
<details>
<summary>
Datasets
</summary>

The sign videos are categorized by:

```text
1. country
2. source organization
3. session number
4. camera angle
5. person code ((d: deaf | h: hearing)(m: male | f: female)000001)
6. equivalent text language word
```

The files are labeled as follows:

```text
country_organization_sessionNumber_cameraAngle_personCode_word.extension
```

The text data includes:

```text
7. word/sentence mappings to videos
8. spoken language sentences and phrases
9. spoken language sentences & corresponding sign video label sequences
10. preprocessing data such as word-to-numbers, misspellings, named-entities etc
```

[See the *sign-language-datasets* repo and its *release files* for the actual data & details](https://github.com/sign-language-translator/sign-language-datasets)

</details>
</ol>

## How to install the package

```bash
pip install sign-language-translator
```

<details>
<summary>Editable mode:</summary>

```bash
git clone https://github.com/sign-language-translator/sign-language-translator.git
cd sign-language-translator
pip install -e .
```

```bash
pip install -e git+https://github.com/sign-language-translator/sign-language-translator.git#egg=sign_language_translator
```

</details>

## Usage

Head over to <span style="font-size:large;">[sign-language-translator.**readthedocs**.io](https://sign-language-translator.readthedocs.io)</span> to see the detailed usage in Python, Command line and GUI.

See the [*test cases*](https://github.com/sign-language-translator/sign-language-translator/blob/main/tests) or the [*notebooks* repo](https://github.com/sign-language-translator/notebooks) to see the internal code in action.

Also see [How to build a custom sign language translator](#how-to-build-a-translator-for-sign-language).

```bash
$ slt

Usage: slt [OPTIONS] COMMAND [ARGS]...
   Sign Language Translator (SLT) command line interface.
   Documentation: https://sign-language-translator.readthedocs.io
Options:
  --version  Show the version and exit.
  --help     Show this message and exit.
Commands:
  complete   Complete a sequence using Language Models.
  download   Download resource files with regex.
  embed      Embed Videos Using Selected Model.
  translate  Translate text into sign language or vice versa.
```

```python
# Documentation: https://sign-language-translator.readthedocs.io
import sign_language_translator as slt
help(slt)

# The core model of the project (rule-based text-to-sign translator)
# which enables us to generate synthetic training datasets
model = slt.models.ConcatenativeSynthesis(
   text_language="urdu", sign_language="psl", sign_format="video"
)
text = "سیب اچھا ہے"
sign = model.translate(text) # tokenize, map, download & concatenate
sign.show(inline_player="html5") # jupyter notebook
sign.save(f"{text}.mp4")

# # Load any model
# # print(list(slt.ModelCodes))
# model = slt.get_model(slt.ModelCodes.Gesture) # sign-to-text (pytorch)
# sign = slt.Video("video.mp4")
# text = model.translate(sign)
# print(text)
# # sign.show()

# # DocStrings
# help(slt.languages.SignLanguage)
# help(slt.languages.text.Urdu)
# help(slt.Video)
# help(slt.models.MediaPipeLandmarksModel)
# help(slt.models.TransformerLanguageModel)
```

https://github.com/sign-language-translator/sign-language-translator/assets/118578823/b5da28ef-d04d-44c0-9ed8-1343ac004255

## Languages

<details>
<summary style="font-weight:bold;">Text Languages</summary>

Available Functions:

- Text Normalization
- Tokenization (word, phrase & sentence)
- Token Classification (Tagging)
- Word Sense Disambiguation

| Name                                                                                                                                   | Vocabulary         | Ambiguous tokens | Signs |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | --------------- | ----- |
| [Urdu](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/urdu.py) | 2090 words+phrases | 227              | 790   |

</details>

<details>
<summary style="font-weight:bold;">Sign Languages</summary>

Available Functions:

- Word & phrase mapping to signs
- Sentence restructuring according to grammar
- Sentence simplification (drop stopwords)

| Name                                                                                                                                                                       | Vocabulary | Dataset | Parallel Corpus |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------- | --- |
| [Pakistan Sign Language](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/sign/pakistan_sign_language.py) | 789        | 3 hours | n transcribed sentences with translations in m text languages |

</details>

## Models

<details>
<summary style="font-weight:bold;">Translation: Text to sign Language</summary>

<!-- [Available Trained models]() -->

| Name                                                                                                                                                                              | Architecture        | Description                                                                                                                                            | Input  | Output   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ | -------- |
| [Concatenative Synthesis](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py) | Rules + Hash Tables | The Core Rule-Based translator mainly used to synthesize translation dataset.<br/>Initialize it using TextLanguage, SignLanguage & SignFormat objects. | string | slt.Sign |

<!--                                                                                                                                                                              | [pose-gen]()        | Encoder-Decoder Transformers (Seq2Seq)                                                                                                              | Generates a sequence of pose vectors conditioned on input text. | torch.Tensor<br/>(batch, token_ids) | torch.Tensor<br/>(batch, n_frames, n_landmarks*3) | -->

</details>

<!-- <details>
<summary>Translation: Sign Language to Text</summary>

[Available Trained models]()

| Name        | Architecture                                | Description                                                                                                  | Input format                                             | Output format                       |
| ----------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- | ----------------------------------- |
| [gesture]() | CNN+Encoder - Decoder Transformer (seq2seq) | Encodes the pose vectors depicting sign language sentence and generates text conditioned on those encodings. | torch.Tensor<br/>(batch, n_frames=1000, n_landmarks * 3) | torch.Tensor<br/>(batch, token_ids) |
</details> -->

<!--
<details>
<summary>Video: Synthesis/Generation</summary>

[Available Trained models]()

| Name        | Architecture                                | Description                                                                                                  | Input format                                             | Output format                       |
| ----------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- | ----------------------------------- |
| [gesture]() | CNN+Encoder - Decoder Transformer (seq2seq) | Encodes the pose vectors depicting sign language sentence and generates text conditioned on those encodings. | torch.Tensor<br/>(batch, n_frames=1000, n_landmarks * 3) | torch.Tensor<br/>(batch, token_ids) |
</details>
-->

<details>
<summary style="font-weight:bold;">Video: Embedding/Feature extraction</summary>

<!-- [Available Trained models]() -->

| Name                                                                                                                                                                                                 | Architecture                                                                                                               | Description                                                                                       | Input format                                                 | Output format                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------- |
| [MediaPipe Landmarks<br>(Pose + Hands)](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/video_embedding/mediapipe_landmarks_model.py) | CNN based pipelines. See Here: [Pose](https://arxiv.org/pdf/2006.10204.pdf), [Hands](https://arxiv.org/pdf/2006.10214.pdf) | Encodes videos into pose vectors (3D world or 2D image) depicting the movements of the performer. | List of numpy images<br/>(n_frames, height, width, channels) | torch.Tensor<br/>(n_frames, n_landmarks * 5) |
</details>

<details>
<summary style="font-weight:bold;">Data generation: Language Models</summary>

[Available Trained models](https://github.com/sign-language-translator/sign-language-datasets/releases/tag/v0.0.1)

| Name                                                                                                                                                                                             | Architecture                    | Description                                                                                         | Input format                                                | Output format                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [N-Gram Langauge Model](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/ngram_language_model.py)                  | Hash Tables                     | Predicts the next token based on learned statistics about previous N tokens.                        | List of tokens                                              | (token, probability)                                                          |
| [Transformer Language Model](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/transformer_language_model/model.py) | Decoder-only Transformers (GPT) | Predicts next token using query-key-value attention, linear transformations and soft probabilities. | torch.Tensor<br/>(batch, token_ids)<br/><br/>List of tokens | torch.Tensor<br/>(batch, token_ids, vocab_size)<br/><br/>(token, probability) |
</details>

<!--
## Servers

| Name (repository) | Framework | Docker  | Status       |
| ----------------- | --------- | ------- | ------------ |
| [slt-models]()    | FastAPI   | [url]() | Coming Soon! |
| [slt-backend]()   | Django    | [url]() | Coming Soon! |
| [slt-frontend]()  | React     | [url]() | Coming Soon! |

You can interact with the live version of the above servers at [something.com](https://www.something.com)
-->

## How to Build a Translator for Sign Language

To create your own sign language translator, you'll need these essential components:

<ol>
<li>
<details>
<summary>Data Collection</summary>

   1. Gather a collection of [videos](link_to_videos) featuring individuals performing sign language gestures.
   2. Prepare a [JSON file](https://github.com/sign-language-translator/sign-language-datasets/blob/main/sign_recordings/collection_to_label_to_language_to_words.json) that maps video file names to corresponding text language words, phrases, or sentences that represent the gestures.
   3. Prepare a [parallel corpus](link_to_parallel_corpus) containing text language sentences and sequences of sign language video filenames.

</details>
</li>

<li>
<details>
<summary>Language Processing</summary>

   1. Implement a subclass of `slt.languages.TextLanguage`:
      - Tokenize your text language and assign appropriate tags to the tokens for streamlined processing.
   2. Create a subclass of `slt.languages.SignLanguage`:
      - Map text tokens to video filenames using the provided JSON data.
      - Rearrange the sequence of video filenames to align with the grammar and structure of sign language.

</details>
</li>

<li>
<details>
<summary>Rule-Based Translation</summary>

   1. Pass instances of your classes from the previous step to `slt.models.ConcatenativeSynthesis` class to obtain a rule-based translator object.
   2. Construct sentences in your text language and use the rule-based translator to generate sign language translations. (You can use our language models to generate such texts.)

</details>
</li>

<li>
<details>
<summary>Model Fine-Tuning</summary>

   1. Utilize the sign language videos and corresponding text sentences from the previous step.
   2. Apply our training pipeline to fine-tune a chosen model for improved accuracy and translation quality.

</details>
</li>
</ol>

Remember to contribute back to the community:

- Share your data, code, and models by creating a pull request (PR), allowing others to benefit from your efforts.
- Create your own sign language translator (e.g. as your university thesis) and contribute to a more inclusive and accessible world.

See more at [Build Custom Translator section in ReadTheDocs](https://sign-language-translator.readthedocs.io/en/latest/#building-custom-translators) or in this [notebook](https://github.com/sign-language-translator/notebooks/blob/main/translation/concatenative_synthesis.ipynb).
<!-- TODO: rename this notebook ^ -->

## Directory Tree

<pre>
<b style="font-size:large;">sign-language-translator</b>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/.readthedocs.yaml">.readthedocs.yaml</a>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/MANIFEST.in">MANIFEST.in</a>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/README.md">README.md</a>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/poetry.lock">poetry.lock</a>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/pyproject.toml">pyproject.toml</a>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/requirements.txt">requirements.txt</a>
├── <b>docs</b>
│   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/docs">*</a>
├── <b>tests</b>
│   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/tests">*</a>
│
└── <b style="font-size:large;">sign_language_translator</b>
    ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/cli.py">cli.py</a>
    ├── <i><b>assets</b></i> (auto-downloaded)
    │   └── <a href="https://github.com/sign-language-translator/sign-language-datasets">*</a>
    │
    ├── <b>config</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/assets.py">assets.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/enums.py">enums.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/settings.py">settings.py</a>
    │   ├── <i><a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/urls.json">urls.json</a></i>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/utils.py">utils.py</a>
    │
    ├── <b>data_collection</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/data_collection/completeness.py">completeness.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/data_collection/scraping.py">scraping.py</a>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/data_collection/synonyms.py">synonyms.py</a>
    │
    ├── <b>languages</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/utils.py">utils.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/vocab.p">vocab.py</a>
    │   ├── sign
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/sign/mapping_rules.py">mapping_rules.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/sign/pakistan_sign_language.py">pakistan_sign_language.py</a>
    │   │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/sign/sign_language.py">sign_language.py</a>
    │   │
    │   └── text
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/english.py">english.py</a>
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/text_language.py">text_language.py</a>
    │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/urdu.py">urdu.py</a>
    │
    ├── <b>models</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/_utils.py">_utils.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/utils.py">utils.py</a>
    │   ├── language_models
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/abstract_language_model.py">abstract_language_model.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/beam_sampling.py">beam_sampling.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/mixer.py">mixer.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/ngram_language_model.py">ngram_language_model.py</a>
    │   │   └── transformer_language_model
    │   │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/transformer_language_model/layers.py">layers.py</a>
    │   │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/transformer_language_model/model.py">model.py</a>
    │   │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/transformer_language_model/train.py">train.py</a>
    │   │
    │   ├── sign_to_text
    │   ├── text_to_sign
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py">concatenative_synthesis.py</a>
    │   │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/t2s_model.py">t2s_model.py</a>
    │   │
    │   └── video_embedding
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/video_embedding/mediapipe_landmarks_model.py">mediapipe_landmarks_model.py</a>
    │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/video_embedding/video_embedding_model.py">video_embedding_model.py</a>
    │
    ├── <b>text</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/metrics.py">metrics.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/preprocess.py">preprocess.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/subtitles.py">subtitles.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/tagger.py">tagger.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/tokenizer.py">tokenizer.py</a>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/utils.py">utils.py</a>
    │
    ├── <b>utils</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/arrays.py">arrays.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/download.py">download.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/tree.py">tree.py</a>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/utils.py">utils.py</a>
    │
    └── <b>vision</b>
        ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/_utils.py">_utils.py</a>
        ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/utils.py">utils.py</a>
        ├── landmarks
        ├── sign
        │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/sign/sign.py">sign.py</a>
        │
        └── video
            ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/video/display.py">display.py</a>
            ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/video/transformations.py">transformations.py</a>
            ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/video/video_iterators.py">video_iterators.py</a>
            └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/video/video.py">video.py</a>
</pre>

## How to Contribute

<details>
<summary style="font-weight:bold;">Datasets:</summary>

- Contribute by scraping, compiling, and centralizing video datasets.
- Help with labeling word mapping datasets.
- Establish connections with Academies for the Deaf to collaboratively develop standardized *sign language grammar* and integrate it into the rule-based translators.

</details>

<details>
<summary style="font-weight:bold;">New Code:</summary>

- Create dedicated sign language classes catering to various regions.
- Develop text language processing classes for diverse languages.
- Experiment with training models using diverse hyper-parameters.
- Don't forget to integrate `string short codes` of your classes and models into **`enums.py`**, and ensure to update functions like `get_model()` and `get_.*_language()`.
- Enhance the codebase with comprehensive docstrings, exemplary usage cases, and thorough test cases.

</details>

<details>
<summary style="font-weight:bold;">Existing Code:</summary>

- Optimize the codebase by implementing techniques like parallel processing and batching.
- Strengthen the project's documentation with clear docstrings, illustrative usage scenarios, and robust test coverage.
- Contribute to the documentation for [sign-language-translator ReadTheDocs](https://github.com/sign-language-translator/sign-language-translator/blob/main/docs/index.rst) to empower users with comprehensive insights. Currently it needs a template for auto-generated pages.

</details>

<details open>
<summary style="font-weight:bold;">Product Development:
</summary>

- Engage in the development efforts across [MLOps](), [backend](), [web](), and [mobile]() domains, depending on your expertise and interests.

</details>

## Research Papers & Citation

Stay Tuned!

## Upcoming/Roadmap

<details>
<summary>CLEAN_ARCHITECTURE_VISION: v0.7</summary>

```python
# urls.json, extra-urls.json

# bugfix: inaccurate num_frames in video file metadata
# improvement: video wrapper class uses list of sources instead of linked list of videos
# video transformations

# landmarks wrapper class
# landmark augmentation

# subtitles
# trim signs before concatenation

# stabilize video batch using landmarks
```

</details>

<details>
<summary>LANGUAGES: v0.8</summary>

```python
# implement NLP classes for English & Hindi
# Improve vocab class
# expand reference clip data by scraping everything
```

</details>

<details>
<summary>MISCELLANEOUS</summary>

```python
# clean demonstration notebooks
# host video dataset online, descriptive filenames, zip extraction
# dataset info table
# sequence diagram for creating a translator
# make scraping dependencies optional (beautifulsoup4, deep_translator). remove overly specific scrapping functions
# GUI with gradio
```

</details>

<details>
<summary>DEEP_TRANSLATION: v0.9-v1.x</summary>

```python
# parallel text corpus
# sign to text with custom seq2seq transformer
# sign to text with fine-tuned whisper
# pose vector generation with fine-tuned flan-T5
# motion transfer
# pose2video: stable diffusion or GAN?
# speech to text
# text to speech
# LanguageModel: experiment by dropping space tokens & bidirectional prediction
```

</details>

<details>
<summary>RESEARCH PAPERs</summary>

```python
# datasets: clips, text, sentences, disambiguation
# rule based translation: describe entire repo
# deep sign-to-text: pipeline + experiments
# deep text-to-sign: pipeline + experiments
```

</details>

<details>
<summary>PRODUCT DEVELOPMENT</summary>

```python
# ML inference server
# Django backend server
# React Frontend
# React Native mobile app
```

</details>

## Credits and Gratitude

This project started in October 2021 as a BS Computer Science final year project with 3 students and 1 supervisor. After 9 months at university, it became a hobby project for Mudassar who has continued it till at least 2023-11-10.

<details>
<summary> Immense gratitude towards: (click to expand)</summary>

- [Mudassar Iqbal](https://github.com/mdsrqbl) for coding the project so far.
- Rabbia Arshad for help in initial R&D and web development.
- Waqas Bin Abbas for assistance in initial video data collection process.
- Kamran Malik for setting the initial project scope, idea of motion transfer and connecting us with Hamza Foundation.
- [Hamza Foundation](https://www.youtube.com/@pslhamzafoundationacademyf7624/videos) (especially Ms Benish, Ms Rashda & Mr Zeeshan) for agreeing to collaborate and providing the reference clips, hearing-impaired performers for data creation, and creating the text2gloss dataset.
- [UrduHack](https://github.com/urduhack/urduhack) (espacially Ikram Ali) for their work on Urdu character normalization.

- [Telha Bilal](https://github.com/TelhaBilal) for help in designing the architecture of some modules.

</details>

## Bonus

Count total number of **lines of code** (Package: **9287** + Tests: **1419**):

```bash
git ls-files | grep '\.py' | xargs wc -l
```

**Just for `Fun`**

```text
Q: What was the deaf student's favorite course?
A: Communication skills
```
