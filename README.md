<div align="center">

# Sign Language Translator ⠎⠇⠞

<img width="61.8%" alt="SLT: Sign Language Translator" src="https://github.com/sign-language-translator/sign-language-translator/assets/118578823/d4723333-3d25-413d-83a1-a4bbdc8da15a">

</br>

*Build Custom Translators and Translate between Sign Language & Text with AI.*

[![python](https://img.shields.io/pypi/pyversions/sign-language-translator?logo=python)](https://pypi.org/project/sign-language-translator/)
[![PyPi](https://img.shields.io/pypi/v/sign-language-translator?logo=pypi)](https://pypi.org/project/sign-language-translator/)
[![Downloads](https://img.shields.io/pepy/dt/sign_language_translator?color=purple&logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4MDAiIGhlaWdodD0iODAwIiBmaWxsPSJub25lIiB2aWV3Qm94PSIwIDAgMjQgMjQiPjxwYXRoIGZpbGw9IiMwMDAiIGZpbGwtcnVsZT0iZXZlbm9kZCIgZD0iTTkuMiAyLjhjLS4yLjMtLjIuOC0uMiAxLjZWMTFINy44Yy0uOSAwLTEuMyAwLTEuNS4yYS44LjggMCAwIDAtLjMuNmMwIC4zLjMuNiAxIDEuMmw0LjEgNC40LjcuNmEuNy43IDAgMCAwIC40IDBsLjctLjZMMTcgMTNjLjYtLjYuOS0xIC45LTEuMmEuOC44IDAgMCAwLS4zLS42Yy0uMi0uMi0uNi0uMi0xLjUtLjJIMTVWNC40YzAtLjggMC0xLjMtLjItMS42YTEuNSAxLjUgMCAwIDAtLjYtLjZjLS4zLS4yLS44LS4yLTEuNi0uMmgtMS4yYy0uOCAwLTEuMyAwLTEuNi4yYTEuNSAxLjUgMCAwIDAtLjYuNnpNNSAyMWExIDEgMCAwIDAgMSAxaDEyYTEgMSAwIDEgMCAwLTJINmExIDEgMCAwIDAtMSAxeiIgY2xpcC1ydWxlPSJldmVub2RkIi8+PC9zdmc+)](https://pepy.tech/project/sign-language-translator/)
![Release Workflow Status](https://img.shields.io/github/actions/workflow/status/sign-language-translator/sign-language-translator/release.yml?branch=main&logo=pytest)
[![codecov](https://codecov.io/gh/sign-language-translator/sign-language-translator/branch/main/graph/badge.svg)](https://codecov.io/gh/sign-language-translator/sign-language-translator)
[![Documentation Status](https://img.shields.io/readthedocs/sign-language-translator?logo=readthedocs&)](https://sign-language-translator.readthedocs.io/)
![GitHub Repo stars](https://img.shields.io/github/stars/sign-language-translator/sign-language-translator?logo=github)

</div>

---

1. [Overview](#overview)
   1. [Solution](#solution)
   2. [Major Components](#major-components)
   3. [Goals](#goals)
2. [**Installation** `🛠️`](#how-to-install-the-package)
3. [**Usage**](#usage)
   1. [Python `🐍`](#python)
   2. [Command Line <span style="color:green">**`>_`**</span>](#command-line)
4. [Languages](#languages)
5. [Models](#models)
6. [How to Build a Translator for your Sign Language](#how-to-build-a-translator-for-sign-language)
7. [Directory Tree](#directory-tree)
8. [How to Contribute](#how-to-contribute)
9. [Research Papers & Citation](#research-papers--citation)
10. [Credits and Gratitude](#credits-and-gratitude)

## Overview

Sign language consists of gestures and expressions used mainly by the hearing-impaired to talk. This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.

The goal is to provide a user-friendly translation API and a framework for building sign language translators that can easily adapt to any regional sign language. Unlike most other projects, this python library can translate full sentences and not just the alphabet.
<!-- This is the package that powers the [slt_ai website](https://github.com/mdsrqbl/slt_ai). -->

A big hurdle is the lack of datasets (global & regional) and frameworks that deep learning engineers and software developers can use to build useful products for the target community. This project aims to empower sign language translation by providing robust components, tools, datasets and models for both sign language to text and text to sign language conversion. It aims to facilitate the creation of sign language translators for any region, while building the way towards sign language standardization.

### Solution

I've have built an *extensible rule-based* text-to-sign translation system that can be used to generate training data for *Deep Learning* models for both sign to text & text to sign translation.

To create a rule-based translation system for your regional language, you can inherit the TextLanguage and SignLanguage classes and pass them as arguments to the ConcatenativeSynthesis class. To write sample texts of supported words, you can use our language models. Then, you can use that system to fine-tune our AI models. See the <kbd>[documentation](https://sign-language-translator.readthedocs.io)</kbd> and our <kbd>[datasets](https://github.com/sign-language-translator/sign-language-datasets)</kbd> for more.

### Major Components

<ol>
<li>
<details>
<summary><b>
Sign language to Text
</summary></b>

1. Extract features from sign language videos
   1. See the [`slt.models.video_embedding`](https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/video_embedding) sub-package and the [`$ slt embed`](https://sign-language-translator.readthedocs.io/en/latest/#embed-videos) command.
   2. Currently Mediapipe 3D landmarks are being used for deep learning.
2. Transcribe and translate signs into multiple text languages to generalize the model.
3. To train for word-for-word gloss writing task, also use a synthetic dataset made by concatenating signs for each word in a text. (See [`slt.models.ConcatenativeSynthesis`](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py))
4. Fine-tune a neural network, such as one from [`slt.models.sign_to_text`](https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/sign_to_text) or the encoder of any multilingual seq2seq model, on your dataset.

</details>
</li>

<li>
<details>
<summary><b>
Text to Sign Language
</summary></b>

There are two approaches to this problem:

1. Rule Based Concatenation
   1. Label a Sign Language Dictionary with all word tokens that can be mapped to those signs. See our mapping format [here](https://github.com/sign-language-translator/sign-language-datasets/blob/main/parallel_texts/pk-dictionary-mapping.json).
   2. Parse the input text and play appropriate video clips for each token.
      1. Build a text processor by inheriting `slt.languages.TextLanguage` (see [`slt.languages.text`](https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/languages/text) sub-package for details)
      2. Map the text grammar & words to sign language by inheriting `slt.languages.SignLanguage` (see [`slt.languages.sign`](https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/languages/sign) sub-package for details)
      3. Use our rule-based model [`slt.models.ConcatenativeSynthesis`](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py) for translation.
   3. It is faster but the **word sense has to be disambiguated** in the input. See the deep learning approach to automatically handle ambiguous words & **words not in dictionary**.

2. Deep learning (seq2seq)
   1. Either generate the sequence of filenames that should be concatenated <!-- TODO: `slt.models.mGlossBART` -->
      1. you will need a [parallel corpus](https://github.com/sign-language-translator/sign-language-datasets/blob/main/parallel_texts/pk-sentence-mapping.json) of normal text sentences against sign language gloss (sign sequence written word-for-word)
   2. Or synthesize the signs directly by using a pre-trained multilingual text encoder and
      1. a GAN or diffusion model or decoder to synthesize a sequence of pose vectors (`shape = (time, num_landmarks * num_coordinates)`) <!-- TODO: `slt.models.SignPoseGAN` -->
         1. Move an Avatar with those pose vectors (Easy) <!-- TODO: `slt.models.Avatar` -->
         2. Use motion transfer to generate a video (Medium) <!-- TODO: `slt.models.SignMotionTransfer` -->
         3. Synthesize a video frame for each vector (Difficult) <!-- TODO: `slt.models.DeepPoseToImage` -->
      2. a video synthesis model (Very Difficult) <!-- TODO: `slt.models.DeepSignVideoGAN` -->

</details>

<li>
<details>
<summary><b>
Language Processing
</summary></b>

 1. Sign Processing
    - 3D world landmarks extraction with Mediapipe.
    - Pose Visualization with matplotlib.
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
<summary><b>
Data Collection and Creation
</summary></b>

See our dataset conventions [here](https://github.com/sign-language-translator/sign-language-datasets?tab=readme-ov-file#datasets).

Utilities available in this package:

1. Clip extraction from long videos using timestamps ([notebook](https://github.com/sign-language-translator/notebooks/blob/main/data_collection/clip_extractor.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/sign-language-translator/notebooks/blob/main/data_collection/clip_extractor.ipynb))
2. Multithreaded Web scraping
3. Language Models to generate sentences composed of supported words

Try to capture variations in signs in a scalable and diversity accommodating way and enable advancing sign language standardization efforts.

</details>

<li>
<details>
<summary><b>
Datasets
</summary></b>

For our datasets, see the [*sign-language-datasets* repo](https://github.com/sign-language-translator/sign-language-datasets) and its [releases](https://github.com/sign-language-translator/sign-language-datasets/releases).
See these [docs](https://sign-language-translator.readthedocs.io/en/latest/datasets.html) for more on building a dataset of Sign Language videos (or motion capture gloves' output features).

**Your data should include**:

1. A word level Dictionary (Videos of individual signs & corresponding Text tokens (words & phrases))
2. Parallel sentences
   1. Normal text language sentences against sign language videos.
   2. Normal text language sentences against the [text gloss](https://github.com/sign-language-translator/sign-language-datasets#glossary) of the corresponding sign language sentence.
   3. Sign language sentences against their text gloss
   4. Sign language sentences against translations in multiple text languages
3. Grammatical rules of the sign language
   1. Word order (e.g. SUBJECT OBJECT VERB TIME)
   2. Meaningless words (e.g. "the", "am", "are")
   3. Ambiguous words (e.g. spring(coil) & spring(water-fountain))

**Try to incorporate**:

1. Multiple camera angles
2. Diverse performers to capture all *accents* of the signs
3. Uniqueness in labeling of word tokens
4. Variations in signs for the same concept

</details>
</ol>

### Goals

1. Enable integration of sign language into existing applications.
2. Improve education quality for the deaf and increase litracy rates.
3. Promote communication inclusivity of the deaf.

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

Head over to <span style="font-size:large;">[sign-language-translator.**readthedocs**.io](https://sign-language-translator.readthedocs.io)</span> to see the detailed usage in Python & CLI. <!-- and GUI. -->

See the [*test cases*](https://github.com/sign-language-translator/sign-language-translator/blob/main/tests) or the [*notebooks* repo](https://github.com/sign-language-translator/notebooks) to see the internal code in action.

Also see the [How to build a custom sign language translator](#how-to-build-a-translator-for-sign-language) section.

### Python

```python
# Documentation: https://sign-language-translator.readthedocs.io
import sign_language_translator as slt
# print(slt.TextLanguageCodes, slt.SignLanguageCodes)

# The core model of the project (rule-based text-to-sign translator)
# which enables us to generate synthetic training datasets
model = slt.models.ConcatenativeSynthesis(
   text_language="urdu", sign_language="pk-sl", sign_format="video"
)
text = "یہ بہت اچھا ہے۔" # "This very good is."
sign = model.translate(text) # tokenize, map, download & concatenate
sign.show()
# sign.save(f"{text}.mp4")

model.text_language = "hindi"
sign = model.translate("पाँच घंटे।") # "5 hours."
```

![this very good is](https://github.com/sign-language-translator/sign-language-translator/assets/118578823/7f4ff312-df03-4b11-837b-5fb895c9f08e)

```python
import sign_language_translator as slt

# # Load sign-to-text model (pytorch) (COMING SOON!)
# translation_model = slt.get_model(slt.ModelCodes.Gesture)
embedding_model = slt.models.MediaPipeLandmarksModel()

sign = slt.Video("video.mp4")
embedding = embedding_model.embed(sign.iter_frames())
# text = translation_model.translate(embedding)

# print(text)
sign.show()
# slt.Landmarks(embedding, connections="mediapipe-world").show()
```

```python
# custom translator (https://sign-language-translator.readthedocs.io/en/latest/#building-custom-translators)
help(slt.languages.SignLanguage)
help(slt.languages.text.Urdu)
help(slt.models.ConcatenativeSynthesis)
# help(slt.models.TransformerLanguageModel)
```

### Command Line

```bash
$ slt

Usage: slt [OPTIONS] COMMAND [ARGS]...
   Sign Language Translator (SLT) command line interface.
   Documentation: https://sign-language-translator.readthedocs.io
Options:
  --version  Show the version and exit.
  --help     Show this message and exit.
Commands:
  assets     Assets manager to download & display Datasets & Models.
  complete   Complete a sequence using Language Models.
  embed      Embed Videos Using Selected Model.
  translate  Translate text into sign language or vice versa.
```

**Generate training examples**: write a sentence with a language model and synthesize a sign language video from it with a single command:

```bash
slt translate --model-code rule-based --text-lang urdu --sign-lang pk-sl --sign-format video \
"$(slt complete '<' --model-code urdu-mixed-ngram --join '')"
```

## Languages

<details>
<summary><b>Text Languages</b></summary>

Available Functions:

- Text Normalization
- Tokenization (word, phrase & sentence)
- Token Classification (Tagging)
- Word Sense Disambiguation

| Name                                                                                                                                   | Vocabulary         | Ambiguous tokens | Signs |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ---------------- | ----- |
| [Urdu](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/urdu.py) | 2090 words+phrases | 227              | 790   |
| [Hindi](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/hindi.py) | 34 words+phrases | 5              | 16   |

</details>

<details>
<summary><b>Sign Languages</b></summary>

Available Functions:

- Word & phrase mapping to signs
- Sentence restructuring according to grammar
- Sentence simplification (drop stopwords)

| Name                                                                                                                                                                       | Vocabulary | Dataset | Parallel Corpus                                               |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------- | ------------------------------------------------------------- |
| [Pakistan Sign Language](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/sign/pakistan_sign_language.py) | 789        | 23 hours | n gloss sentences with translations in m text languages |

</details>

## Models

<details>
<summary><b>Translation</b>: Text to sign Language</summary>

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
<summary><b>Sign Embedding/Feature extraction</b>:</summary>

<!-- [Available Trained models]() -->

| Name                                                                                                                                                                                                 | Architecture                                                                                                               | Description                                                                                       | Input format                                                 | Output format                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------- |
| [MediaPipe Landmarks<br>(Pose + Hands)](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/video_embedding/mediapipe_landmarks_model.py) | CNN based pipelines. See Here: [Pose](https://arxiv.org/pdf/2006.10204.pdf), [Hands](https://arxiv.org/pdf/2006.10214.pdf) | Encodes videos into pose vectors (3D world or 2D image) depicting the movements of the performer. | List of numpy images<br/>(n_frames, height, width, channels) | torch.Tensor<br/>(n_frames, n_landmarks * 5) |
</details>

<details>
<summary><b>Data generation</b>: Language Models</summary>

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

See the code at [Build Custom Translator section in ReadTheDocs](https://sign-language-translator.readthedocs.io/en/latest/#building-custom-translators) or in this [notebook](https://github.com/sign-language-translator/notebooks/blob/main/translation/concatenative_synthesis.ipynb). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sign-language-translator/notebooks/blob/main/translation/concatenative_synthesis.ipynb)

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
    ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/cli.py">cli.py</a> <sub><sup>────────────────────────────────────────────────────── `> slt` command line interface</sup></sub>
    ├── <i><b>assets</b></i> <sub><sup>(auto-downloaded)</sup></sub>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-datasets">*</a>
    │
    ├── <b>config</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/assets.py">assets.py</a> <sub><sup>───────────────────────────────────────── download, extract and remove models & datasets</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/enums.py">enums.py</a> <sub><sup>─────────────────────────────────────────── string short codes to identify models & classes</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/settings.py">settings.py</a> <sub><sup>────────────────────────────────────── global variables in repository design-pattern</sup></sub>
    │   ├── <i><a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/urls.json">urls.json</a></i>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/utils.py">utils.py</a>
    │
    ├── <b>languages</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/utils.py">utils.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/vocab.p">vocab.py</a> <sub><sup>─────────────────────────────────────────── reads word mapping datasets</sup></sub>
    │   ├── sign
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/sign/mapping_rules.py">mapping_rules.py</a> <sub><sup>────────────────────── strategy design-pattern for word mapping</sup></sub>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/sign/pakistan_sign_language.py">pakistan_sign_language.py</a>
    │   │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/sign/sign_language.py">sign_language.py</a> <sub><sup>────────────────────── Base class for sign mapping and sentence restructuring</sup></sub>
    │   │
    │   └── text
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/english.py">english.py</a>
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/hindi.py">hindi.py</a>
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/text_language.py">text_language.py</a> <sub><sup>────────────────────── Base class for text normalization, tokenization & tagging</sup></sub>
    │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/urdu.py">urdu.py</a>
    │
    ├── <b>models</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/_utils.py">_utils.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/utils.py">utils.py</a>
    │   ├── language_models
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/abstract_language_model.py">abstract_language_model.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/beam_sampling.py">beam_sampling.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/mixer.py">mixer.py</a> <sub><sup>───────────────────────────────────── wrap multiple models into a single object</sup></sub>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/ngram_language_model.py">ngram_language_model.py</a> <sub><sup>────────── uses hash-tables & frequency to predict next token</sup></sub>
    │   │   └── transformer_language_model
    │   │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/transformer_language_model/layers.py">layers.py</a>
    │   │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/transformer_language_model/model.py">model.py</a> <sub><sup>───────────────────────────── decoder-only transformer with controllable vocabulary</sup></sub>
    │   │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/transformer_language_model/train.py">train.py</a>
    │   │
    │   ├── sign_to_text
    │   ├── text_to_sign
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py">concatenative_synthesis.py</a> <sub><sup>──── join sign clip of each word in text using rules</sup></sub>
    │   │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/t2s_model.py">t2s_model.py</a> <sub><sup>───────────────────────────── Base class</sup></sub>
    │   │
    │   ├── text_embedding
    │   └── video_embedding
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/video_embedding/mediapipe_landmarks_model.py">mediapipe_landmarks_model.py</a> <sub><sup>─ 3D coordinates of points on body</sup></sub>
    │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/video_embedding/video_embedding_model.py">video_embedding_model.py</a> <sub><sup>──────── Base class</sup></sub>
    │
    ├── <b>text</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/metrics.py">metrics.py</a> <sub><sup>──────────────────────────────────────── numeric score techniques</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/preprocess.py">preprocess.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/subtitles.py">subtitles.py</a> <sub><sup>───────────────────────────────────── WebVTT</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/synonyms.py">synonyms.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/tagger.py">tagger.py</a> <sub><sup>────────────────────────────────────────── classify tokens to assist in mapping</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/tokenizer.py">tokenizer.py</a> <sub><sup>───────────────────────────────────── break text into words, phrases, sentences etc</sup></sub>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/utils.py">utils.py</a>
    │
    ├── <b>utils</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/archive.py">archive.py</a> <sub><sup>──────────────────────────────────────── zip datasets</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/arrays.py">arrays.py</a> <sub><sup>────────────────────────────────────────── common interface for numpy.ndarray and torch.Tensor</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/download.py">download.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/parallel.py">parallel.py</a> <sub><sup>────────────────────────────────────── multi-threading</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/tree.py">tree.py</a> <sub><sup>───────────────────────────────────────────── print file hierarchy</sup></sub>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/utils.py">utils.py</a>
    │
    └── <b>vision</b>
        ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/_utils.py">_utils.py</a>
        ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/utils.py">utils.py</a>
        ├── landmarks
        ├── sign
        │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/sign/sign.py">sign.py</a> <sub><sup>────────────────────────────────────── Base class to wrap around sign clips</sup></sub>
        │
        └── video
            ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/video/display.py">display.py</a> <sub><sup>───────────────────────────────── jupyter notebooks inline video & pop-up in CLI</sup></sub>
            ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/video/transformations.py">transformations.py</a> <sub><sup>─────────────────── strategy design-pattern for image augmentation</sup></sub>
            ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/video/video_iterators.py">video_iterators.py</a> <sub><sup>─────────────────── adapter design-pattern for video reading</sup></sub>
            └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/video/video.py">video.py</a>
</pre>

## How to Contribute

<details>
<summary><b>Datasets</b>:</summary>

See our datasets & conventions [here](https://github.com/sign-language-translator/sign-language-datasets).

- Contribute by scraping, compiling, and centralizing video datasets.
- Help with labeling [word mapping datasets](https://github.com/sign-language-translator/sign-language-datasets/tree/main/parallel_texts).
- Establish connections with Academies for the Deaf to collaboratively develop standardized *sign language grammar* and integrate it into the rule-based translators.

</details>

<details>
<summary><b>New Code</b>:</summary>

- Create dedicated sign language classes catering to various regions.
- Develop text language processing classes for diverse languages.
- Experiment with training models using diverse hyper-parameters.
- Don't forget to integrate `string short codes` of your classes and models into **`enums.py`**, and ensure to update functions like `get_model()` and `get_.*_language()`.
- Enhance the codebase with comprehensive docstrings, exemplary usage cases, and thorough test cases.

</details>

<details>
<summary><b>Existing Code</b>:</summary>

- Optimize the codebase by implementing techniques like parallel processing and batching.
- Strengthen the project's documentation with clear docstrings, illustrative usage scenarios, and robust test coverage.
- Contribute to the documentation for [sign-language-translator ReadTheDocs](https://github.com/sign-language-translator/sign-language-translator/blob/main/docs/index.rst) to empower users with comprehensive insights. Currently it needs a template for auto-generated pages.

</details>

<details open>
<summary><b>Product Development</b>:
</summary>

- Engage in the development efforts across [MLOps](), [backend](), [web](), and [mobile]() domains, depending on your expertise and interests.

</details>

## Research Papers & Citation

Stay Tuned!

## Upcoming/Roadmap

<details>
<summary>TEXT_PROCESSING_IMPROVEMENTS: v0.7</summary>

```python
# improve urdu code
# add `hindi.py` (NotImplementedError) + `english.py`
# text embedding models
# synonyms by similarity
```

</details>

<details>
<summary>LANDMARKS_WRAPPER: v0.8</summary>

```python
# landmarks wrapper class
# landmark augmentation

# subtitles
# trim signs before concatenation

# stabilize video batch using landmarks
```

</details>

<details>
<summary>LANGUAGES: v0.9</summary>

```python
# implement NLP classes for English & Hindi
# expand dictionary video data by scraping everything
```

</details>

<details>
<summary>MISCELLANEOUS</summary>

```python
# bugfix: inaccurate num_frames in video file metadata
# improvement: video wrapper class uses list of sources instead of linked list of videos
# video transformations

# clean demonstration notebooks
# * host video dataset online, descriptive filenames, zip extraction
# dataset info table
# sequence diagram for creating a translator
# make deep_translator optional.
# GUI with gradio
```

</details>

<details>
<summary>DEEP_TRANSLATION: v1.X</summary>

```python
# parallel text corpus
# LanguageModel: Dropping space tokens, bidirectional prediction & train on max vocab but mask with supported only when inferring.
# sign to text with custom seq2seq transformer
# sign to text with fine-tuned whisper
# pose vector generation with fine-tuned mBERT
# custom 3DLandmark model (training data = mediapipe's output on activity recognition or any dataset)
# motion transfer
# pose2video: stable diffusion or GAN?
# speech to text
# text to speech
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

This project started in October 2021 as a BS Computer Science final year project with 3 students and 1 supervisor. After 9 months at university, it became a hobby project for Mudassar who has continued it till at least 2024-01-15.

<details>
<summary> Immense gratitude towards: (click to expand)</summary>

- [Mudassar Iqbal](https://github.com/mdsrqbl) for coding the project so far.
- [Rabbia Arshad](https://github.com/Rabbia-Arshad) for help in initial R&D and web development.
- Waqas Bin Abbas for assistance in initial video data collection process.
- Kamran Malik for setting the initial project scope, idea of motion transfer and connecting us with Hamza Foundation.
- [Hamza Foundation](https://www.youtube.com/@pslhamzafoundationacademyf7624/videos) (especially Ms Benish, Ms Rashda & Mr Zeeshan) for agreeing to collaborate and providing their sign dictionary, hearing-impaired performers for data creation, and creating the text2gloss dataset.
- [UrduHack](https://github.com/urduhack/urduhack) for their work on Urdu character normalization.
- [Telha Bilal](https://github.com/TelhaBilal) for help in designing the architecture of some modules.

</details>

## Bonus

Count total number of **lines of code** (Package: **9,960** + Tests: **1,622**):

```bash
git ls-files | grep '\.py' | xargs wc -l
```

**Just for *fun***

```text
Q: What was the deaf student's favorite course?
A: Communication skills
```
