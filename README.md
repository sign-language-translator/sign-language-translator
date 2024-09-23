<div align="center">

# Sign Language Translator ⠎⠇⠞

<img width="61.8%" alt="SLT: Sign Language Translator" src="https://github.com/sign-language-translator/sign-language-translator/assets/118578823/d4723333-3d25-413d-83a1-a4bbdc8da15a">

<br>
</br>

***Build Custom Translators and Translate between Sign Language & Text with AI.***

[![python](https://img.shields.io/pypi/pyversions/sign-language-translator?logo=python)](https://pypi.org/project/sign-language-translator/)
[![PyPi](https://img.shields.io/pypi/v/sign-language-translator?logo=pypi)](https://pypi.org/project/sign-language-translator/)
![GitHub branch check runs](https://img.shields.io/github/check-runs/sign-language-translator/sign-language-translator/main?logo=pytest)
[![codecov](https://codecov.io/gh/sign-language-translator/sign-language-translator/branch/main/graph/badge.svg?precision=1)](https://codecov.io/gh/sign-language-translator/sign-language-translator)
[![Documentation Status](https://img.shields.io/readthedocs/sign-language-translator?logo=readthedocs&)](https://sign-language-translator.readthedocs.io/)<br>
[![GitHub Repo stars](https://img.shields.io/github/stars/sign-language-translator/sign-language-translator?logo=github)](https://github.com/sign-language-translator/sign-language-translator/stargazers)
[![Repository Views per Month](https://img.shields.io/badge/dynamic/xml?url=https%3A%2F%2Fu8views.com%2Fapi%2Fv1%2Fgithub%2Fprofiles%2F118578823%2Fviews%2Fday-week-month-total-count.svg&query=%2F*%5Blocal-name()%3D'svg'%5D%2F*%5Blocal-name()%3D'g'%5D%5B3%5D%2F*%5Blocal-name()%3D'g'%5D%2F*%5Blocal-name()%3D'text'%5D%5B2%5D&label=%F0%9F%91%81%EF%B8%8F%20views%2Fmonth&color=6d96ff&cacheSeconds=5000)](https://u8views.com/github/mdsrqbl)
[![Downloads](https://img.shields.io/pepy/dt/sign_language_translator?color=purple&logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4MDAiIGhlaWdodD0iODAwIiBmaWxsPSJub25lIiB2aWV3Qm94PSIwIDAgMjQgMjQiPjxwYXRoIGZpbGw9IiMwMDAiIGZpbGwtcnVsZT0iZXZlbm9kZCIgZD0iTTkuMiAyLjhjLS4yLjMtLjIuOC0uMiAxLjZWMTFINy44Yy0uOSAwLTEuMyAwLTEuNS4yYS44LjggMCAwIDAtLjMuNmMwIC4zLjMuNiAxIDEuMmw0LjEgNC40LjcuNmEuNy43IDAgMCAwIC40IDBsLjctLjZMMTcgMTNjLjYtLjYuOS0xIC45LTEuMmEuOC44IDAgMCAwLS4zLS42Yy0uMi0uMi0uNi0uMi0xLjUtLjJIMTVWNC40YzAtLjggMC0xLjMtLjItMS42YTEuNSAxLjUgMCAwIDAtLjYtLjZjLS4zLS4yLS44LS4yLTEuNi0uMmgtMS4yYy0uOCAwLTEuMyAwLTEuNi4yYTEuNSAxLjUgMCAwIDAtLjYuNnpNNSAyMWExIDEgMCAwIDAgMSAxaDEyYTEgMSAwIDEgMCAwLTJINmExIDEgMCAwIDAtMSAxeiIgY2xpcC1ydWxlPSJldmVub2RkIi8+PC9zdmc+)](https://pepy.tech/projects/sign-language-translator/)<br>
[![HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%8C%90%20Web%20Demo-%F0%9F%A4%97%20hf.co%2FsltAI-mediumpurple)](https://huggingface.co/sltAI)

| **Support Us** ❤️ | [![PayPal](https://img.shields.io/badge/PayPal-00457C?logo=paypal&logoColor=white)](https://www.paypal.com/donate/?hosted_button_id=7SNGNSKUQXQW2) |
| - | - |

</div>

---

1. [Overview](#overview)
   1. [Solution](#solution)
   2. [Major Components](#major-components)
   3. [Goals](#goals)
2. [**Installation** `🛠️`](#how-to-install-the-package)
3. [**Usage**](#usage)
   1. [Web 🌐](#web-gui)
   2. [Python `🐍`](#python)
   3. [Command Line <span style="color:green">**`>_`**</span>](#command-line)
4. [Languages](#languages)
5. [Models](#models)
6. [How to Build a Translator for your Sign Language](#how-to-build-a-translator-for-sign-language)
7. [Module Hierarchy](#module-hierarchy)
8. [How to Contribute](#how-to-contribute)
9. [Citation, License & Research Papers](#citation-licence--research-papers)
10. [Credits and Gratitude](#credits-and-gratitude)

## Overview

Sign language consists of gestures and expressions used mainly by the hearing-impaired to talk. This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.

<details>
<summary>This python library provides a user-friendly translation API and a framework for building sign language translators that can easily adapt to any regional sign language...</summary>
</br>
A big hurdle is the lack of datasets (global & regional) and frameworks that deep learning engineers and software developers can use to build useful products for the target community. This project aims to empower sign language translation by providing robust components, tools, datasets and models for both sign language to text and text to sign language conversion. It aims to facilitate the creation of sign language translators for any region, while building the way towards sign language standardization.

</br>
Unlike most other projects, this python library can translate full sentences and not just the alphabet.

</details>

<!-- This is the package that powers the [slt_ai website](https://github.com/mdsrqbl/slt_ai). -->

### Solution

This package comes with an *extensible rule-based* text-to-sign translation system that can be used to generate training data for *Deep Learning* models for both sign to text & text to sign translation. <!-- Pipelines for fine-tuning our deep learning models are also available. -->

> [!Tip]
> To create a rule-based translation system for your regional language, you can inherit the TextLanguage and SignLanguage classes and pass them as arguments to the ConcatenativeSynthesis class. To write sample texts of supported words, you can use our language models. Then, you can use that system to fine-tune our deep learning models.

See the <kbd>[documentation](https://sign-language-translator.readthedocs.io)</kbd> and our <kbd>[datasets](https://github.com/sign-language-translator/sign-language-datasets)</kbd> for details.

### Major Components

<ol>
<li>
<details>
<summary><b>
Sign language to Text
</b></summary>

1. Extract features from sign language videos
   1. See the [`slt.models.video_embedding`](https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/video_embedding) sub-package and the [`$ slt embed`](https://slt.readthedocs.io/en/latest/#embed-videos) command.
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
</b></summary>

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
</li>

<li>
<details>
<summary><b>
Language Processing
</b></summary>

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
</li>

<li>
<details>
<summary><b>
Datasets
</b></summary>

For our datasets & conventions, see the [*sign-language-datasets* repo](https://github.com/sign-language-translator/sign-language-datasets) and its [releases](https://github.com/sign-language-translator/sign-language-datasets/releases).
See this [documentation](https://slt.readthedocs.io/en/latest/datasets.html) for more on building a dataset of Sign Language videos (or motion capture gloves' output features).

**Your data should include**:

1. A word level Dictionary (Videos of individual signs & corresponding Text tokens (words & phrases))
2. Replications of the dictionary. (Set up multiple syncronized cameras and record random people performing the dictionary videos. ([notebook](https://colab.research.google.com/github/sign-language-translator/notebooks/blob/main/data_collection/clip_extractor.ipynb)))
3. Parallel sentences
   1. Normal text language sentences against sign language videos. (Use our Language Models to generate sentences composed of dictionary words.)
   2. Normal text language sentences against the [text gloss](https://github.com/sign-language-translator/sign-language-datasets#glossary) of the corresponding sign language sentence.
   3. Sign language sentences against their text gloss
   4. Sign language sentences against translations in multiple text languages
4. Grammatical rules of the sign language
   1. Word order (e.g. SUBJECT OBJECT VERB TIME)
   2. Meaningless words (e.g. "the", "am", "are")
   3. Ambiguous words (e.g. spring(coil) & spring(water-fountain))

**Try to incorporate**:

1. Multiple camera angles
2. Diverse performers to capture all *accents* of the signs
3. Uniqueness in labeling of word tokens
4. Variations in signs for the same concept

Try to capture variations in signs in a scalable and diversity accommodating way and enable advancing sign language standardization efforts.

</details>
</li>
</ol>

### Goals

1. Enable **integration** of sign language into existing applications.
2. Assist construction of **custom** solutions for resource poor sign langauges.
3. Improve **education** quality for the deaf and elevate literacy rates.
4. Promote communication **inclusivity** of the hearing impaired.
5. Establish a framework for sign language **standardization**.

## How to install the package

```bash
pip install sign-language-translator
```

<details>
<summary>Editable mode (<code>git clone</code>):</summary>

The package ships with some optional dependencies as well (e.g. deep_translator for synonym finding and mediapipe for a pretrained pose extraction model). Install them by appending `[all]`, `[full]`, `[mediapipe]` or `[synonyms]` to the project name in the command (e.g `pip install sign-langauge-translator[full]`).

```bash
git clone https://github.com/sign-language-translator/sign-language-translator.git
cd sign-language-translator
pip install -e ".[all]"
```

```bash
pip install -e git+https://github.com/sign-language-translator/sign-language-translator.git#egg=sign_language_translator
```

</details>

## Usage

Head over to [slt.**readthedocs**.io](https://slt.readthedocs.io) to see the detailed usage in Python, CLI and gradio GUI.
See the [*test cases*](https://github.com/sign-language-translator/sign-language-translator/blob/main/tests) or the [*notebooks* repo](https://github.com/sign-language-translator/notebooks) to see the internal code in action.

### Web GUI

Individual models deployed on HuggingFace Spaces:

[![HuggingFace Spaces](https://img.shields.io/badge/text%20to%20sign-ConcatenativeSynthesis%2BLM-mediumslateblue)](https://huggingface.co/spaces/sltAI/ConcatenativeSynthesis)

<!-- #FF3270 #FFD702 #861FFF #097EFF #10B981 #3B82F6 #6366F1 #F59E0B #EF4444 -->

### Python

```python
import sign_language_translator as slt

# The core model of the project (rule-based text-to-sign translator)
# which enables us to generate synthetic training datasets
model = slt.models.ConcatenativeSynthesis(
   text_language="urdu", sign_language="pk-sl", sign_format="video" )

text = "یہ بہت اچھا ہے۔" # "this-very-good-is"
sign = model.translate(text) # tokenize, map, download & concatenate
sign.show()

model.sign_format = slt.SignFormatCodes.LANDMARKS
model.sign_embedding_model = "mediapipe-world"

# ==== English ==== #
model.text_language = slt.languages.text.English()
sign_2 = model.translate("This is an apple.")
sign_2.save("this-is-an-apple.csv", overwrite=True)

# ==== Hindi ==== #
model.text_language = slt.TextLanguageCodes.HINDI
sign_3 = model.translate("कैसे हैं आप?") # "how-are-you"
sign_3.save_animation("how-are-you.gif", overwrite=True)
```

| ![this very good is](https://github.com/sign-language-translator/sign-language-translator/assets/118578823/7f4ff312-df03-4b11-837b-5fb895c9f08e) | <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/4d54a197-d723-4cc4-a3ba-cae98e681003" /><source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/45e71098-7a94-4a9e-ad24-1773369b65d5" /><img alt="how are you (landmark 3d plot)" src="https://github.com/user-attachments/assets/45e71098-7a94-4a9e-ad24-1773369b65d5" /></picture> |
| :-: | :-: |
| "یہ بہت اچھا ہے۔" (this-very-good-is) | "कैसे हैं आप?" (how-are-you) |

</br>

```python
import sign_language_translator as slt

# sign = slt.Video("path/to/video.mp4")
sign = slt.Video.load_asset("pk-hfad-1_aap-ka-nam-kya(what)-hy")  # your name what is? (auto-downloaded)
sign.show_frames_grid()

# Extract Pose Vector for feature reduction
embedding_model = slt.models.MediaPipeLandmarksModel()      # pip install "sign_language_translator[mediapipe]"  # (or [all])
embedding = embedding_model.embed(sign.iter_frames())

slt.Landmarks(embedding.reshape((-1, 75, 5)),
              connections="mediapipe-world"  ).show()

# # Load sign-to-text model (pytorch) (COMING SOON!)
# translation_model = slt.get_model(slt.ModelCodes.Gesture)
# text = translation_model.translate(embedding)
# print(text)
```

```python
# custom translator (https://slt.readthedocs.io/en/latest/#building-custom-translators)
help(slt.languages.SignLanguage)
help(slt.languages.text.Urdu)
help(slt.models.ConcatenativeSynthesis)
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

| Name | Vocabulary | Ambiguous tokens | Signs |
| - | - | - | - |
| [English](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/english.py) | 1591 words+phrases | 167 | 776 |
| [Urdu](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/urdu.py)    | 2080 words+phrases | 227 | 776 |
| [Hindi](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/hindi.py)   |  137 words+phrases |   5 |  84 |

</details>

<details>
<summary><b>Sign Languages</b></summary>

Available Functions:

- Word & phrase mapping to signs
- Sentence restructuring according to grammar
- Sentence simplification (drop stopwords)

| Name | Vocabulary | Dataset | Parallel Corpus |
| - | - | - | :-: |
| [Pakistan Sign Language](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/sign/pakistan_sign_language.py) | 776 | 23 hours | [details](https://github.com/sign-language-translator/sign-language-datasets#datasets) |

</details>

## Models

<details>
<summary><b>Translation</b>: Text to sign Language</summary>

<!-- [Available Trained models]() -->

| Name                                                                                                                                                                              | Architecture        | Description                                                                                                                                            | Input  | Output                     | Web Demo                                                                                                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Concatenative Synthesis](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py) | Rules + Hash Tables | The Core Rule-Based translator mainly used to synthesize translation dataset.<br/>Initialize it using TextLanguage, SignLanguage & SignFormat objects. | string | slt.Video \| slt.Landmarks | [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/sltAI/ConcatenativeSynthesis) |

<!--                                                                                                                                                                              | [pose-gen]()        | Encoder-Decoder Transformers (Seq2Seq)                                                                                                              | Generates a sequence of pose vectors conditioned on input text. | torch.Tensor<br/>(batch, token_ids) | torch.Tensor<br/>(batch, n_frames, n_landmarks*3) |  | -->

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

<details>
<summary><b>Text Embedding</b>:</summary>

[Available Trained models](https://github.com/sign-language-translator/sign-language-datasets/releases/)

| Name                                                                                                                                                                  | Architecture | Description                                                                                                             | Input format | Output format             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------ | ------------------------- |
| [Vector Lookup](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_embedding/vector_lookup_model.py) | HashTable    | Finds token index and returns the coresponding vector. Tokenizes sentences and computes average vector of known tokens. | string       | torch.Tensor<br/>(n_dim,) |

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

   1. Gather a collection of [dictionary videos](https://github.com/sign-language-translator/sign-language-datasets/releases/tag/v0.0.2) (word level) featuring individuals performing sign language gestures. These can be obtained from schools & organizations for the deaf. You should record multiple people perform the same sign to capture various *accents* of the sign. Set up multiple cameras in different locations in parallel to further augment the data.
   2. Prepare a [JSON file](https://github.com/sign-language-translator/sign-language-datasets/blob/main/parallel_texts/pk-dictionary-mapping.json) that maps dictionary video file names to corresponding text language words & phrases that are synonymous with the gestures.
   3. Prepare a synthetic data [parallel corpus](https://github.com/sign-language-translator/sign-language-datasets/blob/main/parallel_texts/pk-synthetic-sentence-mapping.json) containing text language sentences and sequences of sign language video filenames. You can use langauge models to generate these sentences & sequences.
   4. Prepare a dataset of sign language [sentence videos](https://github.com/sign-language-translator/sign-language-datasets/releases/tag/v0.0.3) that are labeled with [translations & glosses](https://github.com/sign-language-translator/sign-language-datasets/blob/main/parallel_texts/pk-sentence-mapping.json) in multiple text languages.

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
<summary>Deep Learning Model Fine-Tuning</summary>

   1. Utilize the (synthetic & real) sign language videos and corresponding text sentences from the previous step.
   2. Apply our training pipeline to fine-tune a chosen model for improved accuracy and translation quality.

</details>
</li>
</ol>

Remember to contribute back to the community:

- Share your data, code, and models by creating a pull request, allowing others to benefit from your efforts.
- Create your own sign language translator (e.g. as your university thesis) and contribute to a more inclusive and accessible world.

See the `code` at [Build Custom Translator section in ReadTheDocs](https://sign-language-translator.readthedocs.io/en/latest/#building-custom-translators) or in this [notebook](https://github.com/sign-language-translator/notebooks/blob/main/translation/concatenative_synthesis.ipynb). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sign-language-translator/notebooks/blob/main/translation/concatenative_synthesis.ipynb)

## Module Hierarchy

<details>
<summary><b style="font-size:large;"><code>sign-language-translator</code></b> (Click to see file descriptions)</summary>

<pre>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/README.md">README.md</a>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/pyproject.toml">pyproject.toml</a>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/requirements.txt">requirements.txt</a>
├── <b>docs</b>
│   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/docs">*</a>
├── <b>tests</b>
│   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/tests">*</a>
│
└── <b style="font-size:large;">sign_language_translator</b>
    ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/cli.py">cli.py</a> <sub><sup>`> slt` command line interface</sup></sub>
    ├── <i><b>assets</b></i> <sub><sup>(auto-downloaded)</sup></sub>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-datasets">*</a>
    │
    ├── <b>config</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/assets.py">assets.py</a> <sub><sup>download, extract and remove models & datasets</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/colors.py">colors.py</a> <sub><sup>named RGB tuples for visualization</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/enums.py">enums.py</a> <sub><sup>string short codes to identify models & classes</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/settings.py">settings.py</a> <sub><sup>global variables in repository design-pattern</sup></sub>
    │   ├── <i><a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/urls.json">urls.json</a></i>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/utils.py">utils.py</a>
    │
    ├── <b>languages</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/utils.py">utils.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/vocab.p">vocab.py</a> <sub><sup>reads word mapping <a href="https://github.com/sign-language-translator/sign-language-datasets/tree/main/parallel_texts">datasets</a></sup></sub>
    │   ├── sign
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/sign/mapping_rules.py">mapping_rules.py</a> <sub><sup>strategy design-pattern for word to sign mapping</sup></sub>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/sign/pakistan_sign_language.py">pakistan_sign_language.py</a>
    │   │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/sign/sign_language.py">sign_language.py</a> <sub><sup>Base class for text to sign mapping and sentence restructuring</sup></sub>
    │   │
    │   └── text
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/english.py">english.py</a>
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/hindi.py">hindi.py</a>
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/text_language.py">text_language.py</a> <sub><sup>Base class for text normalization, tokenization & tagging</sup></sub>
    │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/urdu.py">urdu.py</a>
    │
    ├── <b>models</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/_utils.py">_utils.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/utils.py">utils.py</a>
    │   ├── language_models
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/abstract_language_model.py">abstract_language_model.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/beam_sampling.py">beam_sampling.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/mixer.py">mixer.py</a> <sub><sup>wrap multiple language models into a single object</sup></sub>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/ngram_language_model.py">ngram_language_model.py</a> <sub><sup>uses hash-tables & frequency to predict next token</sup></sub>
    │   │   └── transformer_language_model
    │   │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/transformer_language_model/layers.py">layers.py</a>
    │   │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/transformer_language_model/model.py">model.py</a> <sub><sup>decoder-only transformer with controllable vocabulary</sup></sub>
    │   │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/transformer_language_model/train.py">train.py</a>
    │   │
    │   ├── sign_to_text
    │   ├── text_to_sign
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py">concatenative_synthesis.py</a> <sub><sup>join sign clip of each word in text using rules</sup></sub>
    │   │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/t2s_model.py">t2s_model.py</a> <sub><sup>Base class</sup></sub>
    │   │
    │   ├── text_embedding
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_embedding/text_embedding_model.py">text_embedding_model.py</a> <sub><sup>Base class</sup></sub>
    │   │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_embedding/vector_lookup_model.py">vector_lookup_model.py</a> <sub><sup>retrieves word embedding from a vector database</sup></sub>
    │   │
    │   └── video_embedding
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/video_embedding/mediapipe_landmarks_model.py">mediapipe_landmarks_model.py</a> <sub><sup>2D & 3D coordinates of points on body</sup></sub>
    │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/video_embedding/video_embedding_model.py">video_embedding_model.py</a> <sub><sup>Base class</sup></sub>
    │
    ├── <b>text</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/metrics.py">metrics.py</a> <sub><sup>numeric score techniques</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/preprocess.py">preprocess.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/subtitles.py">subtitles.py</a> <sub><sup>WebVTT</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/synonyms.py">synonyms.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/tagger.py">tagger.py</a> <sub><sup>classify tokens to assist in mapping</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/tokenizer.py">tokenizer.py</a> <sub><sup>break text into words, phrases, sentences etc</sup></sub>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/text/utils.py">utils.py</a>
    │
    ├── <b>utils</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/archive.py">archive.py</a> <sub><sup>zip datasets</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/arrays.py">arrays.py</a> <sub><sup>common interface & operations for numpy.ndarray and torch.Tensor</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/download.py">download.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/parallel.py">parallel.py</a> <sub><sup>multi-threading</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/tree.py">tree.py</a> <sub><sup>print file hierarchy</sup></sub>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/utils/utils.py">utils.py</a>
    │
    └── <b>vision</b>
        ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/_utils.py">_utils.py</a>
        ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/utils.py">utils.py</a>
        ├── landmarks
        │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/landmarks/connections.py">connections.py</a> <sub><sup>drawing configurations for different landmarks models</sup></sub>
        │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/landmarks/display.py">display.py</a> <sub><sup>visualize points & lines on 3D plot</sup></sub>
        │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/landmarks/landmarks.py">landmarks.py</a> <sub><sup>wrapper for sequence of collection of points on body</sup></sub>
        │
        ├── sign
        │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/sign/sign.py">sign.py</a> <sub><sup>Base class to wrap around sign clips</sup></sub>
        │
        └── video
            ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/video/display.py">display.py</a> <sub><sup>jupyter notebooks inline video & pop-up in CLI</sup></sub>
            ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/video/transformations.py">transformations.py</a> <sub><sup>strategy design-pattern for image augmentation</sup></sub>
            ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/video/video_iterators.py">video_iterators.py</a> <sub><sup>adapter design-pattern for video reading</sup></sub>
            └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/vision/video/video.py">video.py</a>
</pre>

</details>

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
- Don't forget to integrate `string short codes` of your classes and models into **`enums.py`**, and ensure to update factory functions like `get_model()` and `get_.*_language()`.
- Enhance the codebase with comprehensive docstrings, exemplary usage cases, and thorough test cases.

</details>

<details open>
<summary><b>Existing Code</b>:</summary>

- Implement the `# ToDo` from the code or fix `# bug` / `# type: ignore` or anything from the [roadmap](#upcomingroadmap).
- Optimize the codebase by implementing techniques like parallel processing and batching.
- Strengthen the project with clear docstrings containing illustrative examples and with robust test coverage.
- Contribute to the documentation for [sign-language-translator ReadTheDocs](https://github.com/sign-language-translator/sign-language-translator/blob/main/docs/index.rst) to empower users with comprehensive insights. Currently it needs a **better template** for the auto-generated pages.

</details>

<details>
<summary><b>Product Development</b>:
</summary>

- Engage in the development efforts across [MLOps](https://huggingface.co/sltAI) & [web-frontend](https://github.com/sign-language-translator/slt-frontend) <!-- [backend](https://github.com/sign-language-translator/slt-backend), and [mobile](https://github.com/sign-language-translator/slt-mobile) --> domains, depending on your expertise and interests.

</details>

## Upcoming/Roadmap

<details open>
<summary>LANDMARKS_WRAPPER: v0.8</summary>

```python
# 0.8.2: landmark augmentation (zoom, rotate, move, noise, duration, rectify, stabilize, __repr__)
# 0.8.3: trim signs before concatenation, insert transition frames
# 0.8.4: plotly & three.js/mixamo display , pass matplotlib kwargs all the way down

# 0.8.5: subtitles/captions
# 0.8.6: stabilize video batch using landmarks, draw/overlay 2D landmarks on video/image
```

</details>

<details>
<summary>CLEAN_UP: v0.9</summary>

```python
# mock test cases which require internet when internet isn't available / test for  dummy languages
# improve langauge classes architecture (for easy customization via inheritance) | clean-up slt.languages.text.* code
# ? add a generic SignedTextLanguage class which just maps text lang to signs based on mappinng.json ?
# add progress bar to slt.models.MediaPipeLandmarksModel

# rename 'country' to 'region' & rename wordless_wordless to wordless.mp4 # insert video type to archives: .*.videos-`(dictionary|sentences)(-replication)?`-mp4.zip
# decide mediapipe-all = world & image concactenated in landmark dim or feature dim?
# expand dictionary video data by scraping everything
# upload the 12 person dictionary replication landmark dataset
```

</details>

<details>
<summary>DEEP_TRANSLATION: v0.9 - v1.2</summary>

```python
# 0.9.1: TransformerLanguageModel - Drop space tokens & bidirectional prediction. infer on specific vocab only .... pretrain on max vocab and mixed data. finetune on balanced data (wiki==news==novels==poetry==reviews) .... then RLHF on coherent generations (Comparison data: generate 100 examples (at high temperature) and cut them at random points and regerate the rest and label these pairs for coherence[ and novelity].) (use same model/BERT as reward model with regression head.) (ranking loss with margin) (each token is a time step) (min KL Divergance from base - exploration without mode collapse) ... label disambiguation data and freeze all and finetune disambiguated_tokens_embeddings  (disambiguated embedding: word ± 0.1*(sense1 - sense2).normalize()) .... generate data on broken compound words and finetune their token_embeddings ... generate sentences of supported words and translate to other languages.
# 0.9.2: sign to text with custom seq2seq transformer
# 0.9.3: pose vector generation from text with custom seq2seq transformer
# 0.9.4: sign to text with fine-tuned whisper
# 0.9.5: pose vector generation with fine-tuned mBERT
# 0.9.6: custom 3DLandmark model (training data = mediapipe's output on activity recognition or any dataset)
# 1.0.0: all models trained on custom landmark model
# 🎉
# 1.0.1: video to text model (connect custom landmark model with sign2text model and finetune)
# 1.1.0: motion transfer
# 1.1.1: custom pose2video: stable diffusion or GAN?
# 1.2.0: speech to sign
# 1.2.1: sign to speech
```

</details>

<details>
<summary>MISCELLANEOUS</summary>

Issues

```python
# bugfix:      inaccurate num_frames in video file metadata
# bugfix:      Expression of type "Literal[False]" cannot be assigned to member "SHOW_DOWNLOAD_PROGRESS" of class "Settings"
# feature:     video transformations (e.g. stabilization with image pose landmarks, watermark text/logo)
# improvement: SignFilename.parse("videos/pk-hfad-1_airplane.mp4").gloss  # airplane
```

Miscellaneous

```python
# parallel text corpus
# clean demonstration notebooks
# * host video dataset online, descriptive filenames
# dataset info table
# sequence diagram for creating a translator
# GUI with gradio or something
```

Research Papers

```python
# datasets: clips, text, sentences, disambiguation
# rule based translation: describe entire repo
# deep sign-to-text: pipeline + experiments
# deep text-to-sign: pipeline + experiments
```

Servers / Product

```python
# ML inference server
# Django backend server
# React Native mobile app
```

[![Total Views](https://u8views.com/api/v1/github/profiles/118578823/views/total-count.svg)](https://u8views.com/github/mdsrqbl)

</details>

## Citation, Licence & Research Papers

```bibtex
@software{mdsr2023slt,
  author       = {Mudassar Iqbal},
  title        = {Sign Language Translator: Python Library and AI Framework},
  year         = {2023},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/sign-language-translator/sign-language-translator}},
}
```

This project is licensed under the [Apache 2.0 License](https://github.com/sign-language-translator/sign-language-translator/blob/main/LICENSE). You are permitted to use the library, create modified versions, or incorporate pieces of the code into your own work. Your product or research, whether commercial or non-commercial, must provide appropriate credit to the original author(s) by citing this repository.

Stay Tuned for research Papers!

## Credits and Gratitude

This project started in October 2021 as a BS Computer Science final year project with 3 students and 1 supervisor. After 9 months at university, it became a hobby project for [Mudassar](https://github.com/mdsrqbl) who has continued it till at least 2024-09-23.

## Bonus

Count total number of **lines of code** (Package: **14,034** + Tests: **2,928**):

```bash
git ls-files | grep '\.py' | xargs wc -l
```

**Just for *fun* 🙃**

```text
Q: What was the deaf student's favorite course?
A: Communication skills
```

```text
Q: Why was the ML engineer sad?
A: Triplet loss
```

<details>
<summary><b>Star History</b></summary>

<a href="https://star-history.com/#sign-language-translator/sign-language-translator&Timeline">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=sign-language-translator/sign-language-translator&type=Timeline&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=sign-language-translator/sign-language-translator&type=Timeline" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=sign-language-translator/sign-language-translator&type=Timeline" />
  </picture>
</a>

</details>
