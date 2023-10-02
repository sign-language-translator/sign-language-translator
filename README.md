# Sign Language Translator ⠎⠇⠞

[![python](https://img.shields.io/pypi/pyversions/sign-language-translator)](https://pypi.org/project/sign-language-translator/)
[![PyPi](https://img.shields.io/pypi/v/sign-language-translator)](https://pypi.org/project/sign-language-translator/)
[![Downloads](https://static.pepy.tech/personalized-badge/sign-language-translator?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/sign-language-translator/)

![Release Workflow Status](https://img.shields.io/github/actions/workflow/status/sign-language-translator/sign-language-translator/release.yml?branch=main)
[![codecov](https://codecov.io/gh/sign-language-translator/sign-language-translator/branch/main/graph/badge.svg)](https://codecov.io/gh/sign-language-translator/sign-language-translator)
[![Documentation Status](https://readthedocs.org/projects/sign-language-translator/badge/?version=latest)](https://sign-language-translator.readthedocs.io/en/latest/?badge=latest)

1. [Overview](#overview)
   1. [Solution](#solution)
   2. [Major Components and Goals](#major-components-and-goals)
2. [How to install the package](#how-to-install-the-package)
3. [Usage](#usage)
   1. [Command Line](#command-line)
      <!-- 1. [configure](#configure) -->
      1. [$ slt download ...](#download)
      2. [$ slt translate ...](#translate)
      3. [$ slt complete ...](#complete)
      4. [$ slt embed ...](#embed-videos)
   2. [Python](#python)
      1. [Translation](#basics)
      2. [Text language processor](#text-language-processor)
      3. [Sign language processor](#sign-language-processor)
      4. [Video processing](#vision)
      5. [Language models](#language-models)
4. [Models](#models)
5. [How to Build a Translator for your Sign Language](#how-to-build-a-translator-for-sign-language)
6. [Directory Tree](#directory-tree)
7. [How to Contribute](#how-to-contribute)
8. [Research Papers & Citation](#research-papers--citation)
9. [Upcoming/Roadmap](#upcomingroadmap)
10. [Credits and Gratitude](#credits-and-gratitude)
11. [Bonus](#bonus)
    1. Number of lines of code
    2. :)

## Overview

Sign language consists of gestures and expressions used mainly by the hearing-impaired to talk. This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.

The goal is to provide a user friendly API to novel Sign Language Translation solutions that can easily adapt to any regional sign language. Unlike most other projects, this python library can translate full sentences and not just the alphabet.
<!-- This is the package that powers the [slt_ai website](https://github.com/mdsrqbl/slt_ai). -->

A bigger hurdle is the lack of datasets and frameworks that deep learning engineers and software developers can use to build useful products for the target community. This project aims to empower sign language translation by providing robust components, tools and models for both sign language to text and text to sign language conversion. It seeks to advance the development of sign language translators for various regions while providing a way towards sign language standardization.

### Solution

We've have built an *extensible rule-based* text-to-sign translation system that can be used to generate training data for *Deep Learning* models for both sign to text & text to sign translation.

To create a rule-based translation system for your regional language, you can inherit the TextLanguage and SignLanguage classes and pass them as arguments to the ConcatenativeSynthesis class. To write sample texts of supported words, you can use our language models. Then, you can use that system to fine-tune our AI models.

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

See the [*test cases*](https://github.com/sign-language-translator/sign-language-translator/tree/main/tests) or [the *notebooks* repo](https://github.com/sign-language-translator/notebooks) for detailed use
but here is the general API:

### Command Line

You can use the following functionalities of the SLT package via CLI as well. A command entered without any arguments will show the help. *The useable model-codes are listed in help*.<br>
Note: Objects & models do not persist in memory across commands, so this is a quick but inefficient way to use this package. In production, create a server which uses the python interface.

#### Download

Download dataset files or models if you need them. The parameters are regular expressions.

```bash
slt download --overwrite true '.*\.json' '.*\.mp4'
```

```bash
slt download --progress-bar true '.*/tlm_14.0M.pt'
```

By default, auto-download is enabled. Default download directory is `/install-directory/sign_language_translator/sign-language-resources/`. (See slt.config.settings.Settings)

#### Translate

Translate text to sign language using a rule-based model

```bash
slt translate \
--model-code "concatenative" \
--text-lang urdu --sign-lang psl \
--sign-format 'mediapipe-landmarks' \
"وہ سکول گیا تھا۔" \
'مجھے COVID نہیں ہے!'
```

#### Complete

Auto complete a sentence using our language models. This model can write sentences composed of supported words only:

```bash
$ slt complete --end-token ">" --model-code urdu-mixed-ngram "<"
('<', 'وہ', ' ', 'یہ', ' ', 'نہیں', ' ', 'چاہتا', ' ', 'تھا', '۔', '>')
```

<details>
<summary>These models predict next characters until a specified token appears. (e.g. generating names using a mixture of models):</summary>

```bash
$ slt complete \
    --model-code unigram-names --model-weight 1 \
    --model-code bigram-names -w 2 \
    -m trigram-names -w 3 \
    --selection-strategy merge --beam-width 2.5 --end-token "]" \
    "[s"
[shazala]
```

</details>

#### Embed Videos

Embed videos into a sequence of vectors using selected embedding models.

```bash
slt embed videos/*.mp4 --model-code mediapipe-pose-2-hand-1 --embedding-type world --processes 4 --save-format csv
```

### Python

#### Basics

```python
import sign_language_translator as slt

# download dataset or models (if you need them for personal use)
# (by default, resources are auto-downloaded within the install directory)
# slt.set_resource_dir("path/to/folder")  # Helps preventing duplication across environments or using cloud synced data
# slt.utils.download_resource(".*.json")  # downloads into resource_dir
# print(slt.Settings.FILE_TO_URL.keys())  # All downloadable resources

print("All available models:")
print(list(slt.ModelCodes))  # slt.ModelCodeGroups
# print(list(slt.TextLanguageCodes))
# print(list(slt.SignLanguageCodes))
# print(list(slt.SignFormatCodes))
```

**Text to Sign Translation**:

```python
# Load text-to-sign model
# deep_t2s_model = slt.get_model("t2s-flan-T5-base-01.pt") # pytorch
# rule-based model (concatenates clips of each word)
t2s_model = slt.get_model(
    model_code = "concatenative-synthesis", # slt.ModelCodes.CONCATENATIVE_SYNTHESIS
    text_language = "urdu", # or object of any child of slt.languages.text.text_language.TextLanguage class
    sign_language = "pakistan-sign-language", # or object of any child of slt.languages.sign.sign_language.SignLanguage class
    sign_format = "video", # or object of any child of slt.vision.sign_wrappers.sign.Sign class
)

text = "HELLO دنیا!" # HELLO treated as an acronym
sign_language_sentence = t2s_model(text)

# slt_video_object.show() # class: slt.vision.sign_wrappers.video.Video
# slt_video_object.save(f"sentences/{text}.mp4")
```

**Sign to Text Translation**

<details>
<summary>dummy code: (will be finalized in v0.8+)</summary>

```python
# load sign
video = slt.Video("video.mp4")
# features = slt.extract_features(video, "mediapipe_pose_v2_hand_v1")

# Load sign-to-text model
deep_s2t_model = slt.get_model("gesture_mp_base-01") # pytorch

# translate via single call to pipeline
# text = deep_s2t_model.translate(video)

# translate via individual steps
features = deep_s2t_model.extract_features(video.iter_frames())
encoding = deep_s2t_model.encoder(features)
# logits = deep_s2t_model.decoder(encoding, token_ids = [0])
# logits = deep_s2t_model.decoder(encoding, token_ids = [0, logits.argmax(dim=-1)])
# ...
tokens = deep_s2t_model.decode(encoding) # uses beam search to generate a token sequence
text = "".join(tokens) # deep_s2t_model.detokenize(tokens)

print(features.shape)
print(logits.shape)
print(text)
```

</details>

#### Text Language Processor

Process text strings using language specific classes:

```python
from sign_language_translator.languages.text import Urdu
ur_nlp = Urdu()

text = "hello جاؤں COVID-19."

normalized_text = ur_nlp.preprocess(text)
# normalized_text = 'جاؤں COVID-19.' # replace/remove unicode characters

tokens = ur_nlp.tokenize(normalized_text)
# tokens = ['جاؤں', ' ', 'COVID', '-', '19', '.']

# tagged = ur_nlp.tag(tokens)
# tagged = [('جاؤں', Tags.SUPPORTED_WORD), (' ', Tags.SPACE), ...]

tags = ur_nlp.get_tags(tokens)
# tags = [Tags.SUPPORTED_WORD, Tags.SPACE, Tags.ACRONYM, ...]

# word_senses = ur_nlp.get_word_senses("میں")
# word_senses = [["میں(i)", "میں(in)"]]
```

#### Sign Language Processor

This processes the text representation of sign language which mainly deals with the video file names. For video processing, see [vision](#vision) section.

```python
from sign_language_translator.languages.sign import PakistanSignLanguage

psl = PakistanSignLanguage()

tokens = ["he", " ", "went", " ", "to", " ", "school", "."]
tags = 3 * [Tags.WORD, Tags.SPACE] + [Tags.WORD, Tags.PUNCTUATION]
tokens, tags, _ = psl.restructure_sentence(tokens, tags) # ["he", "school", "go"]
signs  = psl.tokens_to_sign_dicts(tokens, tags)
# signs = [
#   {'signs': [['pk-hfad-1_وہ']], 'weights': [1.0]},
#   {'signs': [['pk-hfad-1_school']], 'weights': [1.0]},
#   {'signs': [['pk-hfad-1_گیا']], 'weights': [1.0]}
# ]
```

#### Vision

<details>
<summary>dummy code: (will be finalized in v0.7)</summary>

```python
import sign_language_translator as slt

# load video
video = slt.Video("sign.mp4")
print(video.duration(), video.shape)

# extract features
# model = slt.get_model(slt.ModelCodes.MEDIAPIPE_POSE_V2_HAND_V1)
model = slt.models.MediaPipeLandmarksModel()  # default args
embedding = model.embed(video.frames(), landmark_type="world") # torch.Tensor
print(embedding.shape)  # (n_frames, n_landmarks * 5)

# embed dataset
# slt.models.utils.VideoEmbeddingPipeline(model).process_videos_parallel(
#     ["dataset/*.mp4"], n_processes=12, save_format="csv", ...
# )

# transform / augment data
sign = slt.MediaPipeSign(embedding, landmark_type="world")
sign = sign.rotate(60, 10, 90, degrees=True)
sign = sign.transform(slt.vision.transformations.ZoomLandmarks(1.1, 0.9, 1.0))

# plot
video_visualization = sign.video()
image_visualization = sign.image(steps=5)
overlay_visualization = sign.overlay(video)

# display
video_visualization.show()
image_visualization.show()
overlay_visualization.show()
```

</details>

#### Language models

<details>
<summary>Simple statistical n-gram model:</summary>

```python
from sign_language_translator.models.language_models import NgramLanguageModel

names_data = [
    '[abeera]', '[areej]',  '[farida]',  '[hiba]',    '[kinza]',
    '[mishal]', '[nimra]',  '[rabbia]',  '[tehmina]', '[zoya]',
    '[amjad]',  '[atif]',   '[farhan]',  '[huzaifa]', '[mudassar]',
    '[nasir]',  '[rizwan]', '[shahzad]', '[tayyab]',  '[zain]',
]

# train an n-gram model (considers previous n tokens to predict)
model = NgramLanguageModel(window_size=2, unknown_token="")
model.fit(names_data)

# inference loop
name = '[r'
for _ in range(10):
    nxt, prob = model.next(name) # selects next token randomly from learnt probability distribution
    name += nxt
    if nxt in [']' , model.unknown_token]:
        break
print(name)
# '[rabeej]'

# see ngram model's implementation
print(model.__dict__)
```

</details>

<details>
<summary>Mash up multiple language models & complete generation through beam search:</summary>

```python
from sign_language_translator.models.language_models import MixerLM, BeamSampling, NgramLanguageModel

# using data from previous example
names_data = [...] # or slt.languages.English().vocab.person_names # concat start/end symbols

# train models
SLMs = [
    NgramLanguageModel(window_size=size, unknown_token="")
    for size in range(1,4)
]
[lm.fit(names_data) for lm in SLMs]

# randomly select a model and infer through it
mixed_model = MixerLM(
    models=SLMs,
    selection_probabilities=[1,2,4],
    unknown_token="",
    model_selection_strategy = "choose", # "merge"
)
print(mixed_model)
# Mixer LM: unk_tok=""[3]
# ├── Ngram LM: unk_tok="", window=1, params=85 | prob=14.3%
# ├── Ngram LM: unk_tok="", window=2, params=113 | prob=28.6%
# └── Ngram LM: unk_tok="", window=3, params=96 | prob=57.1%

# use Beam Search to find high likelihood names
sampler = BeamSampling(mixed_model, beam_width=3) #, scoring_function = ...)
name = sampler.complete('[')
print(name)
# [rabbia]
```

</details>

<details open>
<summary>Write sentences composed of only those words for which sign videos are available so that the rule-based text-to-sign model can generate training examples for a deep learning model:</summary>

```python
from sign_language_translator.models.language_models import TransformerLanguageModel

# model = slt.get_model("ur-supported-gpt")
model = TransformerLanguageModel.load("models/tlm_14.0M.pt")
# sampler = BeamSampling(model, ...)
# sampler.complete(["<"])

# see probabilities of all tokens
model.next_all(["میں", " ", "وزیراعظم", " ",])
# (["سے", "عمران", ...], [0.1415926535, 0.7182818284, ...])
```

</details>

## Models

<details>
<summary style="font-weight:bold;">Translation: Text to sign Language</summary>

<!-- [Available Trained models]() -->

| Name                                                                                                                                                                              | Architecture        | Description                                                                                                                                         | Input                                                           | Output                              |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------- |
| [Concatenative Synthesis](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py) | Rules + Hash Tables | The Core Rule-Based translator mainly used to synthesize translation dataset.<br/>Initialize it using TextLanguage, SignLanguage & SignFormat objects. | string                                                          | slt.SignFile                        |

<!--                                                                                                                                                                              | [pose-gen]()        | Encoder-Decoder Transformers (Seq2Seq)                                                                                                              | Generates a sequence of pose vectors conditioned on input text. | torch.Tensor<br/>(batch, token_ids) | torch.Tensor<br/>(batch, n_frames, n_landmarks*3) | -->

</details>

<!-- <details>
<summary>Translation: Sign Language to Text</summary>

[Available Trained models]()

| Name        | Architecture                                | Description                                                                                                  | Input format                                         | Output format                       |
| ----------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- | ----------------------------------- |
| [gesture]() | CNN+Encoder - Decoder Transformer (seq2seq) | Encodes the pose vectors depicting sign language sentence and generates text conditioned on those encodings. | torch.Tensor<br/>(batch, n_frames=1000, n_landmarks * 3) | torch.Tensor<br/>(batch, token_ids) |
</details> -->

<!--
<details>
<summary>Video: Synthesis/Generation</summary>

[Available Trained models]()

| Name        | Architecture                                | Description                                                                                                  | Input format                                         | Output format                       |
| ----------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- | ----------------------------------- |
| [gesture]() | CNN+Encoder - Decoder Transformer (seq2seq) | Encodes the pose vectors depicting sign language sentence and generates text conditioned on those encodings. | torch.Tensor<br/>(batch, n_frames=1000, n_landmarks * 3) | torch.Tensor<br/>(batch, token_ids) |
</details>
-->

<details>
<summary style="font-weight:bold;">Video: Embedding/Feature extraction</summary>

<!-- [Available Trained models]() -->

| Name                        | Architecture                                                                                                               | Description                                                                                                  | Input format                                         | Output format                       |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- | ----------------------------------- |
| [MediaPipe Landmarks<br>(Pose + Hands)](https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/video_embedding/mediapipe_landmarks_model.py) | CNN based pipelines. See Here: [Pose](https://arxiv.org/pdf/2006.10204.pdf), [Hands](https://arxiv.org/pdf/2006.10214.pdf) | Encodes videos into pose vectors (3D world or 2D image) depicting the movements of the performer. | List of numpy images<br/>(n_frames, height, width, channels) | torch.Tensor<br/>(n_frames, n_landmarks * 5) |
</details>

<details>
<summary style="font-weight:bold;">Data generation: Language Models</summary>

[Available Trained models](https://github.com/sign-language-translator/sign-language-datasets/releases/tag/v0.0.1)

| Name                           | Architecture                    | Description                                                                  | Input format                                                 | Output format                                                        |
| ------------------------------ | ------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------- |
| [N-Gram Langauge Model](https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/ngram_language_model.py)      | Hash Tables                     | Predicts the next token based on learned statistics about previous N tokens. | List of tokens                                               | (token, probability)                                                 |
| [Transformer Language Model](https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/transformer_language_model/model.py) | Decoder-only Transformers (GPT) | Predicts next token using query-key-value attention, linear transformations and soft probabilities.   | torch.Tensor<br/>(batch, token_ids)<br/><br/>List of tokens | torch.Tensor<br/>(batch, token_ids, vocab_size)<br/><br/>(token, probability) |
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

## Directory Tree

<pre>
<b style="font-size:large;">sign-language-translator</b>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/MANIFEST.in">MANIFEST.in</a>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/README.md">README.md</a>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/poetry.lock">poetry.lock</a>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/pyproject.toml">pyproject.toml</a>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/requirements.txt">requirements.txt</a>
├── <b>tests</b>
│   └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/tests">*</a>
│
└── <b style="font-size:large;">sign_language_translator</b>
    ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/cli.py">cli.py</a>
    ├── <b>config</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/enums.py">enums.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/helpers.py">helpers.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/settings.py">settings.py</a>
    │   └── <i><a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/urls.yaml">urls.yaml</a></i>
    │
    ├── <b>data_collection</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/data_collection/completeness.py">completeness.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/data_collection/scraping.py">scraping.py</a>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/data_collection/synonyms.py">synonyms.py</a>
    │
    ├── <b>languages</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/languages/utils.py">utils.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/languages/vocab.p">vocab.py</a>
    │   ├── sign
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/languages/sign/mapping_rules.py">mapping_rules.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/languages/sign/pakistan_sign_language.py">pakistan_sign_language.py</a>
    │   │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/languages/sign/sign_language.py">sign_language.py</a>
    │   │
    │   └── text
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/languages/text/english.py">english.py</a>
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/languages/text/text_language.py">text_language.py</a>
    │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/languages/text/urdu.py">urdu.py</a>
    │
    ├── <b>models</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/_utils.py">_utils.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/utils.py">utils.py</a>
    │   ├── language_models
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/abstract_language_model.py">abstract_language_model.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/beam_sampling.py">beam_sampling.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/mixer.py">mixer.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/ngram_language_model.py">ngram_language_model.py</a>
    │   │   └── transformer_language_model
    │   │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/transformer_language_model/layers.py">layers.py</a>
    │   │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/transformer_language_model/model.py">model.py</a>
    │   │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/transformer_language_model/train.py">train.py</a>
    │   │
    │   ├── sign_to_text
    │   ├── text_to_sign
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py">concatenative_synthesis.py</a>
    │   │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/text_to_sign/t2s_model.py">t2s_model.py</a>
    │   │
    │   └── video_embedding
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/video_embedding/mediapipe_landmarks_model.py">mediapipe_landmarks_model.py</a>
    │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/video_embedding/video_embedding_model.py">video_embedding_model.py</a>
    │
    ├── <i><b>sign-language-resources</b></i> (auto-downloaded)
    │   └── <a href="https://github.com/sign-language-translator/sign-language-datasets">*</a>
    │
    ├── <b>text</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/text/metrics.py">metrics.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/text/preprocess.py">preprocess.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/text/subtitles.py">subtitles.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/text/tagger.py">tagger.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/text/tokenizer.py">tokenizer.py</a>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/text/utils.py">utils.py</a>
    │
    ├── <b>utils</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/utils/download.py">download.py</a>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/utils/tree.py">tree.py</a>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/utils/utils.py">utils.py</a>
    │
    └── <b>vision</b>
        └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/vision/utils.py">utils.py</a>
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
- Contribute to the documentation for [sign-language-translator.readthedocs.io](https://sign-language-translator.readthedocs.io) to empower users with comprehensive insights.

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
# class according to feature type:
   # landmarks
# video transformations
# landmark augmentation
# concatenative synthesis returns features
# subtitles
# make scraping dependencies optional (beautifulsoup4, deep_translator)
# GUI with gradio
```

</details>

<details>
<summary>MISCELLANEOUS</summary>

```python
# clean demonstration notebooks
# expand reference clip data by scraping everything
# data info table
# https://sign-language-translator.readthedocs.io/en/latest/
# sequence diagram for creating a translator
```

</details>

<details>
<summary>DEEP_TRANSLATION: v0.8-v1.x</summary>

```python
# sign to text with fine-tuned whisper
# pose vector generation with fine-tuned flan-T5
# motion transfer
# pose2video: stable diffusion or GAN?
# speech to text
# text to speech
# LanguageModel: experiment by dropping space tokens
# parallel text corpus
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

This project started in October 2021 as a BS Computer Science final year project with 3 students and 1 supervisor. After 9 months at university, it became a hobby project for Mudassar who has continued it till at least 2023-09-18.

<details>
<summary> Immense gratitude towards: (click to expand)</summary>

- [Mudassar Iqbal](https://github.com/mdsrqbl) for coding the project so far.
- Rabbia Arshad for help in initial R&D and web development.
- Waqas Bin Abbas for assistance in initial video data collection process.
- Kamran Malik for setting the initial project scope, idea of motion transfer and connecting us with Hamza Foundation.
- [Hamza Foundation](https://www.youtube.com/@pslhamzafoundationacademyf7624/videos) (especially Ms Benish, Ms Rashda & Mr Zeeshan) for agreeing for collaboration and providing the reference clips, hearing-impaired performers for data creation, and creating the text2gloss dataset.
- [UrduHack](https://github.com/urduhack/urduhack) (espacially Ikram Ali) for their work on Urdu character normalization.

- [Telha Bilal](https://github.com/TelhaBilal) for help in designing the architecture of some modules.

</details>

## Bonus

Count total number of **lines of code** (Package: **7059** + Tests: **957**):

```bash
git ls-files | grep '\.py' | xargs wc -l
```

**Just for `Fun`**

```text
Q: What was the deaf student's favorite course?
A: Communication skills
```
