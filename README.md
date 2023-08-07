# Sign Language Translator ⠎⠇⠞

[![python](https://img.shields.io/pypi/pyversions/sign-language-translator)](https://pypi.org/project/sign-language-translator/)
[![PyPi](https://img.shields.io/pypi/v/sign-language-translator)](https://pypi.org/project/sign-language-translator/)
[![Downloads](https://static.pepy.tech/personalized-badge/sign-language-translator?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/sign-language-translator)

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
   2. [Python](#python)
      1. [basic translation](#basic-translation)
      2. [text language processor](#text-language-processor)
      3. [sign language processor](#sign-language-processor)
      4. [video/feature processing](#vision)
      5. [language models](#language-models)
4. [Directory Tree](#directory-tree)
5. [Research Paper](#research-paper)
6. [Upcoming/Roadmap](#upcomingroadmap)
7. [How to Contribute](#how-to-contribute)
8. [Credits and Gratitude](#credits-and-gratitude)
9. [Bonus](#bonus)
   1. Number of lines of code
   2. :)

## Overview

Sign language consists of gestures and expressions used mainly by the hearing-impaired to talk. This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.

The goal is to provide a user friendly API to novel Sign Language Translation solutions that can easily adapt to any regional sign language. Unlike most other projects, this python library can translate full sentences and not just the alphabet. This is the package that powers the [slt_ai website](https://github.com/mdsrqbl/slt_ai).

A bigger hurdle is the lack of datasets and frameworks that deep learning engineers and software developers can use to build useful products for the target community. This project aims to empower sign language translation by providing robust components, tools and models for both sign language to text and text to sign language conversion. It seeks to advance the development of sign language translators for various regions while providing a way towards sign language standardization.

### Solution

We've have built an *extensible rule-based* text-to-sign translation system that can be used to generate training data for *Deep Learning* models for both sign to text & text to sign translation.

To create a rule-based translation system for your regional language, you can inherit the TextLanguage and SignLanguage classes and pass them as arguments to the ConcatenativeSynthesis class. To write sample texts of supported words, you can use our language models. Then, you can use that system to fine-tune our AI models.

### Major Components and Goals


1. `Sign language to Text`
    <details><summary>...</summary>

    - Extract pose vectors (2D or 3D) from videos and map them to corresponding text representations of the performed signs.

    - Fine-tuned a neural network, such as a state-of-the-art speech-to-text model, with gradual unfreezing starting from the input layers to convert pose vectors to text.

    </details>

2. `Text to Sign Language`
    <details><summary>...</summary>

    - This is a relatively easier task if you parse input text and play appropriate video clips for each word.

    1. Motion Transfer
         - Concatenate pose vectors in the time dimension and transfer the movements onto any given image of a person. This ensures smooth transitions between video clips.
    2. Sign Feature Synthesis
         - Condition a pose sequence generation model on a pre-trained text encoder (e.g., fine-tune decoder of a multilingual T5) to output pose vectors instead of text tokens. This solves challenges related to unknown synonyms or hard-to-tokenize/process words or phrases.

    </details>

3. `Language Processing Utilities`
    <details><summary>...</summary>

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

4. `Data Collection and Creation`
    <details><summary>...</summary>

    - Capture variations in signs in a scalable and diversity accommodating way and enable advancing sign language standardization efforts.

      1. Clip extraction from long videos using timestamps
      2. Multithreaded Web scraping
      3. Language Models to generate sentences composed of supported word

    </details>

5. `Datasets`
    <details><summary>...</summary>

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

You can use the following functionalities of the SLT package via CLI as well. A command entered without any arguments will show the help. *The useable model-codes are listed in help*.
Note: Objects & models do not persist in memory across commands, so this is a quick but inefficient way to use this package. In production, create a server which uses the python interface.

#### Download

Download dataset files or models if you need them. The parameters are regular expressions.

```bash
slt download --overwrite true '.*\.json' '.*\.mp4'
```

```bash
slt download --progress-bar true 't2s_model_base.pt'
```

By default, auto-download is enabled. Default download directory is `/install-directory/sign_language_translator/sign-language-resources/`. (See slt.config.settings.Settings)

#### Translate

Translate text to sign language using a rule-based model

```bash
slt translate \
--model-code "concatenative" \
--text-lang urdu --sign-lang psl \
--sign-features 'mp-landmarks' \
"وہ سکول گیا تھا۔" \
'مجھے COVID نہیں ہے!'
```

#### Complete

Auto complete a sentence using our language models. This model can write sentences composed of supported words only:

```bash
$ slt complete --end-token ">" --model-code urdu-mixed-ngram "<"
('<', 'وہ', ' ', 'یہ', ' ', 'نہیں', ' ', 'چاہتا', ' ', 'تھا', '۔', '>')
```

These models predict next characters until a specified token appears. (e.g. generating names using a mixture of models):

```bash
$ slt complete \
    --model-code unigram-names --model-weight 1 \
    --model-code bigram-names -w 2 \
    -m trigram-names -w 3 \
    --selection-strategy merge --beam-width 2.5 --end-token "]" \
    "[s"
[shazala]
```

### Python

#### basic translation

```python
import sign_language_translator as slt

# download dataset or models (if you need them separate)
# (by default, dataset is auto-downloaded within the install directory)
# slt.set_resource_dir("path/to/sign-language-datasets") # optional. Helps when data is synced with cloud

# slt.utils.download("path", "url") # optional
# slt.utils.download_resource(".*.json") # optional

print("All available models:")
print(list(slt.ModelCodes)) # from slt.config.enums
# print(list(slt.TextLanguageCodes))
# print(list(slt.SignLanguageCodes))
```

**Text to Sign**:

```python
# Load text-to-sign model
# deep_t2s_model = slt.get_model("t2s-flan-T5-base-01.pt") # pytorch
# rule-based model (concatenates clips of each word)
t2s_model = slt.get_model(
    model_code = "concatenative-synthesis", # slt.ModelCodes.CONCATENATIVE_SYNTHESIS.value
    text_language = "urdu", # or object of any child of slt.languages.text.text_language.TextLanguage class
    sign_language = "PakistanSignLanguage", # or object of any child of slt.languages.sign.sign_language.SignLanguage class
    sign_feature_model = "mediapipe_pose_v2_hand_v1",
)

text = "HELLO دنیا!" # HELLO treated as an acronym
sign_language_sentence = t2s_model(text)

# moviepy_video_object = sign_language_sentence.video()
# moviepy_video_object.ipython_display()
# moviepy_video_object.write_videofile(f"sentences/{text}.mp4")
```

**Sign to text**
dummy code: (will be finalized in v0.8+)

```python
# load sign
video = slt.read_video("video.mp4")
# features = slt.extract_features(video, "mediapipe_pose_v2_hand_v1")

# Load sign-to-text model
deep_s2t_model = slt.get_model("gesture_mp_base-01") # pytorch
features = deep_s2t_model.extract_features(video)

# translate
text = deep_s2t_model(features)
print(text)
```

#### Text Language Processor

```python

from sign_language_translator.languages.text import Urdu
ur_nlp = Urdu()

text = "hello جاؤں COVID-19."

normalized_text = ur_nlp.preprocess(text)
# normalized_text = 'جاؤں COVID-19.'

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
tags = [Tags.WORD, Tags.SPACE] * 3 + [Tags.WORD, Tags.PUNCTUATION]
tokens, tags, _ = psl.restructure_sentence(tokens, tags) # ["he", "school", "go"]
signs  = psl.tokens_to_sign_dicts(tokens, tags)
# signs = [
#   {'signs': [['pk-hfad-1_وہ']], 'weights': [1.0]},
#   {'signs': [['pk-hfad-1_school']], 'weights': [1.0]},
#   {'signs': [['pk-hfad-1_گیا']], 'weights': [1.0]}
# ]
```

#### Vision

dummy code: (will be finalized in v0.7)

```python
import sign_language_translator as slt

# load video
video = slt.read_video("sign.mp4")

# extract features
features = video.extract_features(model="mediapipe_pose_v2_hand_v1")

# transform / augment data
features = slt.vision.rotate_landmarks(features, xyz=[60, 10, 90], degrees=True)

# plot
video = slt.vision.plot_landmarks(features)

# display
video.show()
print(features.numpy())
```

#### Language models

Simple statistical n-gram model:

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

Mash up multiple language models & complete generation through beam search:

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

Write sentences composed of only those words for which sign videos are available so that the rule-based text-to-sign model can generate training examples for a deep learning model:

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

## Directory Tree

<pre>
<big><b>sign-language-translator</b></big>
├── <a href="">MANIFEST.in</a>
├── <a href="">README.md</a>
├── <a href="">poetry.lock</a>
├── <a href="">pyproject.toml</a>
├── <a href="">requirements.txt</a>
├── <b>tests</b>
│   └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/tests">*</a>
│
└── <big><b>sign_language_translator</b></big>
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
    │   │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/layers.py">layers.py</a>
    │   │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/model.py">model.py</a>
    │   │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/train.py">train.py</a>
    │   │
    │   ├── sign_to_text
    │   └── text_to_sign
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py">concatenative_synthesis.py</a>
    │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/text_to_sign/t2s_model.py">t2s_model.py</a>
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
    │   ├── <del>data_loader.py</del>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/utils/download.py">download.py</a>
    │   ├── <del>landmarks_info.py</del>
    │   ├── <del>sign_data_attributes.py</del>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/utils/tree.py">tree.py</a>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/utils/utils.py">utils.py</a>
    │
    └── <b>vision</b>
        ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/vision/concatenate.py">concatenate.py</a>
        ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/vision/embed.py">embed.py</a>
        ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/vision/transforms.py">transforms.py</a>
        └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/vision/visualization.py">visualization.py</a>
</pre>

## Research Paper

Stay Tuned!

## Upcoming/Roadmap

```python
# :CLEAN_ARCHITECTURE_VISION: v0.7
    # class according to feature type (pose/video/mesh)
    # video loader
    # feature extraction
    # feature augmentation
    # feature model names enums
    # subtitles
    # 2D/3D avatars

# push notebooks
# expand reference clip data by scraping everything
# data info table
# https://sign-language-translator.readthedocs.io/en/latest/

# :DEEP_TRANSLATION: v0.8-v1.x
    # sign to text with fine-tuned whisper
    # pose vector generation with fine-tuned flan-T5
    # motion transfer
    # pose2video: stable diffusion or GAN?
    # speech to text
    # text to speech
    # LanguageModel: experiment by dropping space tokens
    # parallel text corpus

# RESEARCH PAPERs
    # datasets: clips, text, sentences, disambiguation
    # rule based translation: describe entire repo
    # deep sign-to-text: pipeline + experiments
    # deep text-to-sign: pipeline + experiments

# PRODUCT DEVELOPMENT
    # ML inference server
    # Django backend server
    # React Frontend
    # React Native mobile app
```

## How to Contribute

- Datasets:
  - Scrape and upload video datasets.
  - Label word mapping datasets.
  - Reach out to to Academies for Deaf and have them write down *sign language grammar*.
- New Code
  - Make sign language classes for various languages.
  - Make text language classes for various languages.
  - Train models with various hyper-parameters.
  - Remember to add `string short codes` of your classes and models to **`enums.py`** and update get_model(), get_.*_language().
  - Add docstrings, Example usage and test cases.
- Existing Code:
  - Optimize the code.
  - Add docstrings, Example usage and test cases.
  - Write documentation for `sign-language-translator.readthedocs.io`
- Contribute in [MLOps]()/[backend]()/[web]()/[mobile]() development.

## Credits and Gratitude

This project started in October 2021 as a BS Computer Science final year project with 3 students and 1 supervisor. After 9 months at university, it became a hobby project for Mudassar who has continued it till at least 2023-08-07.

Immense gratitude towards:

- [Mudassar Iqbal](https://github.com/mdsrqbl) for leading and coding the project so far.
- Rabbia Arshad for help in initial R&D and web development.
- Waqas Bin Abbas for assistance in initial video data collection process.
- Kamran Malik for setting the initial project scope, idea of motion transfer and connecting us with Hamza Foundation.
- [Hamza Foundation](https://www.youtube.com/@pslhamzafoundationacademyf7624/videos) (especially Ms Benish, Ms Rashda & Mr Zeeshan) for agreeing for collaboration and providing the reference clips, hearing-impaired performers for data creation, and creating the text2gloss dataset.
- [UrduHack](https://github.com/urduhack/urduhack) (espacially Ikram Ali) for their work on Urdu character normalization.

- [Telha Bilal](https://github.com/TelhaBilal) for help in designing the architecture of some modules.

## Bonus

Count total number of **lines of code** (Package: **9248** + Tests: **877**):

```bash
git ls-files | grep '\.py' | xargs wc -l
```

**Just for `Fun`**

```text
Q: What was the deaf student's favorite course?
A: Communication skills
```
