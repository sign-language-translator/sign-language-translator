# Sign Language Translator ⠎⠇⠞

[![python](https://img.shields.io/pypi/pyversions/sign-language-translator)](https://pypi.org/project/sign-language-translator/)
[![PyPi](https://img.shields.io/pypi/v/sign-language-translator)](https://pypi.org/project/sign-language-translator/)
[![Downloads](https://static.pepy.tech/personalized-badge/sign-language-translator?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/sign-language-translator)

1. [dummy code:](#dummy-code)
         1. [text language processor](#text-language-processor)
         2. [sign language processor](#sign-language-processor)
         3. [Vision](#vision)
         4. [language models](#language-models)
   1. [Directory Tree](#directory-tree)
   2. [Upcoming/Roadmap](#upcomingroadmap)
   3. [Research Paper](#research-paper)
   4. [Credits and Gratitude](#credits-and-gratitude)
   5. [Bonus](#bonus)

## Overview

Sign language consists of gestures and expressions used mainly by the hearing-impaired to talk. This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.

The goal is to provide a user friendly API to novel Sign Language Translation solutions that can easily adapt to any regional sign language. Unlike most other projects, this python library can translate full sentences and not just the alphabet. This is the package that powers the [slt_ai website](https://github.com/mdsrqbl/slt_ai).

A bigger hurdle is the lack of datasets and frameworks that deep learning engineers and software developers can use to build useful products for the target community. This project aims to empower sign language translation by providing robust components, tools and models for both sign language to text and text to sign language conversion. It seeks to advance the development of sign language translators for various regions while providing a way towards sign language standardization.

### Solution

We've have built an *extensible rule-based* text-to-sign translation system that can be used to generate training data for *Deep Learning* models for both sign to text & text to sign translation.

To create a rule-based translation system for your regional language, you can inherit the TextLanguage and SignLanguage classes and pass them as arguments to the ConcatenativeSynthesis class. To write sample texts of supported words, you can use our language models. Then, you can use that system to fine-tune our AI models.

### Major Components and Goals

1. `Sign language to Text`

    - Extract pose vectors (2D or 3D) from videos and map them to corresponding text representations of the performed signs.
    - Fine-tuned a neural network, such as a state-of-the-art speech-to-text model, with gradual unfreezing starting from the input layers to convert pose vectors to text.

2. `Text to Sign Language`
    - This is a relatively easier task if you parse input text and play appropriate video clips for each word.

    1. Motion Transfer
         - Concatenate pose vectors in the time dimension and transfer the movements onto any given image of a person. This ensures smooth transitions between video clips.
    2. Sign Feature Synthesis
         - Condition a pose sequence generation model on a pre-trained text encoder (e.g., fine-tuned decoder of a multilingual T5) to output pose vectors instead of text tokens. This solves challenges related to unknown synonyms or hard-to-tokenize/process words or phrases.

3. `Language Processing Utilities`
    1. Sign Processing
        - 3D world landmarks extraction with Mediapipe.
        - Pose Visualization with matplotlib and moviepy.
        - Pose transformations (data augmentation) with scipy.
    2. Text Processing
        - Normalize text input by substituting unknown characters/spellings with supported words.
        - Disambiguate ambiguous words to ensure accurate translation.
        - Tokenize text (word & sentence level).
        - Classify tokens and mark them with Tags.

4. `Data Collection and Creation`
    - Capture variations in signs in a scalable and diversity accommodating way and enable advancing sign language standardization efforts.

      1. Clip extraction from long videos using timestamps
      2. Multithreaded Web scraping
      3. Language Models to generate sentences composed of supported word

5. `Datasets`

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
    1. word/sentence mappings to videos
    2. spoken language sentences and phrases
    3. spoken language sentences & corresponding sign video label sequences
    4. preprocessing data such as word-to-numbers, misspellings, named-entities etc
    ```

    [See the *sign-language-datasets* repo and its *release files* for the actual data & details](https://github.com/sign-language-translator/sign-language-datasets)

## How to install the package

Production mode:

```bash
pip install sign-language-translator
```

Editable mode:

```bash
git clone https://github.com/sign-language-translator/sign-language-translator.git
cd sign-language-translator
pip install -e .
```

```bash
pip install -e git+https://github.com/sign-language-translator/sign-language-translator.git#egg=sign_language_translator
```

## Usage

see the [*test cases*](https://github.com/sign-language-translator/sign-language-translator/tree/main/tests) or [the *notebooks* repo](https://github.com/sign-language-translator/notebooks) for detailed use

### Command Line

<!-- #### configure

(Optional) Set the dataset directory.

```bash
slt configure --dataset-dir "/path/to/sign-language-datasets"
``` -->

#### Download

Download dataset files or models. The parameters are regular expressions.

```bash
slt download --overwrite true '.*\.json' '.*\.txt'
```

```bash
slt download --progress-bar true 't2s_model_base.pth'
```

(By default, the stuff is downloaded into `/install-directory/sign_language_translator/sign-language-resources/`)

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

Auto complete a sentence using our language models. You can write sentences of only supported words or just predict next characters until a specified token.

```bash
slt complete --model en-char-lm-1 --eos "." 'auto-complete is gre'
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
```

```python
# Load text-to-sign model
# deep_t2s_model = slt.get_model("t2s-flan-T5-base-01.pth") # pytorch
# rule-based model (concatenates clips of each word)
t2s_model = slt.get_model(
    model_code = "ConcatenativeSynthesis", # slt.enums.ModelCodes.CONCATENATIVE_SYNTHESIS.value
    text_language = "English", # or object of any child of slt.languages.text.text_language.TextLanguage class
    sign_language = "PakistanSignLanguage", # or object of any child of slt.languages.sign.sign_language.SignLanguage class
    sign_feature_model = "mediapipe_pose_v2_hand_v1",
)

text = "hello world!"
sign_language_sentence = t2s_model(text)

# moviepy_video_object = sign_language_sentence.video()
# moviepy_video_object.ipython_display()
# moviepy_video_object.write_videofile(f"sentences/{text}.mp4")
```

dummy code:

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

#### text language processor

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

#### sign language processor

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

dummy code:

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

#### language models

```python
from sign_language_translator.models.language_models import SimpleLanguageModel
names_data = [
    '[abeera]', '[areej]', '[farida]', '[hiba]', '[kinza]',
    '[mishal]', '[nimra]', '[rabbia]', '[tehmina]','[zoya]',
    '[amjad]', '[atif]', '[farhan]', '[huzaifa]', '[majeed]',
    '[nasir]', '[rizwan]', '[mudassar]', '[tayyab]', '[zain]',
]

# train an n-gram model (considers previous n tokens to predict)
model = SimpleLanguageModel(window_size=2, unknown_token="")
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

```python
from sign_language_translator.models.language_models import MixerLM, BeamSampling, SimpleLanguageModel

names_data = [...] # slt.languages.English().vocab.person_names # concat start/end symbols

SLMs = [
    SimpleLanguageModel(window_size=size, unknown_token="")
    for size in range(1,4)
]
[lm.fit(names_data) for lm in SLMs]

# randomly select a model and infer through it
mixed_model = slt.models.Mixer(
    models=SLMs,
    selection_probabilities=[1,2,4],
    unknown_token=""
)
print(mixed_model)
# Mixer LM: unk_tok=""[3]
# ├── Simple LM: unk_tok="", window=1, params=85 | prob=14.3%
# ├── Simple LM: unk_tok="", window=2, params=113 | prob=28.6%
# └── Simple LM: unk_tok="", window=3, params=96 | prob=57.1%

# use Beam Search to find high likelihood names
sampler = BeamSampling(mixed_model, beam_width=3) #, scoring_function = ...)
name = sampler.complete('[')
print(name)
# [rabbia]
```

Write sentences composed of only those words for which sign videos are available so that the rule-based model can generate training examples for a deep learning model:

```python
from sign_language_translator.models.language_models import TransformerLanguageModel

model = TransformerLanguageModel.load("ur-supported-word-lm.pth")
model.next_all(["میں(متکلم)", " ", "وزیراعظم", " ",])
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
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/utils.py">utils.py</a>
    │   ├── language_models
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/abstract_language_model.py">abstract_language_model.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/beam_sampling.py">beam_sampling.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/mixer.py">mixer.py</a>
    │   │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/simple_language_model.py">simple_language_model.py</a>
    │   │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/language_models/transformer_language_model.py">transformer_language_model.py</a>
    │   │
    │   ├── sign_to_text
    │   └── text_to_sign
    │       ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py">concatenative_synthesis.py</a>
    │       └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/text_to_sign/t2s_model.py">t2s_model.py</a>
    │
    ├── <i><b>sign-language-resources</b></i>
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
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/utils/tree.py">tree.py</a>
    │
    └── <b>vision</b>
        ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/vision/concatenate.py">concatenate.py</a>
        ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/vision/embed.py">embed.py</a>
        ├── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/vision/transforms.py">transforms.py</a>
        └── <a href="https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/vision/visualization.py">visualization.py</a>
</pre>

## Upcoming/Roadmap

```python
# :LANGUAGE_MODELS: v0.6
    # download model, CLI
    # token_to_id
    # GPT
    # basic sentence dataset

# :CLEAN_ARCHITECTURE_VISION: v0.7
    # class according to feature type (pose/video/mesh)
    # video loader
    # feature extraction
    # feature model names
    # subtitles
    # 2D/3D avatars

# push notebooks
# expand reference clip data by scraping everything
# data info table
# https://sign-language-translator.readthedocs.io/en/latest/

# :DEEP_TRANSLATION: v0.8-v1.x
    # sign to text with fine-tuned whisper
    # pose vector generation with flan-T5
    # motion transfer
    # pose2video: stable diffusion?

# RESEARCH PAPERs
    # datasets: clips, text, sentences, disambiguation
    # rule based translation: describe entire repo
    # deep sign-to-text: pipeline + experiments
    # deep text-to-sign: pipeline + experiments

# WEB DEVELOPMENT
    # ML inference server
    # Django backend server
    # React Frontend
    # React Native mobile app
```

## Research Paper

Stay Tuned!

## Credits and Gratitude

This project started in October 2021 as a BS Computer Science final year project with 3 students and 1 supervisor. After 9 months at university, it became a hobby project for Mudassar who has continued it till at least 2023-07-16.

Immense gratitude towards:

- [Mudassar Iqbal](https://github.com/mdsrqbl) for leading and coding the project so far.
- Rabbia Arshad for help in initial R&D and web development.
- Waqas Bin Abbas for assistance in initial video data collection process.
- Kamran Malik for setting the initial project scope, idea of motion transfer and connecting us with Hamza Foundation.
- [Hamza Foundation](https://www.youtube.com/@pslhamzafoundationacademyf7624/videos) (especially Ms Benish, Ms Rashda & Mr Zeeshan) for agreeing for collaboration and providing the reference clips, hearing-impaired performers for data creation, and creating the text2gloss dataset.
- [UrduHack](https://github.com/urduhack/urduhack) (espacially Ikram Ali) for their work on Urdu character normalization.

- [Telha Bilal](https://github.com/TelhaBilal) for help in designing the architecture of some modules.

## Bonus

Count total number of **lines of code** (Package: **7063** + Tests: **774**):

```bash
git ls-files | grep '\.py' | xargs wc -l
```

**Just for `Fun`**

```text
Q: What was the deaf student's favorite course?
A: Communication skills
```

**Publish package on `PyPI`**

1. Install Poetry [(official docs)](https://python-poetry.org/docs/#installation) and twine:

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    ```bash
    pip install twine
    ```

2. Initialize Poetry using the following command to create a new pyproject.toml file with the necessary project information

    existing project:

    ```bash
    poetry init
    ```

    new project:

    ```bash
    poetry new project-name
    ```

    add dependencies to pyproject.toml

    ```bash
    poetry add $(cat requirements.txt)
    ```

3. Build Distribution Files (might wanna add "dist/" to .gitignore)

    ```bash
    poetry build
    ```

4. Publish to PyPI

    ```bash
    twine upload dist/*
    ```

    Provide the credentials associated with your PyPI account.

5. Automate the Release Process (GitHub Actions)

    - Set up PyPI Configuration:
    - on Github repository page, go to "Settings" > "Secrets" and add PYPI_API_TOKEN as secret
    - Create and push .github/workflows/release.yml. Configure the workflow to:
    - trigger on the main branch update event.
    - check if version has updated using git diff
    - build and test your package.
    - publish the package to PyPI using Twine using secret credentials.

    - before actual publish, test your publishing process on test.pypi.org with `twine upload --repository testpypi dist/*`

    With this setup, whenever there is a new version update on the main branch, the CI tool will automatically build and release the package to PyPI.
