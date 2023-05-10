# Sign Language Translator ⠎⠇⠞

## Overview
Sign language consists of gestures and expressions used mainly by the hearing-impaired to talk. This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.

The goal is to provide a user friendly API to novel Sign Language Translation solutions that can easily adapt to any regional sign language. This is the package that powers the [slt_ai website](https://github.com/mdsrqbl/slt_ai).

A bigger hurdle is the lack of datasets and frameworks that deep learning engineers and software developers can use to build useful products for the target community. That is what this project aims to deliver.

#### Major Components and Goals ####
1. `Sign language to Text`
    - In speech to text translation, features such as mel-spectrograms are extracted from audio and fed into neural networks which then output text tokens corresponding to what was said in the audio.
    - Similarly, features such as pose vectors (2D or 3D) are extracted from video, and to be mapped to text corresponding to the performed signs, they are fed into a neural network which is a checkpoint of a SOTA speech-to-text model fine-tuned using gradual unfreezing starting from the layers near input towards the output layers.

2. `Text to Sign Language`
    - This is a relatively easier task as it can even be solved with HashTables. Just parse the input text and play appropriate video clip for each word.

    1. Motion Transfer
        - This allows for seamless transitions between the clips. The idea is to concatenate pose vectors in the time dimension and transfer the movements onto any given image of any person.
    2. Sign Feature Synthesis
        - This is similar to speech synthesis. It solves the challenge of unknown synonyms or hard to tokenize/process words/phrases.
        - It can also be fine-tuned to make avatars move in desired ways using only text.

3. `Preprocessing Utilities`
    1. Pose Extraction
        - Mediapipe 3D world coordinates and 2D image coordinates
        - Pose Visualization
    2. Text normalization
        - Since the supported vocabulary is handcrafted, unknown words (or spellings) must be substituted with the supported words.

4. `Data Collection and Creation`
    Sign languages are very diverse and every small region will require their own translator. So, the product needed first is the one that can help build sign language translators. This framework is designed with the capacity to handle all the variations and even lead to sign standardization.

   1. Clip extraction from long videos
   2. Multithreaded Web scraping
   3. Language Models to write sentences of supported words

## How to install the package
Production mode:
```
pip install sign_language_translator
```

Editable mode:
```
git clone https://github.com/sign-language-translator/sign-language-translator.git
cd sign-language-translator
pip install -e .
```
```
pip install -e git+https://github.com/sign-language-translator/sign-language-translator.git
```

## Datasets
The sign videos are categorized by:
1. country
2. source organization
3. session (remove this)
4. camera angle
5. person code

The text data includes:
1. word/sentence mappings to videos
2. spoken language word sequences
3. spoken language sentences & corresponding sign video label sequences
4. preprocessing data such as word-to-numbers, misspellings, named-entities etc

[See the *sign-language-datasets* repo and its *release files* for the actual data & details](https://github.com/sign-language-translator/sign-language-datasets)

## Usage
[see notebooks repo for detailed use](https://github.com/sign-language-translator/notebooks)

```python
import sign_language_translator as slt

# download dataset
# slt.download_data("path/to")
slt.set_dataset_dir("path/to/sign-language-datasets")

# download translation models
# model = slt.download_model("...")
t2s_model = slt.get_model(
    task= "text-to-sign",
    approach = "concatenative", # "generative"
    text_language = "Urdu",
    sign_language = "PakistanSignLanguage",
    sign_feature_model = "mediapipe_pose_v2_hand_v1",
)

sign_language_sentence = t2s_model.translate("hello world!")

sign_language_sentence.video().ipython_display()

```
## Directory Tree
    sign-language-translator
    ├── README.md
    ├── pyproject.toml
    ├── requirements.txt
    ├── roadmap.md
    ├── sign_language_translator
    │   ├── config
    │   │   ├── enums.py
    │   │   ├── helpers.py
    │   │   └── settings.py
    │   │
    │   ├── data_collection
    │   │   ├── completeness.py
    │   │   ├── scraping.py
    │   │   └── synonyms.py
    │   │
    │   ├── languages
    │   │   ├── sign_collection.py
    │   │   ├── vocab.py
    │   │   ├── sign
    │   │   │   ├── pakistan_sign_language.py
    │   │   │   └── sign_language.py
    │   │   │
    │   │   └── text
    │   │       ├── text_language.py
    │   │       ├── english.py
    │   │       └── urdu.py
    │   │
    │   ├── models
    │   │   ├── language_models
    │   │   │   ├── abstract_language_model.py
    │   │   │   ├── beam_sampling.py
    │   │   │   ├── mixer.py
    │   │   │   ├── simple_language_model.py
    │   │   │   └── transformer_language_model.py
    │   │   │
    │   │   ├── sign_to_text
    │   │   └── text_to_sign
    │   │       ├── concatenative_synthesis.py
    │   │       └── t2s_model.py
    │   │
    │   ├── text
    │   │   ├── metrics.py
    │   │   ├── preprocess.py
    │   │   ├── subtitles.py
    │   │   ├── tokens.py
    │   │   └── utils.py
    │   │
    │   ├── utils
    │   │   ├── data_loader.py
    │   │   ├── landmarks_info.py
    │   │   ├── sign_data_attributes.py
    │   │   └── tree.py
    │   │
    │   └── vision
    │       ├── concatenate.py
    │       ├── embed.py
    │       ├── transforms.py
    │       └── visualization.py
    │
    │
    └── tests

## Research Paper
Stay Tuned!

## Credits and Gratitude
This project started in October 2021 as a BS Computer Science final year project with 3 students and 1 supervisor at PUCIT. After 9 months at university, it became a hobby project for Mudassar who has continued it till at least 2023-04-20.

Immense gratitude towards:
- [Mudassar Iqbal](https://github.com/mdsrqbl) for leading and coding the project so far.
- Rabbia Arshad for help in web development and initial R&D.
- Waqas Bin Abbas for assistance in video data collection process.
- Dr Kamran Malik for teaching us AI, setting the initial project scope, idea of motion transfer and connecting us with Hamza Foundation.
- Hamza Foundation (especially Ms Benish, Ms Rashda & Mr Zeeshan) for agreeing for collaboration and providing the reference clips, hearing-impaired performers for data creation, and creating the text2gloss dataset.

## Bonus
