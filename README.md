# Sign Language Translator

## Overview
Sign language consists of gestures and expressions used mainly by the hearing-impaired to talk. This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.

The goal is to provide a user friendly api like HuggingFace to novel Sign Language Translation solutions that can easily adapt to any regional sign language. This is the package that powers the [slt_ai website](https://github.com/mdsrqbl/slt_ai).

A bigger hurdle is the lack of datasets and frameworks that deep learing engineers and software developers can use to build usefull products. That is what this project aims to deliver.

#### Major Components and Goals ####
1. Sign language to Text
    - In speech to text translation, features such as mel-spectrograms are extracted from audio and fed into neural networks which then output text tokens corresponding to what was said in the audio.
    - Similarly, pose vectors (2D or 3D) are extracted from video, and to be mapped to text corresponding to the performed signs, they are fed into a neural network which is a checkpoint of a SOTA speech-to-text model finetuned using gradual unfreezing starting from the layers near input towards the output layers.

2. Text to Sign Language
    - This is a relatively easier task as it can even be solved with HashTables. Just parse the input text and play approproate video clip for each word.

    1. Motion Transfer
        - This allows for seamless transitions between the clips. The idea is to concatenate pose vectors in the time dimention and transfer the movements onto any given image of any person.
    2. Pose Synthesis
        - This is similar to speech synthesis. It solves the challenge of unknown synonyms or hard to tokenize/process words/phrases.
        - It can also be finetuned to make avatars move in desired ways using only text.

3. Preprocessing Utilities
    1. Pose Extraction
        - Mediapipe 3D world coordinates and 2D image coordinates
        - Pose Visualization
    2. Text normalization
        - Since the supported vocabulary is handcrafted, unknown words (or spellings) must be substituted with the supported words.

## How to install the package
    $ git clone https://github.com/sign-language-translator/sign-language-translator.git
    $ cd sign-language-translator
    $ pip install -e .

- pip install -e git+https://github.com/sign-language-translator/sign-language-translator.git

## Directory Tree
    sign-language-translator
    ├── README.md
    ├── roadmap.md
    ├── pyproject.toml
    ├── requirements.txt
    └── sign_language_translator
        ├── data_collection
        │   ├── completeness.py
        │   ├── scraping.py
        │   └── synonyms.py
        │
        ├── models
        │   ├── language_models
        │   │   ├── abstract_language_model.py
        │   │   ├── beam_sampling.py
        │   │   ├── mixer.py
        │   │   ├── simple_language_model.py
        │   │   └── transformer_language_model.py
        │   │
        │   ├── sign_to_text
        │   └── text_to_sign
        │
        ├── text
        │   ├── metrics.py
        │   ├── preprocess.py
        │   ├── subtitles.py
        │   ├── tokens.py
        │   ├── utils.py
        │   └── vocab.py
        │
        ├── utils
        │   ├── dataLoader.py
        │   ├── landmarksInfo.py
        │   ├── signDataAttributes.py
        │   └── tree.py
        │
        └── vision
            ├── concatenate.py
            ├── embed.py
            ├── transforms.py
            └── visualization.py

## Datasets
[see dataset repo and its release files](https://github.com/sign-language-translator/sign-language-datasets)

## Research Paper

## Credits and Gratitude

## Bonus
