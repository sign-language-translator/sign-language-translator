Welcome to Sign Language Translator's documentation!
====================================================

`Sign Language Translator <https://github.com/sign-language-translator/sign-language-translator>`_ is a `python package <https://pypi.org/project/sign-language-translator>`_ built to allow developers to create and integrate custom & state of the art sign language translation solutions into their applications.
It brings you the power of building a translator for any region's sign language.

All you have to do is to override the ``sign_language_translator.languages.SignLanguage`` class and pass its object to the rule-based text-to-sign translator (``sign_language_translator.models.ConcatenativeSynthesis``).

This package also enables you to easily train & finetune deep learning models on custom sign language datasets which can be hand-crafted, scrapped, or generated via the rule-based translator. See the :doc:`datasets <./datasets>` page for more details about training.


.. toctree::
   :maxdepth: 4
   :hidden:

   Installation & Usage <self>
   sign_language_translator
   datasets

.. contents::

Installation
============

Install the package from pypi.org

.. code-block:: bash

   pip install sign-language-translator

or editable mode from github.

.. code-block:: bash

   git clone https://github.com/sign-language-translator/sign-language-translator.git
   cd sign-language-translator
   pip install -e .

.. note::
   This package is currently available for Python 3.9, 3.10 & 3.11

Usage
=====

The package is available as a python module, via command line interface (CLI), and as a `gradio` based GUI as well.

Python
------

Translation
^^^^^^^^^^^

.. code-block:: python
   :linenos:
   :caption: Basics
   :emphasize-lines: 1

   import sign_language_translator as slt

   # download dataset or models (if you need them for personal use)
   # (by default, resources are auto-downloaded within the install directory)
   # slt.Assets.set_root_dir("path/to/folder")  # Helps preventing duplication across environments or using cloud synced data
   # slt.Assets.download(".*.json")  # downloads into resource_dir
   # print(slt.Settings.FILE_TO_URL.keys())  # All downloadable resources

   print("All available models:")
   print(list(slt.ModelCodes))  # slt.ModelCodeGroups
   # print(list(slt.TextLanguageCodes))
   # print(list(slt.SignLanguageCodes))
   # print(list(slt.SignFormatCodes))

.. code-block:: python
   :linenos:
   :caption: Text to Sign Language Translation
   :emphasize-lines: 4,7,9,11

   import sign_language_translator as slt

   # The core model of the project (rule-based text-to-sign translator)
   # which enables us to generate synthetic training datasets
   model = slt.models.ConcatenativeSynthesis(
      text_language="urdu", sign_language="pk-sl", sign_format="video" )

   text = "یہ اچھا ہے۔" # "this-good-is"
   sign = model.translate(text) # tokenize, map, download & concatenate
   sign.show()

   model.sign_format = slt.SignFormatCodes.LANDMARKS
   model.sign_embedding_model = "mediapipe-world"

   model.text_language = slt.languages.text.English()
   sign_2 = model.translate("This is an apple.")
   sign_2.save("this-is-an-apple.csv", overwrite=True)

   model.text_language = slt.TextLanguageCodes.HINDI
   sign_3 = model.translate("कैसे हैं आप?") # "how-are-you"
   sign_3.save_animation("how-are-you.gif", overwrite=True)

.. code-block:: python
   :linenos: 2,6,7,11,12
   :caption: Sign Language to Text Translation

   import sign_language_translator as slt

   # sign = slt.Video("path/to/video.mp4")
   sign = slt.Video.load_asset("pk-hfad-1_aap-ka-nam-kya(what)-hy")  # your name what is? (auto-downloaded)
   sign.show_frames_grid()

   # Extract Pose Vector for feature reduction
   embedding_model = slt.models.MediaPipeLandmarksModel()      # pip install "sign_language_translator[mediapipe]"  # (or [all])
   embedding = embedding_model.embed(sign.iter_frames())

   slt.Landmarks(embedding.reshape((-1, 75, 5)),
               connections="mediapipe-world").show()

   # # Load sign-to-text model (pytorch) (COMING SOON!)
   # translation_model = slt.get_model(slt.ModelCodes.Gesture)
   # text = translation_model.translate(embedding)
   # print(text)

Building Custom Translators
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Problem understanding is the most crucial part of solving it.
Translation is a sequence-to-sequence (seq2seq) problem, not classification or segmentation or any other.
For translation, we need parallel corpora of text language sentences and corresponding sign language videos.
Since there is no universal sign language so signs vary even within the same city. Hence, no significant datasets would be available for your regional sign language.

Rule Based Translator
*********************

.. note::
   This approach can only work for text-to-sign language translation for a limited unambiguous vocabulary.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/sign-language-translator/notebooks/blob/main/translation/concatenative_synthesis.ipynb
   :alt: Open In Colab

We start by building our sign language dataset for sign language recognition & sign language production (both require 1:1 mapping of one language to the other).
First gather sign language video dictionaries for various regions of the world to eventually train a multilingual model. These can be scraped off the internet or recorded manually against reference clips or images.
Label the videos with all the text language words that have the same meaning as the sign. If there are multiple signs in the video, make sure to write the gloss (words in text have 1:1 correspondence with the signs in the video) and the translation (text follows the grammar of the spoken language).

`Here <https://github.com/sign-language-translator/sign-language-datasets/blob/main/parallel_texts/pk-dictionary-mapping.json>`_ is the format used in the library to store mappings. But you only need to add a dict to your language processing classes.

.. code-block:: python
   :caption: Word mappings to signs

   mappings = {
      "hello": [  # token to list of video-file-name sequences
         ["pk-org-1_hello"]  # this sequence contains only one clip
      ],
      "world": [
         ["xx-yyy-#_part-1", "xx-yyy-#_part-2"]  # two clips played consecutively make the right sign
         ["pk-org-1_world"],  # This is another possible sign less commonly used for the same word
      ],
   }

Place the actual video files in the ``assets/videos`` or ``assets/datasets/xx-yyy-#.videos-mp4.zip`` (or preprocessed files in similar directory structure e.g. ``assets/landmarks``). Otherwise update the asset manager with the URLs as follows:

.. code-block:: python
   :caption: Fetching signs for tokens

      import sign-language-translator as slt

      # help(slt.Assets)
      # print(slt.Assets.ROOT_DIR)
      # slt.Assets.set_root_dir("path/to/centralized/folder")

      slt.Assets.FILE_TO_URL.update({
         "videos/xx-yyy-#_word.mp4": "https://...",
         "datasets/xx-yyy-#.videos-mp4.zip": "https://...",
         "datasets/xx-yyy-#.landmarks-[estimation-model-name]-csv.zip": "https://...",
      })

      # paths = slt.Assets.download("videos/xx-yyy-#_word.mp4")
      # paths = slt.Assets.extract("...mp4", "datasets/...zip")

Now use our rule-based translator (``slt.models.ConcatenativeSynthesis``) as follows:

.. code-block:: python
   :linenos:
   :caption: Custom rule-based Translator (text to sign)

   import sign_language_translator as slt

   class MySignLanguage(slt.languages.SignLanguage):
      # load and save mappings in __init__
      # override the abstract functions
      def restructure_sentence(...):
         # according to the sign language grammar
         ...

      def tokens_to_sign_dicts(...):
         # map words to all possible videos
         ...

      # for reference, see the implementation inside slt.languages.sign.pakistan_sign_language

   # optionally implement a text language processor as well
   # class MyChinese(slt.languages.TextLanguage):
       # ...

   model = slt.models.ConcatenativeSynthesis(
      text_language = slt.languages.text.English(),
      sign_format = "video",
      sign_language = MySignLanguage(),
   )

   text = """Some text that will be tokenized, tagged, rearranged, mapped to video files
             (which will be downloaded, concatenated and returned)."""
   video = model.translate(text)
   video.show()
   # video.save(f"{text}.mp4")

Deep Learning based
*******************

.. note::
   This approach can work for both sign-to-text and text-to-sign-language translation

You can use 3 types of Parallel corpus as training data (`more details <https://github.com/sign-language-translator/sign-language-datasets#problem-overview>`_):
#. Sentences (bunch of signs performed consecutively in a single video to form a meaningful message.)
#. Synthetic sentences (made by the rule-based translator from the text of spoken languages & dictionary videos.)
#. Replications (recordings of people performing the signs in dictionary videos and sentences.)

`Here <https://github.com/sign-language-translator/sign-language-datasets/blob/main/parallel_texts/pk-sentence-mapping.json>`_ is a format that you can use for data labeling.  You can use the following `JSON schema <https://github.com/sign-language-translator/sign-language-datasets/blob/main/schemas/mapping-schema.json>`_ to validate your labeled data.

You can use `language models <Complete>`_ to write sentences for synthetic data. The language models can be masked to use only specified words in the output so that the rule-based translator can translate them.

Get the best out of your model by training it for multiple languages and multiple tasks. For example, provide task as start-of-sequence-token and target-text-language as the second token to the decoder of the seq2seq model.

.. code-block:: python
   :linenos:
   :caption: Fine-tuning Deep learning Translators

   import sign_language_translator as slt

   # pretrained_model = slt.get_model(slt.ModelCodes.Gesture)  # sign landmarks to text (COMING SOON!)
   # original training code: 

   # pytorch training loop to finetune our model on your dataset
   for epoch in range(10):
      for sign, text in train_dataset:
         ...

.. The training strategy I used was to translate the text labels to many languages using Google Translate and pretrain the encoder of a speech-to-text model on that. Then train on a mixture of the 3 types of parallel corpus mentioned above.

See more in the package `readme <https://github.com/sign-language-translator/sign-language-translator#how-to-build-a-translator-for-sign-language>`_.

Text Language Processing
^^^^^^^^^^^^^^^^^^^^^^^^

Process text strings using language specific classes:

.. code-block:: python
   :linenos:
   :caption: English Text Processor

   from sign_language_translator.languages.text import English
   en_nlp = English()

   text = "Hello , I lived in U.S.A.! What about you? 😄"
   text = en_nlp.preprocess(text)              # 'Hello, I lived in U.S.A.! What about you?'

   sentences = en_nlp.sentence_tokenize(text)  # ['Hello, I lived in U.S.A.!', 'What about you?']
   tokens = en_nlp.tokenize(sentences[1])      # ['What', 'about', 'you', '?']
   tagged = en_nlp.tag(tokens)                 # [('What', Tags.SUPPORTED_WORD), ('about', Tags.WORD), ('you', Tags.AMBIGUOUS), ('?', Tags.PUNCTUATION)]
   senses = en_nlp.get_word_senses(["close", "orange"])  # [['close(shut)', 'close(near)'], ['orange(fruit)', 'orange(color)']]


.. code-block:: python
   :linenos:
   :caption: Urdu Text Processor

   from sign_language_translator.languages.text import Urdu
   ur_nlp = Urdu()

   english_script = ur_nlp.romanize("کاش یہ اتنا آسان ہوتا۔")
   # english_script: "kash yh atna aasan hota."

   text = "hello جاؤں COVID-19."

   normalized_text = ur_nlp.preprocess(text)
   # normalized_text: 'جاؤں COVID-19.' # replace/remove unicode characters

   tokens = ur_nlp.tokenize(normalized_text)
   # tokens: ['جاؤں', ' ', 'COVID', '-', '19', '.']

   # tagged = ur_nlp.tag(tokens)
   # tagged: [('جاؤں', Tags.SUPPORTED_WORD), (' ', Tags.SPACE), ...]

   tags = ur_nlp.get_tags(tokens)
   # tags: [Tags.SUPPORTED_WORD, Tags.SPACE, Tags.ACRONYM, ...]

   word_senses = ur_nlp.get_word_senses("میں")
   # word_senses: [["میں(i)", "میں(in)"]]  # multiple meanings


.. code-block:: python
   :linenos:
   :caption: Hindi Text Processor

   from sign_language_translator.languages.text import Hindi
   hi_nlp = Hindi()

   text = "नमस्ते आप कैसे हैं? "
   text = hi_nlp.preprocess(text)  # 'नमस्ते आप कैसे हैं?'

   hi_nlp.romanize(text)           # 'nmste āp kaise hain?'
   tokens = hi_nlp.tokenize(text)  # ['नमस्ते', 'आप', 'कैसे', 'हैं', '?']
   tags = hi_nlp.get_tags(tokens)  # [Tags.WORD, Tags.SUPPORTED_WORD, Tags.SUPPORTED_WORD, Tags.SUPPORTED_WORD, Tags.PUNCTUATION]
   senses = hi_nlp.get_word_senses(["सोने"])  # [['सोने(स्वर्ण)']]

Sign Language Processing
^^^^^^^^^^^^^^^^^^^^^^^^

This processes a representation of sign language which mainly consists of the file names of videos.
There are two main parts:
   1. word to video mapping
   2. word rearrangement according to grammar.

For video processing, see :ref:`vision` section.

.. code-block:: python
   :linenos:
   :caption: Pakistan Sign Language Processor

   from sign_language_translator.languages.sign import PakistanSignLanguage

   psl = PakistanSignLanguage()

   tokens = ["he", " ", "went", " ", "to", " ", "school", "."]
   tags = 3 * [Tags.WORD, Tags.SPACE] + [Tags.WORD, Tags.PUNCTUATION]
   tokens, tags, _ = psl.restructure_sentence(tokens, tags, None) # ["he", "school", "go"]
   signs  = psl.tokens_to_sign_dicts(tokens, tags)
   # signs = [
   #   {'signs': [['pk-hfad-1_that']], 'weights': [1.0]},
   #   {'signs': [['pk-hfad-1_school']], 'weights': [1.0]},
   #   {'signs': [['pk-hfad-1_gia']], 'weights': [1.0]}
   # ]

Vision
^^^^^^

This covers the functionalities of representing sign language as objects (sequence of frames e.g. video or sequence of vectors e.g. pose landmarks).
Those objects have built-in functions for data augmentation and visualization etc.

.. code-block:: python
   :linenos:
   :caption: Sign Processing

   import sign_language_translator as slt

   # load video
   video = slt.Video("sign.mp4")
   print(video.duration, video.shape)

   # extract features
   model = slt.models.MediaPipeLandmarksModel()  # default args
   embedding = model.embed(video.iter_frames(), landmark_type="world") # torch.Tensor
   print(embedding.shape)  # (n_frames, n_landmarks * 5)

   # embed dataset
   # slt.models.utils.VideoEmbeddingPipeline(model).process_videos_parallel(
   #     ["dataset/*.mp4"], n_processes=12, save_format="csv", ...
   # )

   # Show the landmarks
   landmarks = slt.Landmarks(embedding.reshape((-1, 75, 5)),
                             connections="mediapipe-world")
   landmarks.show()
   landmarks.show_frames_grid(3, 5)

   # transform / augment data
   # tform = slt.vision.landmarks.transformations.RotateLandmarks(60, 10, 90, degrees=True)
   # landmarks = tform(landmarks)
   # landmarks.show()
   # landmarks.transform(tform, inplace=True)
   # landmarks.show()

Language models
^^^^^^^^^^^^^^^

In order to generate synthetic training data via the rule-based model, we need a lot of sentences consisting of supported words only. 
(Supported word: a piece of text for which sign language video is available.)
These language models were built to write such sentences. See the :doc:`datasets <./datasets>` page for training process.

.. code-block:: python
   :linenos:
   :caption: Simple Character-level N-Gram Language Model (uses statistics based hashmaps)
   :emphasize-lines: 11,12,18

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
      # select next token randomly from learnt probability distribution
      nxt, prob = model.next(name)

      name += nxt
      if nxt in [']' , model.unknown_token]:
         break

   print(name)
   # '[rabeej]'

   # see ngram model's implementation
   # print(model.__dict__)

Mash up multiple language models & complete generation through beam search:

.. code-block:: python
   :linenos:
   :caption: Model Mixer & Beam Search
   :emphasize-lines: 19,20,21,35,36

   from sign_language_translator.models.language_models import MixerLM, BeamSampling, NgramLanguageModel

   names_data = names_data = [
      '[abeera]', '[areej]',  '[farida]',  '[hiba]',    '[kinza]',
      '[mishal]', '[nimra]',  '[rabbia]',  '[tehmina]', '[zoya]',
      '[amjad]',  '[atif]',   '[farhan]',  '[huzaifa]', '[mudassar]',
      '[nasir]',  '[rizwan]', '[shahzad]', '[tayyab]',  '[zain]',
   ] # or slt.languages.English().vocab.person_names # gotta concat start/end symbols tho

   # train models
   LMs = [
      NgramLanguageModel(window_size=size, unknown_token="")
      for size in range(1,4)
   ]
   for lm in LMs:
      lm.fit(names_data)

   # combine the models into one object
   mixed_model = MixerLM(
      models=LMs,
      selection_probabilities=[1,2,4],
      unknown_token="",
      model_selection_strategy = "choose", # "merge"
   )
   print(mixed_model)
   # Mixer LM: unk_tok=""[3]
   # ├── Ngram LM: unk_tok="", window=1, params=85 | prob=14.3%
   # ├── Ngram LM: unk_tok="", window=2, params=113 | prob=28.6%
   # └── Ngram LM: unk_tok="", window=3, params=96 | prob=57.1%

   # randomly select a LM and infer through it
   print(mixed_model.next("[m"))

   # use Beam Search to find high likelihood names
   sampler = BeamSampling(mixed_model, beam_width=3) #, scoring_function = ...)
   name = sampler.complete('[')
   print(name)
   # [rabbia]

Use a pre-trained language model:

.. code-block:: python
   :linenos:
   :caption: Transformer Language Model

   from sign_language_translator.models.language_models import TransformerLanguageModel

   # model = slt.get_model("ur-supported-gpt")
   model = TransformerLanguageModel.load("models/tlm_14.0M.pt")
   # sampler = BeamSampling(model, ...)
   # sampler.complete(["<"])

   # see probabilities of all tokens
   model.next_all(["میں", " ", "وزیراعظم", " ",])
   # (["سے", "عمران", ...], [0.1415926535, 0.7182818284, ...])

Text Embedding
^^^^^^^^^^^^^^

Embed text words & phrases into pre-trained vectors using a selected embedding model.
It is useful for finding synonyms in other languages and for building controllable language models.

.. code-block:: python
   :linenos:
   :emphasize-lines: 2, 5, 11, 13
   :caption: Pretrained Text Embedding

   import torch
   from sign_language_translator.models import VectorLookupModel, get_model

   # Custom Model
   model = VectorLookupModel(["hello", "world"], torch.Tensor([[0, 1], [2, 3]]))
   vector = model.embed("hello")  # torch.Tensor([0, 1])
   vector = model.embed("hello world")  # torch.Tensor([1., 2.])  # average of two tokens
   model.save("vectors.pt")

   # Pretrained Model
   model = get_model("lookup-ur-fasttext-cc")
   print(model.description)
   vector = model.embed("تعلیم", align=True, post_normalize=True)
   print(vector.shape)  # (300,)

   # find similar words but in a different language
   en_model = get_model("lookup-en-fasttext-cc")
   en_vectors = en_model.vectors / en_model.vectors.norm(dim=-1, keepdim=True)
   similarities = en_vectors @ vector
   similar_words = [
      (en_model.index_to_token[i], similarities[i].item())
      for i in similarities.argsort(descending=True)[:5]
   ]
   print(similar_words)  # [(education, 0.5469), ...]


Command line (CLI)
------------------

You can use the following functionalities of the SLT package via CLI as well. A command entered without any arguments will show the help. *The useable model-codes are listed in help*.

Note: Objects & models do not persist in memory across commands, so this is a quick but inefficient way to use this package. In production, create a server which uses the python interface.

Assets
^^^^^^

Download or view the resources needed by the sign language translator package using the following commands.

Path
****

Print the root path where the assets are being stored.

.. code-block:: bash

   slt assets path

Download
********

Download dataset files or models if you need them. The parameters are regular expressions.

.. code-block:: bash

   slt assets download --overwrite true '.*\.json' '.*\.mp4'

.. code-block:: bash

   slt assets download --progress-bar true '.*/tlm_14.0M.pt'

By default, auto-download is enabled. Default download directory is `/install-directory/sign_language_translator/sign-language-resources/`. (See slt.config.settings.Settings)

Tree
****

View the directory structure of the present state of the assets folder.

.. code-block:: bash

   $ slt assets tree
   assets
   ├── checksum.json
   ├── pk-dictionary-mapping.json
   ├── text_preprocessing.json
   ├── datasets
   │   ├── pk-hfad-1.landmarks-mediapipe-world-csv.zip
   │   └── pk-hfad-1.videos-mp4.zip
   │
   ├── landmarks
   │   ├── pk-hfad-1_airplane.landmarks-mediapipe-world.csv
   │   └── pk-hfad-1_icecream.landmarks-mediapipe-image.csv
   │
   ├── models
   │   ├── ur-supported-token-unambiguous-mixed-ngram-w1-w6-lm.pkl
   │   └── mediapipe
   │       └── pose_landmarker_heavy.task
   │
   └── videos
      ├── pk-hfad-1_1.mp4
      ├── pk-hfad-1_cow.mp4
      └── pk-hfad-1_me.mp4

.. code-block:: bash
   :caption: Directories only

   slt assets tree --files false

.. code-block:: bash
   :caption: omit files matching the regex

   slt assets tree --ignore ".*mp4" -i ".*csv"

.. code-block:: bash
   :caption: Tree of a custom directory

   slt assets tree -d "path/to/your/assets"

Translate
^^^^^^^^^

Translate text to sign language using a rule-based model:

.. code-block:: bash

   slt translate --model-code "rule-based" \
   --text-lang urdu --sign-lang psl --sign-format 'video' \
   "وہ سکول گیا تھا۔"

Complete
^^^^^^^^

Auto complete a sentence using our language models. This model can write sentences composed of supported words only:

.. code-block:: shell-session

   $ slt complete --end-token ">" --model-code urdu-mixed-ngram "<"
   ('<', 'وہ', ' ', 'یہ', ' ', 'نہیں', ' ', 'چاہتا', ' ', 'تھا', '۔', '>')

These models predict next characters until a specified token appears. (e.g. generating names using a mixture of models):

.. code-block:: shell-session

   $ slt complete \
      --model-code unigram-names --model-weight 1 \
      --model-code bigram-names -w 2 \
      -m trigram-names -w 3 \
      --selection-strategy merge --beam-width 2.5 --end-token "]" \
      "[s"
   [shazala]

Embed Videos
^^^^^^^^^^^^

Embed **videos** into a sequence of vectors using a selected embedding model:

.. code-block:: bash

   slt embed videos/*.mp4 --model-code mediapipe-pose-2-hand-1 --embedding-type world \
      --processes 4 --save-format csv --output-dir ./embeddings

Embed texts
^^^^^^^^^^^^

Embed **texts** into pyTorch state dict pickled file using a selected embedding model.
The target file is a `.pt` containing `{"tokens": ..., "vectors": ...}`.

.. code-block:: bash

   slt embed "hello" "world" "more-tokens.txt" --model-code lookup-en-fasttext-cc


GUI
---

This functionality is not available yet! (but will probably be a `gradio` based light frontend)

.. code-block:: python
   :linenos:

   import sign_language_translator as slt

   slt.launch_gui()

.. code-block:: console

   slt gui

.. < # TODO: insert preview image >
