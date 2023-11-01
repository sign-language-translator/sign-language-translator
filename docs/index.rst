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
   # slt.set_resource_dir("path/to/folder")  # Helps preventing duplication across environments or using cloud synced data
   # slt.utils.download_resource(".*.json")  # downloads into resource_dir
   # print(slt.Settings.FILE_TO_URL.keys())  # All downloadable resources

   print("All available models:")
   print(list(slt.ModelCodes))  # slt.ModelCodeGroups
   # print(list(slt.TextLanguageCodes))
   # print(list(slt.SignLanguageCodes))
   # print(list(slt.SignFormatCodes))

.. code-block:: python
   :linenos:
   :caption: Text to Sign Language Translation
   :emphasize-lines: 4,5,11

   import sign_language_translator as slt

   # print(slt.ModelCodes)
   # model = slt.get_model("transformer-text-to-sign")
   model = slt.models.ConcatenativeSynthesis(
      text_language = "urdu", # or object of any child of slt.languages.text.text_language.TextLanguage class
      sign_language = "pakistan-sign-language", # or object of any child of slt.languages.sign.sign_language.SignLanguage class
      sign_format = "video" # or object of any child of slt.vision.sign.Sign class
   )

   sign_language_sentence = model.translate("یہ اچھا ہے۔")
   # sign_language_sentence.show()
   # sign_language_sentence.save("output.mp4")

.. code-block:: python
   :linenos:
   :caption: Sign Language to Text Translation (dummy code until v0.8)

   import sign_language_translator as slt

   # load sign
   video = slt.Video("video.mp4")
   # features = slt.LandmarksSign("landmarks.csv", landmark_type="world")

   # embed
   embedding_model = slt.get_model("mediapipe_pose_v2_hand_v1")
   features = embedding_model.embed(video.iter_frames())

   # Load sign-to-text model
   deep_s2t_model = slt.get_model(slt.ModelCodes.TRANSFORMER_MP_S2T) # pytorch

   # translate via single call to pipeline
   # text = deep_s2t_model.translate(video)

   # translate via individual steps
   encoding = deep_s2t_model.encoder(features)
   token_ids = [0] # start_token
   for _ in range(5):
      logits = deep_s2t_model.decoder(encoding, token_ids=token_ids)
      token_ids.append(logits.argmax(dim=-1).item())

   tokens = deep_s2t_model.decode(token_ids)
   text = "".join(tokens) # deep_s2t_model.detokenize(tokens)

   print(features.shape)
   print(logits.shape)
   print(text)

Building Custom Translators
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Problem understanding is the most crucial part of solving it.
Translation is a sequence-to-sequence (seq2seq) problem, not classification or any other.
For translation, we need parallel corpora of text language sentences and corresponding sign language videos.
But, no significant datasets are available for any sign language. Furthermore, there is no universal sign language and signs may vary within the same city.

We start by building our sign language dataset for sign language recognition & sign language production (both require 1:1 mapping of one language to the other).
First gather sign language video dictionaries for various regions of the world to eventually train a multilingual model. These can be scraped off the internet or recorded manually against reference clips or images.
Label the videos with all the text language words that have the same meaning as the sign. If there are multiple signs in the video, make sure that the words in text phrase have 1:1 correspondence with the signs in the video.

.. code-block:: json
   :caption: sign-language-resources/sign_recordings/collection_to_label_to_language_to_words.json
      {
      "country-organization-number": {
         "1": {
               "english": [
                  "1",
                  "one"
               ],
               "urdu": [
                  "۱",
                  "ایک"
               ]
            },
         ...
         },
         ...
      }

.. code-block:: yaml
   :caption: sign_language_translator/config/urls.yaml
      datasets:
        - name: reference-clips
          files:
            - name: videos/country-organization-number_1.mp4
              url: https://github.com/sign-language-translator/sign-language-datasets/releases/download/v0.0.2/country-organization-number_1.mp4
      ...

The dataset can now be synthesized by our rule-based translator (``slt.models.ConcatenativeSynthesis``) as follows:

.. code-block:: python
   :linenos:
   :caption: Custom rule-based Translator (text to sign)

   import sign_language_translator as slt

   class MySignLanguage(slt.languages.SignLanguage):
      # override the abstract functions
      def restructure_sentence(...):
         # according to the sign language grammar
         ...

      def tokens_to_sign_dicts(...):
         # map words to all possible videos
         ...

      # for reference, see the implementation inside slt.languages.sign.pakistan_sign_language

   # optionally implement a text language processor as well
   # class Chinese(slt.languages.TextLanguage):
       # ...

   model = slt.models.ConcatenativeSynthesis(
      text_language = slt.languages.text.English(),
      sign_language = MySignLanguage(),
      sign_format = "video",
   )

   text = """Some text that will be tokenized, tagged, rearranged, mapped to video files
             (which will be concatenated and returned)."""
   video = model.translate(text, restructure_sentence=False)

   embedding_model = slt.models.MediaPipeLandmarksModel()
   features = embedding_model.embed(video.iter_frames())

   features.save(f"parallel-dataset/{text}.mp4")

You can use `language models <Complete>`_ (trained on sentence chunks consisting of only the words present in your mapping dataset) to write sentences.

Get the best out of your model by training it for multiple languages and multiple tasks. For example, provide task as start-of-sequence-token and target-text-language as the second token to the decoder of the seq2seq model.

.. code-block:: yml
   :linenos:
   :caption: parallel-corpus

   - pakistan-sign-language-dataset
       - inputs: video_features_1.csv
       - transcription: # generated by rule-based translator
           - english: "this is a sentence."
       - translation: # generated from the transcription by Google translate API
           - urdu: "..."
           - hindi: "..."
           - chinese: "..."

.. code-block:: python
   :linenos:
   :caption: inferring through sign to text translation model

   ...

   encoding = model.encoder(sign_features)
   ids = model.tokens_to_ids(["<|transcribe|>", "<english>"])
   # ids = model.tokens_to_ids(["<|translate|>", "<urdu>"])

   for _ in range(max_length):
       logits = model.decoder(encoding, ids)
       ids.append(logits.argmax(-1).item())

   tokens = model.ids_to_tokens(ids[2:])
   sentence = model.detokenize(tokens)
   print(sentence)

see more in the `readme <https://github.com/sign-language-translator/sign-language-translator#how-to-build-a-translator-for-sign-language>`_.

Text Language Processing
^^^^^^^^^^^^^^^^^^^^^^^^

Process text strings using language specific classes:

.. code-block:: python
   :linenos:
   :caption: Urdu Text Processor

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

Sign Language Processing
^^^^^^^^^^^^^^^^^^^^^^^^

This processes a representation of sign language which mainly consists of the file names of videos.
There are two main parts: 1) word to video mapping and 2) word rearrangement according to grammar.

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
   #   {'signs': [['pk-hfad-1_وہ']], 'weights': [1.0]},
   #   {'signs': [['pk-hfad-1_school']], 'weights': [1.0]},
   #   {'signs': [['pk-hfad-1_گیا']], 'weights': [1.0]}
   # ]

Vision
^^^^^^

This covers the functionalities of representing sign language as objects (sequence of frames e.g. video or sequence of vectors e.g. pose landmarks).
Those objects have built-in functions for data augmentation and visualization etc.

.. code-block:: python
   :linenos:
   :caption: Sign Processing (dummy code until v0.7)

   import sign_language_translator as slt

   # load video
   video = slt.Video("sign.mp4")
   print(video.duration(), video.shape)

   # extract features
   model = slt.models.MediaPipeLandmarksModel()  # default args
   embedding = model.embed(video.iter_frames(), landmark_type="world") # torch.Tensor
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
   image_visualization = sign.frames_grid(steps=5)
   # overlay_visualization = sign.overlay(video) # needs landmark_type="image"

   # display
   video_visualization.show()
   image_visualization.show()
   # overlay_visualization.show()

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

Command line (CLI)
------------------

You can use the following functionalities of the SLT package via CLI as well. A command entered without any arguments will show the help. *The useable model-codes are listed in help*.

Note: Objects & models do not persist in memory across commands, so this is a quick but inefficient way to use this package. In production, create a server which uses the python interface.

Download
^^^^^^^^

Download dataset files or models if you need them. The parameters are regular expressions.

.. code-block:: bash

   slt download --overwrite true '.*\.json' '.*\.mp4'

.. code-block:: bash

   slt download --progress-bar true '.*/tlm_14.0M.pt'

By default, auto-download is enabled. Default download directory is `/install-directory/sign_language_translator/sign-language-resources/`. (See slt.config.settings.Settings)

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

Embed videos into a sequence of vectors using a selected embedding model:

.. code-block:: bash

   slt embed videos/*.mp4 --model-code mediapipe-pose-2-hand-1 --embedding-type world \
      --processes 4 --save-format csv --output-dir ./embeddings

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
