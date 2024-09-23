"""
This module provides a SynonymFinder class that can find synonyms of a given text
by utilizing translation and back-translation or similarity in embedding vectors.

Dependencies:
- deep_translator

Classes:
    - SynonymFinder: A class for finding synonyms using translation and similarity methods.
"""

from collections import Counter
from typing import Dict, List, Optional
from warnings import warn

from urllib3.exceptions import HTTPError

try:
    from deep_translator import GoogleTranslator
    from deep_translator.exceptions import (
        BaseError as DeepTranslatorError,
        TooManyRequests,
    )
except ImportError:
    GoogleTranslator = None
    DeepTranslatorError = None
    TooManyRequests = None

from sign_language_translator.utils.parallel import threaded_map


class SynonymFinder:
    """
    This class provides methods for finding synonyms of a given text using two different approaches:
    1. Translation and back-translation through the 'synonyms_by_translation' method (requires internet).
    2. Embedding-based similarity search through the 'synonyms_by_similarity' method.

    Attributes:
        language (str): The target language for translation. Use 2-letter codes (ISO 639-1).
        translator (GoogleTranslator): The translator object for language translation.
        intermediate_languages (List[str]): List of languages supported by the translator, excluding the current language.
        embedding_model (str): The embedding model for similarity-based synonym finding.

    Methods:
        synonyms_by_translation: Finds synonyms by translating text into an intermediate language and then back-translation.
        synonyms_by_similarity: Finds synonyms based on embedding vector similarity.
        translate: Translates text to the specified target language.

    Example:

        .. code-block:: python

            # Instantiate SynonymFinder with the target language
            synonym_finder = SynonymFinder("en")

            # Find synonyms using translation and back-translation
            text = "happy"
            synonyms = synonym_finder.synonyms_by_translation(text)
            print(f"Synonyms by Translation: {synonyms}")

            # Find synonyms using similarity based on embedding vectors
            text = "joyful"
            synonyms = synonym_finder.synonyms_by_similarity(text)
            print(f"Synonyms by Similarity: {synonyms}")
    """

    def __init__(self, language: str = "en") -> None:
        """
        Initialize a SynonymFinder object.

        Args:
            language (str): The target language for translation based synonyms. Use 2-letter codes (ISO 639-1). Defaults to "en".
        """
        self._language = language

        self._translator = None
        self._intermediate_languages = None
        self._embedding_model = None

    @property
    def language(self) -> str:
        """The target language for translation. Use 2-letter codes (ISO 639-1)."""
        return self._language

    @language.setter
    def language(self, language: str) -> None:
        self._language = language
        self._embedding_model = None

    @property
    def translator(self):
        """
        The deep_translator.GoogleTranslator object with the source language as "auto" and the
        target language as the __init__ argument or according to the current state.
        """
        if self._translator is None:
            if GoogleTranslator is None:
                raise ImportError(
                    "The 'deep_translator' package is required for translation-based synonym finding. "
                    "Install it using `pip install sign-language-translator[synonyms]`."
                )

            self._translator = GoogleTranslator(source="auto", target=self.language)
        return self._translator

    @property
    def intermediate_languages(self) -> List[str]:
        """
        Returns a list of languages supported by the translator, excluding the current language.
        They are used to find synonyms by translation and back-translation. These are 2-letter codes (ISO 639-1).
        """
        if not self._intermediate_languages:
            self._intermediate_languages = list(
                self.translator.get_supported_languages(as_dict=True).values()  # type: ignore
            )
        return self._intermediate_languages

    def synonyms_by_translation(
        self,
        text: str,
        intermediate_languages: Optional[List[str]] = None,
        min_frequency: int = 1,
        time_delay: float = 1e-2,
        timeout: Optional[float] = 10,
        max_n_threads: int = 132,
        lower_case: bool = True,
        progress_bar: bool = True,
        leave: bool = False,
        cache: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> List[str]:
        """
        Translates the given text into intermediate languages and performs back-translation to obtain synonyms.
        Translation is done via the internet using web scraping by the deep_translator library.

        Args:
            text (str): The text to be translated.
            intermediate_languages (Optional[List[str]]): List of intermediate languages to translate the text into. Use 2-letter codes (ISO 639-1). If None, all supported languages of the translator will be used. Defaults to None.
            min_frequency (int): Minimum occurrence count for synonyms to get considered. Value is inclusive. Defaults to 1.
            time_delay (float): Time delay between translation requests (in seconds). Defaults to 1e-2.
            timeout (float | None): The maximum amount of time (in seconds) to wait for a thread to finish. None means wait indefinitely. Defaults to 10.
            max_n_threads (int): Maximum number of threads to use for parallel translation. Defaults to 128.
            lower_case (bool): Whether to convert the synonyms to lowercase. Defaults to True.
            progress_bar (bool): Whether to display a progress bar during translation. Defaults to True.
            leave (bool): Whether to leave the progress bar after translation. Defaults to True.
            cache (Optional[Dict[str, Dict[str, str]]]): A dictionary to save or retrieve the intermediate translations of the `text`. Structure is `{"text": {"language": "translation", ...}, ...}` where each input maps to a dict mapping language code to the text's translation. Defaults to None.

        Returns:
            List[str]: A list of synonyms obtained through back-translation from other languages.
        """

        # setup
        if intermediate_languages is None:
            intermediate_languages = self.intermediate_languages

        def translation_function(text: str, target_lang: str, translations: List[str]):
            if (
                isinstance(cache, dict)
                and (text in cache)
                and (target_lang in cache[text])
            ):
                translations.append(cache[text][target_lang])
            else:
                if translation := self.translate(text, target_lang):
                    translations.append(translation)
                    if isinstance(cache, dict):
                        cache.setdefault(text, {})[target_lang] = translation

        # translation into intermediate languages
        translations = []
        threaded_map(
            translation_function,
            [(text, lang, translations) for lang in intermediate_languages],
            time_delay=time_delay,
            timeout=timeout,
            max_n_threads=max_n_threads,
            progress_bar=progress_bar,
            leave=leave,
            # progress_callback=progress_callback,
        )

        # back-translation into source language
        synonyms = []
        threaded_map(
            translation_function,
            [
                (translation.strip(), self.language, synonyms)
                for translation in set(translations + [text])
                if translation.strip()
            ],
            time_delay=time_delay,
            timeout=timeout,
            max_n_threads=max_n_threads,
            progress_bar=progress_bar,
            leave=leave,
        )

        # preprocess
        if lower_case:
            synonyms = [str(syn).lower() for syn in synonyms]
        synonyms = [stripped for syn in synonyms if (stripped := str(syn).strip())]

        # sort by frequency
        synonyms = [
            txt
            for txt, freq in Counter(synonyms).most_common()
            if freq >= min_frequency
        ]

        return synonyms

    def translate(self, text: str, target_language: str) -> str:
        """
        Translates the given text to the specified target language.

        Args:
            text (str): The text to be translated.
            target_language (str): The target language for translation. Use 2-letter codes (ISO 639-1).

        Returns:
            str: The translated text.
        """
        try:
            self.translator.target = target_language
            return str(self.translator.translate(text)).strip()
        except (
            HTTPError,
            DeepTranslatorError or HTTPError,
            TooManyRequests or HTTPError,
        ) as exc:
            warn(f"Translation failed for '{text}' to '{target_language}'.Error: {exc}")
            return ""

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from sign_language_translator.models._utils import get_model

            self._embedding_model = get_model(f"lookup-{self.language}-fasttext-cc.pt")
        return self._embedding_model

    def synonyms_by_similarity(
        self, text: str, top_k=10, min_similarity=0.5
    ) -> List[str]:
        """Looks into a vector database and returns the closest matches to the input text.

        Args:
            text (str): The input text to find synonyms for.
            top_k (int, optional): The maximum number of synonyms to return. Defaults to 10.
            min_similarity (float, optional): Cut off value for similarity between embedding vectors. Words with greater similarity score than this value are returned as synonyms. Defaults to 0.8.

        Returns:
            List[str]: A list of synonyms for the input text.

        Example:

            .. code-block:: python

                # Instantiate SynonymFinder with the target language
                synonym_finder = SynonymFinder("ur")

                # Find synonyms using similarity based on embedding vectors
                text = "تعلیم"
                synonyms = synonym_finder.synonyms_by_similarity(text, 3)
                print(synonyms)
                # ["تعلیم", "تربیت", "تعلیمی"]
        """

        # TODO: search with a different language or by vector

        vector = self.embedding_model.embed(text)  # type: ignore
        synonyms, scores = self.embedding_model.similar(vector, k=top_k)  # type: ignore

        return [syn for syn, score in zip(synonyms, scores) if score > min_similarity]
