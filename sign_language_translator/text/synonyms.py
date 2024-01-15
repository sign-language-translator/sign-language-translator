"""
This module provides a SynonymFinder class that can find synonyms of a given text
by utilizing translation and back-translation or similarity in embedding vectors.

Dependencies:
- deep_translator

Classes:
    - SynonymFinder: A class for finding synonyms using translation and similarity methods.
"""

from collections import Counter
from typing import List, Optional

from deep_translator import GoogleTranslator
from deep_translator.exceptions import NotValidPayload
from urllib3.exceptions import MaxRetryError, SSLError

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

    def __init__(self, language: str) -> None:
        """
        Initialize a SynonymFinder object.

        Args:
            language (str): The target language for translation based synonyms. Use 2-letter codes (ISO 639-1).
        """
        self.language = language

        self._translator = None
        self._intermediate_languages = None
        self._embedding_model = None

    @property
    def translator(self):
        """
        The GoogleTranslator object with the source language as "auto" and the
        target language as the __init__ argument or according to the current state.
        """
        if self._translator is None:
            self._translator = GoogleTranslator(source="auto", target=self.language)
        return self._translator

    @property
    def intermediate_languages(self) -> List[str]:
        """
        Returns a list of languages supported by the translator, excluding the current language.
        They are used to find synonyms by translation and back-translation. These are 2-letter codes (ISO 639-1).
        """
        if not self._intermediate_languages:
            self._intermediate_languages = [
                lang
                for lang in self.translator.get_supported_languages(as_dict=True).values()  # type: ignore
                if lang != self.language
            ]
        return self._intermediate_languages

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            pass  # "paraphrase-distilroberta-base-v1"
        return self._embedding_model

    def synonyms_by_translation(
        self,
        text: str,
        intermediate_languages: Optional[List[str]] = None,
        time_delay: float = 1e-2,
        timeout: float | None = 10,
        max_n_threads: int = 132,
        lower_case: bool = True,
        progress_bar: bool = True,
        leave: bool = False,
    ) -> List[str]:
        """
        Translates the given text into intermediate languages and performs back-translation to obtain synonyms.
        Translation is done via the internet using web scraping by the deep_translator library.

        Args:
            text (str): The text to be translated.
            intermediate_languages (Optional[List[str]]): List of intermediate languages to translate the text into. Use 2-letter codes (ISO 639-1). If None, all supported languages of the translator will be used. Defaults to None.
            time_delay (float): Time delay between translation requests (in seconds). Defaults to 1e-2.
            timeout (float | None): The maximum amount of time (in seconds) to wait for a thread to finish. None means wait indefinitely. Defaults to 10.
            max_n_threads (int): Maximum number of threads to use for parallel translation. Defaults to 128.
            lower_case (bool): Whether to convert the synonyms to lowercase. Defaults to True.
            progress_bar (bool): Whether to display a progress bar during translation. Defaults to True.
            leave (bool): Whether to leave the progress bar after translation. Defaults to True.

        Returns:
            List[str]: A list of synonyms obtained through back-translation from other languages.
        """

        # setup
        if intermediate_languages is None:
            intermediate_languages = self.intermediate_languages

        def translation_function(text: str, target_lang: str, translations: List[str]):
            try:
                translations.append(self.translate(text, target_lang))
            # catch  and pass
            except Exception:
                pass

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
            synonyms = [syn.lower() for syn in synonyms]

        # sort by frequency
        synonyms = [
            txt_
            for txt, _ in Counter(synonyms).most_common()
            if (txt_ := txt.strip()) not in ("",)
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
            return self.translator.translate(text)
        except (NotValidPayload, MaxRetryError, SSLError):
            return ""

    def synonyms_by_similarity(self, text: str, threshold=0.8) -> List[str]:
        """Looks into a vector database and returns the closest matches to the input text.

        Args:
            text (str): The input text to find synonyms for.
            threshold (float, optional): Cut off value for similarity between embedding vectors. Words with greater similarity score than this value are returned as synonyms. Defaults to 0.8.

        Returns:
            List[str]: A list of synonyms for the input text.
        """
        # TODO: Implement
        return []
