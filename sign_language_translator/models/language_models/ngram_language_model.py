"""This module provides a simple n-gram-based statistical language model implementation.

Classes:
- NgramLanguageModel: A simple n-gram-based statistical language model.
"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from os.path import exists
from typing import Any, Dict, Iterable, List, Tuple

from sign_language_translator.models.language_models.abstract_language_model import (
    LanguageModel,
)
from sign_language_translator.text.utils import make_ngrams
from sign_language_translator.utils import sample_one_index


class NgramLanguageModel(LanguageModel):
    """NgramLanguageModel is a statistical language model based on n-grams. It provides functionality for training the model on a given training corpus, generating the next token based on a context, and saving/loading the model.

    Attributes:
    - window_size (int): The size of the context window for predicting the next token.
    - unknown_token (str): The token representation used for unknown or out-of-vocabulary tokens.
    - sampling_temperature (float): A temperature parameter controlling the sampling probabilities during token generation.
    - name (str): The name of the language model object (optional).

    Methods:
    - train(self, training_corpus): Alias for the fit() method. Trains the language model on the given training corpus.
    - fit(self, training_corpus): Trains the language model on the given training corpus.
    - finetune(self, training_corpus, weightage: float): Fine-tunes the language model on an additional training corpus with a specified weightage.
    - next(self, context: Iterable) -> Tuple[Any, float]: Samples the next token from the learned distribution based on the given context.
    - next_all(self, context: Iterable) -> Tuple[List[Any], List[float]]: Returns a list of possible next tokens and their associated probabilities based on the given context.
    - load(model_path: str) -> NgramLanguageModel: Deserializes the model from a JSON file.
    - save(self, model_path: str, indent=None, ensure_ascii=False): Serializes the model to a JSON file.
    - __str__(self) -> str: Returns a string representation of the NgramLanguageModel instance.

    Private Methods:
    - _to_key_datatype(self, item: Iterable) -> Tuple: Converts an iterable item to the appropriate datatype for use as a key in the model dictionary.
    - _count_ngrams(self, training_corpus: List[Iterable], n: int) -> Dict[Tuple, int]: Counts the occurrences of n-grams in the training corpus.
    - _group_by_context(self, counts: Dict[Tuple, int]): Groups the n-grams by context and calculates the weights for each next token.
    - _count_parameters(self): Counts the total number of weights/probabilities in the model.

    """

    def __init__(
        self,
        window_size=1,
        unknown_token="<unk>",
        sampling_temperature=1.0,
        name=None,
    ) -> None:
        super().__init__(unknown_token=unknown_token, name=name)
        self.window_size = window_size
        self.n_parameters = 0
        self.sampling_temperature = sampling_temperature
        self.model: Dict[Tuple, Dict[str, List]] = {}

        self._NEXT_TOKEN = "NEXT_TOKEN"
        self._WEIGHTS = "WEIGHTS"

        self._sep = "|||"

    def train(self, training_corpus):
        """Alias for fit(). Trains the language model on the given training corpus.

        Args:
            training_corpus (Iterable[Iterable]): The training corpus, an iterable of sequences representing the text data.

        Returns:
            None
        """

        self.fit(training_corpus)

    def fit(self, training_corpus) -> None:
        """Trains the language model on the given training corpus.

        Args:
            training_corpus (Iterable[Iterable]): The training corpus, an iterable of sequences representing the text data.

        Returns:
            None
        """
        counts = self._count_ngrams(training_corpus, self.window_size + 1)
        counts = self._group_by_context(counts)

        # normalize frequencies into probabilities
        for context, _ in counts.items():
            total = sum(counts[context][self._WEIGHTS])
            counts[context][self._WEIGHTS] = [
                freq / total for freq in counts[context][self._WEIGHTS]
            ]

        self.model = counts
        self.n_parameters = self._count_parameters()

    def finetune(self, training_corpus, weightage: float) -> None:
        """Fine-tunes the language model on an additional training corpus with a specified weightage.

        Args:
            training_corpus (Iterable[Iterable]): The additional training corpus, an iterable of sequences representing the text data.
            weightage (float): The weightage for the additional training corpus, a value between 0.0 and 1.0 (inclusive).
                A weightage of 0.0 means no impact from the additional corpus,
                while a weightage of 1.0 means the model is completely updated based on the additional corpus.

        Returns:
            None

        Raises:
            AssertionError: If the weightage is outside the valid range [0.0, 1.0].
        """

        assert 0.0 <= weightage <= 1.0, "provide 0.0 <= weightage <= 1.0"

        old_model = deepcopy(self.model)
        self.fit(training_corpus)

        for context in old_model:
            if context not in self.model:
                self.model[context] = old_model[context]
            else:
                old_weights = dict(
                    zip(
                        old_model[context][self._NEXT_TOKEN],
                        old_model[context][self._WEIGHTS],
                    )
                )

                # integrate weights of existing next_tokens
                for i, next_token in enumerate(self.model[context][self._NEXT_TOKEN]):
                    old_w = old_weights.pop(next_token, 0.0)
                    self.model[context][self._WEIGHTS][i] *= weightage
                    self.model[context][self._WEIGHTS][i] += old_w * (1 - weightage)

                # append weights of remaining next_tokens
                for next_token, old_w in old_weights.items():
                    self.model[context][self._NEXT_TOKEN].append(next_token)
                    self.model[context][self._WEIGHTS].append(old_w * (1 - weightage))

        self.n_parameters = self._count_parameters()

    def next(self, context: Iterable) -> Tuple[Any, float]:
        next_tokens, probabilities = self.next_all(context)

        index = sample_one_index(probabilities, temperature=self.sampling_temperature)
        next_token, probability = next_tokens[index], probabilities[index]

        return (next_token, probability)

    def next_all(self, context: Iterable) -> Tuple[List[Any], List[float]]:
        context = self._to_key_datatype(context[len(context) - self.window_size :])  # type: ignore

        if context not in self.model:
            next_tokens, probabilities = [self.unknown_token], [1.0]
        else:
            next_tokens: List = self.model[context][self._NEXT_TOKEN]
            probabilities: List[float] = self.model[context][self._WEIGHTS]

        return next_tokens, probabilities

    def _to_key_datatype(self, item: Iterable) -> Tuple:
        """Converts an iterable item to the appropriate datatype for use as a key in the model dictionary."""

        return tuple(item)

    def _count_ngrams(
        self, training_corpus: Iterable[Iterable], n: int
    ) -> Dict[Tuple, int]:
        """Counts the occurrences of n-grams in the training corpus."""

        all_ngrams = [
            self._to_key_datatype(ngram)
            for sequence in training_corpus
            for ngram in make_ngrams(sequence, n)
        ]
        counts = dict(Counter(all_ngrams))
        return counts

    def _group_by_context(self, counts: Dict[Tuple, int]):
        """collect all the keys that are the same except the ending part."""

        grouped = {}
        for ngram, freq in counts.items():
            context = ngram[:-1]
            next_token = ngram[-1]

            if context not in grouped:
                grouped[context] = {self._NEXT_TOKEN: [], self._WEIGHTS: []}

            grouped[context][self._NEXT_TOKEN].append(next_token)
            grouped[context][self._WEIGHTS].append(freq)

        return grouped

    def _count_parameters(self):
        """Counts the total number of weights/probabilities in the model."""
        return sum(len(v[self._WEIGHTS]) for v in self.model.values())

    @staticmethod
    def load(model_path: str) -> NgramLanguageModel:
        """Deserializes the model (from JSON).

        Args:
            model_path (str): The source file path.

        Returns:
            NgramLanguageModel: The deserialized NgramLanguageModel instance.
        """

        with open(model_path, "r", encoding="utf-8") as f:
            model_data: Dict[str, Any] = json.load(f)

        window_size = int(model_data["window_size"])
        unknown_token = str(model_data["unknown_token"])
        name = str(model_data["name"])

        next_token_key = str(model_data["NEXT_TOKEN"])
        weights_key = str(model_data["WEIGHTS"])
        _sep = str(model_data["sep"])
        sampling_temperature = float(model_data["sampling_temperature"])

        slm = NgramLanguageModel(
            window_size=window_size,
            unknown_token=unknown_token,
            name=name,
            sampling_temperature=sampling_temperature,
        )
        slm._sep = _sep
        slm._NEXT_TOKEN = next_token_key
        slm._WEIGHTS = weights_key

        slm.model = {
            slm._to_key_datatype(context.split(_sep)): next_token_and_weights
            for context, next_token_and_weights in model_data["model"].items()
        }

        return slm

    def save(
        self, model_path: str, indent=None, ensure_ascii=False, overwrite=False
    ) -> None:
        """
        Serializes the model (as JSON).

        Args:
            model_path (str): The target file path. It will silently overwrite if a file already exists at this path.
            indent (Optional[int]): The indentation level for formatting the JSON data (optional).
            ensure_ascii (bool): Controls whether non-ASCII characters are escaped (optional).
            overwrite (bool): If False, raises FileExistsError if the model already exists. Defaults to False.
        """

        if exists(model_path) and not overwrite:
            raise FileExistsError(f"there is already a file at {model_path = }")

        model_data = dict()
        model_data["window_size"] = self.window_size
        model_data["unknown_token"] = self.unknown_token
        model_data["name"] = self.name

        model_data["NEXT_TOKEN"] = self._NEXT_TOKEN
        model_data["WEIGHTS"] = self._WEIGHTS
        model_data["sep"] = self._sep
        model_data["sampling_temperature"] = self.sampling_temperature

        model_data["model"] = {self._sep.join(k): v for k, v in self.model.items()}

        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=indent, ensure_ascii=ensure_ascii)

    def __str__(self) -> str:
        return f"Ngram LM: {super().__str__()}, window={self.window_size}, params={self.n_parameters}"
