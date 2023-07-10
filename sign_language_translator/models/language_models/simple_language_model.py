"""Write new texts with a statistical language model.
"""

import json
import random
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

import numpy

from sign_language_translator.models.language_models.abstract_language_model import (
    LanguageModel,
)
from sign_language_translator.text.metrics import normpdf
from sign_language_translator.text.utils import make_ngrams


class SimpleLanguageModel(LanguageModel):
    def __init__(
        self,
        window_size=1,
        unknown_token="<unk>",
        name=None,
        sampling_temperature=1.0,
    ) -> None:
        super().__init__(unknown_token=unknown_token, name=name)
        self.window_size = window_size

        self.NEXT_TOKEN = "NEXT_TOKEN"
        self.WEIGHTS = "WEIGHTS"

        self.sep = "|||"
        self.n_parameters = 0
        self.sampling_temperature = sampling_temperature

    def _to_key_datatype(self, item: Iterable) -> Tuple:
        return tuple(item)

    def _count_ngrams(
        self, training_corpus: List[Iterable], n: int
    ) -> Dict[Tuple, int]:
        all_ngrams = [
            self._to_key_datatype(ngram)
            for sequence in training_corpus
            for ngram in make_ngrams(sequence, n)
        ]
        counts = dict(Counter(all_ngrams))
        return counts

    def _group_by_context(self, counts: Dict[Tuple, int]):
        grouped = {}
        for ngram, freq in counts.items():
            context = ngram[:-1]
            next_token = ngram[-1]

            if context not in grouped:
                grouped[context] = {self.NEXT_TOKEN: [], self.WEIGHTS: []}

            grouped[context][self.NEXT_TOKEN].append(next_token)
            grouped[context][self.WEIGHTS].append(freq)

        return grouped

    def train(self, training_corpus):
        self.fit(training_corpus)

    def fit(self, training_corpus) -> None:
        counts = self._count_ngrams(training_corpus, self.window_size + 1)
        counts = self._group_by_context(counts)

        # normalize frequencies into probabilities
        for context in counts:
            total = sum(counts[context][self.WEIGHTS])
            counts[context][self.WEIGHTS] = [
                freq / total for freq in counts[context][self.WEIGHTS]
            ]

        self.model: Dict[Tuple, Dict[str, List]] = counts
        self.n_parameters = self._count_parameters()
        (
            self.data_distribution_mean,
            self.data_distribution_std,
        ) = self._calculate_data_distribution(training_corpus)

    def finetune(self, training_corpus, weightage: float):
        assert 0.0 <= weightage <= 1.0

        old_model = deepcopy(self.model)
        self.fit(training_corpus)

        for context in old_model:
            if context not in self.model:
                self.model[context] = old_model[context]
            else:
                old_weights = dict(
                    zip(
                        old_model[context][self.NEXT_TOKEN],
                        old_model[context][self.WEIGHTS],
                    )
                )

                for i in range(len(self.model[context][self.NEXT_TOKEN])):
                    next_token = self.model[context][self.NEXT_TOKEN][i]
                    old_w = old_weights.pop(next_token, 0.0)
                    self.model[context][self.WEIGHTS][i] *= weightage
                    self.model[context][self.WEIGHTS][i] += old_w * (1 - weightage)

                for next_token, old_w in old_weights.items():
                    self.model[context][self.NEXT_TOKEN].append(next_token)
                    self.model[context][self.WEIGHTS].append(old_w * (1 - weightage))

        self.n_parameters = self._count_parameters()
        (mean, std) = self._calculate_data_distribution(training_corpus)
        self.data_distribution_mean *= (1-weightage)
        self.data_distribution_mean += weightage * mean
        self.data_distribution_std *= (1-weightage)
        self.data_distribution_std += weightage * std

    def _count_parameters(self):
        return sum(len(v[self.WEIGHTS]) for v in self.model.values())

    def _choose_index(self, weights: List[float]) -> int:
        """Select an item based on the given probability distribution. Returns the index of the selected item sampled from weighted random distribution.

        Args:
            weights (List[float]): the relative weights corresponding to each index.

        Returns:
            int: The index of the chosen item.
        """
        return random.choices(
            range(len(weights)),
            weights=[w / self.sampling_temperature for w in weights],
            k=1,
        )[0]

    def next(self, context: Iterable) -> Tuple[Iterable, float]:
        """Samples the next token from the learnt distribution based on the given context.

        Args:
            context (Iterable): A piece of sequence like the training examples.

        Returns:
            Tuple[Iterable, float]: The next token and its associated probability. Token has the same type as the context iterable.
        """

        next_tokens, probabilities = self.next_all(context)

        index = self._choose_index(probabilities)
        next_token, probability = next_tokens[index], probabilities[index]

        return (next_token, probability)

    def next_all(self, context: Iterable) -> Tuple[List[Iterable], List[float]]:
        context = self._to_key_datatype(context[len(context) - self.window_size :])

        if context not in self.model:
            next_tokens, probabilities = [self.unknown_token], [1.0]
        else:
            next_tokens = self.model[context][self.NEXT_TOKEN]
            probabilities = self.model[context][self.WEIGHTS]

        return next_tokens, probabilities

    def load(self, model_path: str, model_data: str = None) -> None:
        """Deserializes the model (from json).

        Args:
            model_path (str): The source file path.
        """

        if model_data is None:
            with open(model_path, "r") as f:
                model_data = json.load(f)

        self.window_size = model_data["window_size"]
        self.unknown_token = model_data["unknown_token"]

        self.NEXT_TOKEN = model_data["NEXT_TOKEN"]
        self.WEIGHTS = model_data["WEIGHTS"]
        self.sep = model_data["sep"]
        self.sampling_temperature = model_data["sampling_temperature"]

        self.model = {
            self._to_key_datatype(context.split(self.sep)): next_token_and_weights
            for context, next_token_and_weights in model_data["model"].items()
        }

    def save(self, model_path: str, indent=None, ensure_ascii=False) -> None:
        """Serializes the model (as json).

        Args:
            model_path (str): The target file path. It will silently overwrite if a file already exists at this path.
        """

        model_data = dict()
        model_data["window_size"] = self.window_size
        model_data["unknown_token"] = self.unknown_token

        model_data["NEXT_TOKEN"] = self.NEXT_TOKEN
        model_data["WEIGHTS"] = self.WEIGHTS
        model_data["sep"] = self.sep
        model_data["sampling_temperature"] = self.sampling_temperature

        model_data["model"] = {self.sep.join(k): v for k, v in self.model.items()}

        with open(model_path, "w") as f:
            json.dump(model_data, indent=indent, ensure_ascii=ensure_ascii)

    def __str__(self) -> str:
        return f"Simple LM: {super().__str__()}, window={self.window_size}, params={self.n_parameters}"

    def probability(self, token_sequence, return_log_probability=True):
        probabilities = []
        for i in range(self.window_size, len(token_sequence) - 1):
            context = self._to_key_datatype(token_sequence[i - self.window_size : i])
            next_token = token_sequence[i]
            if next_token not in self.model.get(context,{}).get(self.NEXT_TOKEN, []):
                return float("-inf") if return_log_probability else 0.0
            _idx = self.model[context][self.NEXT_TOKEN].index(next_token)
            probabilities.append(self.model[context][self.WEIGHTS][_idx])

        probability = (
            numpy.prod(probabilities)
            if not return_log_probability
            else numpy.log(probabilities).sum()
        )
        return probability

    def _calculate_data_distribution(self, corpus):
        log_probabilities = [
            self.probability(item, return_log_probability=True)
            for item in corpus
        ]
        mean = numpy.mean(log_probabilities)
        std = numpy.std(log_probabilities)
        return mean, std

    def calculate_likelihood(self, token_sequence):
        log_prob = self.probability(
            token_sequence, return_log_probability=True
        )
        likelihood = normpdf(
            log_prob,
            self.data_distribution_mean,
            self.data_distribution_std,
        )
        return likelihood
