"""text evaluation metrics
"""

from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy

from .utils import make_ngrams


def cosine_similiarity(text1, text2):

    # word_embedding = word2embd[word]
    # similarities   = cosine_similarity( word_embedding.reshape(1,-1),
    #                                     supported_words_embeddings    ).flatten()
    # smilarities.argsort()[-top_n:] # get indexes of top 5 values in array
    # top_n_thresh_indexes = to

    return 1.0


def word_error_rate(reference, candidate):
    # pip install jiwer
    return 1.0


def rouge(reference, candidate):
    # use counters
    ref = set(reference)
    return len(set(candidate) & ref) / len(ref)


def bleu(reference, candidate):
    # use counters
    can = set(candidate)
    return len(set(reference) & can) / len(can)


def f1_score(reference, candidate):

    bleu_score = bleu(reference, candidate)
    rogue_score = rouge(reference, candidate)

    return 2 * bleu_score * rogue_score / ((bleu_score + rogue_score) or 1)


class NoveltyScore:
    def __init__(
        self,
        n_to_ngram_to_frequency: Dict[int, Dict[Tuple[Any], int]],
        n_to_use: List[int] = None,
        scaling_factor: float = 1.0,
        n_weights=None,
    ) -> None:

        self.n_to_ngram_to_frequency = n_to_ngram_to_frequency
        self.n_to_use: List[int] = (
            n_to_use if n_to_use is not None else list(n_to_ngram_to_frequency.keys())
        )
        self.n_weights: List[float] = (
            n_weights if n_weights else [1.0] * len(self.n_to_use)
        )
        self.scaling_factor: float = scaling_factor
        assert len(n_to_use) == len(n_weights)
        assert set(n_to_use) <= set(
            n_to_ngram_to_frequency.keys()
        ), "cannot use values of n whose frequencies are not available"

    def frequency_to_score(self, frequency: int) -> float:
        return -numpy.log(frequency) * self.scaling_factor

    def __call__(self, sequence: Iterable) -> float:
        return self.score(sequence)

    def score(self, sequence: Iterable[str]) -> float:
        score = 0
        for weight, n in zip(self.n_weights, self.n_to_use):
            ngrams = make_ngrams(sequence, n)
            score += weight * sum(
                [
                    self.frequency_to_score(self.n_to_ngram_to_frequency[n].get(tuple(ng), 1))
                    for ng in ngrams
                ]
            )
        score /= sum(weight)

        return score

    def count(self, tokenized_corpus: List[List[str]]):
        return {
            n: dict(
                Counter(
                    tuple(ngram)
                    for line in tokenized_corpus
                    for ngram in make_ngrams(line, n)
                )
            )
            for n in self.n_to_use
        }
