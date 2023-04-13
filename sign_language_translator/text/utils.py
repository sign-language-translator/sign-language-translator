"""other functions
"""

from typing import Iterable, List

import numpy


def find_similar_available_words(
    candidate: str, vocab, threshold=0.5, n_words=8
) -> List[str]:
    return [candidate]


def make_ngrams(sequence: Iterable, n: int) -> List[Iterable]:
    """Create all possible slices of the given iterable of size n. for example, sequence="1234" and n=2 would create ["12","23","34"].

    Args:
        sequence (Iterable): _description_
        n (int): _description_

    Returns:
        List[Iterable]: _description_
    """

    start = 0
    end = len(sequence) - n

    ngrams = (
        [sequence[i : i + n] for i in range(start, end + 1)] if end >= start else []
    )
    return ngrams

def normpdf(x:float, mean:float, std:float):
    denominator = std * ((2.0 * numpy.pi) ** 0.5)
    numerator = numpy.exp(-0.5 * ((x - mean) / std) ** 2)
    return numerator / denominator


# def calculate_signgram_frequency(
#     self, n_to_ngram_to_frequency: Dict[int, Dict[Tuple[str], int]]
# ):
#     signgram_frequencies = {}
#     for n, ngram_to_freq in n_to_ngram_to_frequency.items():
#         signgram_frequencies[n] = {}
#         for words, freq in ngram_to_freq.items():
#             signgram = self.make_signgram(words)
#             if signgram not in signgram_frequencies[n]:
#                 signgram_frequencies[n][signgram] = 0 + 1
#             signgram_frequencies[n][signgram] += freq

#     return signgram_frequencies