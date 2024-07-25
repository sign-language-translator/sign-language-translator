"""This module provides BeamSampling class for generating completions using beam search sampling."""

from math import exp, log2
from random import random
from typing import Any, Callable, Iterable, Optional, Tuple

from sign_language_translator.models.language_models.abstract_language_model import (
    LanguageModel,
)
from sign_language_translator.utils import sample_one_index


class BeamSampling:
    """BeamSampling class for generating completions using beam search sampling.

    Args:
        model (LanguageModel): The language model used for generating completions.
        beam_width (int, optional): The beam width for beam search. Defaults to 3.
        start_of_sequence_token (str, optional): The start of sequence token. Defaults to "[".
        end_of_sequence_token (str, optional): The end of sequence token. Defaults to "]".
        max_length (int, optional): The maximum length of the generated completions. Defaults to 33.
        scoring_function (Callable[[Iterable, float], float], optional):
            The scoring function used to score the completions. It should accept the generated
            sequence and its overall log probability as arguments. Defaults to a linear function.

        return_log_of_probability (bool, optional):
            A flag indicating whether to return the probability of the completions
            or log2 of it. Defaults to True.
    """

    def __init__(
        self,
        model: LanguageModel,
        beam_width: int = 3,
        start_of_sequence_token="[",
        end_of_sequence_token="]",
        max_length: int = 37,
        scoring_function: Callable[[Iterable, float], float] = (
            lambda seq, log_prob: 10.0 + log_prob / len(seq)  # type: ignore
        ),
        return_log_of_probability: bool = True,
    ) -> None:
        self.model = model
        self.start_of_sequence_token = start_of_sequence_token
        self.end_of_sequence_token = end_of_sequence_token
        self.beam_width = beam_width
        self.max_length = max_length
        self.scoring_function = scoring_function
        self.return_log_of_probability = return_log_of_probability

    def __call__(self, context: Optional[Iterable] = None) -> Iterable:
        return self.complete(initial_context=context)

    def complete(
        self,
        initial_context: Optional[Iterable] = None,
        append_func: Callable[[Any, Any], Any] = lambda context, token: (
            (context + [token])
            if isinstance(context, list)
            else (
                (context + (token,))
                if isinstance(context, tuple)
                else (context + token)
            )
        ),
    ) -> Tuple[Iterable, float]:
        """Generate completions based on the given initial context.

        Args:
            initial_context (Iterable | None, optional): The initial context for completion generation. Defaults to None.
            append_func (Callable[[Any, Any], Any], optional): a function that can append the generated next token to provided context.
                Defaults to a lambda function that can append to list, tuple & str.

        Returns:
            Tuple[Iterable, float]: One generated completion and its score.
        """

        if initial_context is None:
            initial_context = [self.start_of_sequence_token]

        branches = [(initial_context, 0.0)]

        for _ in range(self.max_length):
            n_branches = round(self.beam_width + random() * 0.8 - 0.4)

            # Expand
            new_branches = []
            for context_, score_ in branches:
                if (  # sequence completed
                    context_[-1] == self.end_of_sequence_token  # type: ignore
                    or len(context_) >= self.max_length  # type: ignore
                ):
                    # just as it was. no change.
                    new_branches.append((context_, score_))
                    continue

                # append next tokens
                for _ in range(n_branches):
                    next_token, prob = self.model.next(context_)
                    if next_token == self.model.unknown_token:
                        score = score_
                        next_context = context_
                    else:
                        next_context = append_func(context_, next_token)
                        score = score_ + log2(prob)

                    # no repeats; for diversity.
                    if (next_context, score) not in new_branches:
                        new_branches.append((next_context, score))

            # Prune
            new_branches = sorted(
                new_branches,
                key=lambda item: self.scoring_function(*item),
                reverse=True,
            )[:n_branches]

            if branches == new_branches:
                # no branch has grown further: stop
                break

            branches = new_branches

        # softmax: turn each branch's scores into a probability distribution
        weights = [self.scoring_function(seq, log_prob) for seq, log_prob in branches]
        weights = [exp(w) for w in weights]
        weights = [w / sum(weights) for w in weights]

        # select one branch
        selected_completion, score = branches[sample_one_index(weights)]

        # reformat score
        if not self.return_log_of_probability:
            score = 2**score  # math.exp2(score)

        return selected_completion, score
