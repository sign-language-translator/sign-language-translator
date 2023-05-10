import math
import random
from typing import Any, Iterable

# from ...text.utils import novelity_score
from .abstract_language_model import LanguageModel


class BeamSampling:
    def __init__(
        self,
        model: LanguageModel,
        beam_width: int = 3,
        start_of_sequence_token=["["],  # ["<s>"],
        end_of_sequence_token=["]"],  # ["</s>"],
        max_length=33,
        scoring_function=lambda seq, log_prob: 10.0 + log_prob / len(seq),
        return_log_probability=True,
    ) -> None:
        self.model = model
        self.start_of_sequence_token = start_of_sequence_token
        self.end_of_sequence_token = end_of_sequence_token
        self.beam_width = beam_width
        self.max_length = max_length
        self.scoring_function = scoring_function
        self.return_log_probability = return_log_probability

    def __call__(self, context: Iterable = None) -> Iterable:
        return self.complete(context=context)

    def concatenate(self, context, token):
        if isinstance(context, str):
            joint = context + token
        elif isinstance(context, list):
            joint = context + [token]
        elif isinstance(context, tuple):
            joint = context + (token,)
        # numpy / torch

        return joint

    def complete(self, context: Iterable = None) -> Iterable:
        if context is None:
            context = self.start_of_sequence_token

        branches = [(context, 0.0)]

        for _ in range(self.max_length):
            n_branches = round(self.beam_width + random.random() * 0.8 - 0.4)
            # Expand
            branches_ = []
            for context_, score_ in branches:
                if (
                    # ended
                    context_[-1] == self.end_of_sequence_token
                    or len(context_) >= self.max_length
                ):
                    branches_.append((context_, score_))
                else:
                    for _ in range(n_branches):
                        next_token, prob = self.model.next(context_)
                        if next_token == self.model.unknown_token:
                            score = score_
                            next_context = context_
                        else:
                            next_context = self.concatenate(context_, next_token)
                            score = score_ + math.log(prob)
                        branches_.append((next_context, score))

            # Prune
            branches_ = sorted(
                branches_,
                key=lambda item: self.scoring_function(*item),
                reverse=True,
            )[:n_branches]

            if branches == branches_:
                branches = branches_
                break
            else:
                branches = branches_

        weights = [self.scoring_function(seq, log_prob) for seq, log_prob in branches]

        # softmax
        weights = [math.exp(w) for w in weights]
        weights = [w / sum(weights) for w in weights]

        selected_completion, score = random.choices(
            branches,
            weights=weights,
            k=1,
        )[0]

        if not self.return_log_probability:
            score = math.exp(score)

        return selected_completion, score
