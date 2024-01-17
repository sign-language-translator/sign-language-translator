"""This module provides the MixerLM class, which is a language model that allows sampling from multiple language models.

Classes:
- MixerLM: A language model that combines the outputs of multiple language models based on a selection strategy. It extends the abstract base class LanguageModel.
"""

from __future__ import annotations

import pickle
from os.path import exists
from typing import Any, Iterable, List, Optional, Tuple

from sign_language_translator.models.language_models.abstract_language_model import (
    LanguageModel,
)
from sign_language_translator.utils import sample_one_index


class MixerLM(LanguageModel):
    """The MixerLM class is a language model that combines multiple language models using a mixing strategy. It extends the abstract base class LanguageModel.

    Attributes:
    - models (List[LanguageModel]): List of language models to be combined.
    - selection_probabilities (List[float] | None): The selection probabilities for each language model.
                                If not provided, equal probabilities are assigned.
    - unknown_token (str): The token representation used for unknown or out-of-vocabulary tokens.
    - model_selection_strategy (str): The strategy for selecting the next token from the language models.
        Possible values: "choose" (selects one model and infers through it).
                         "merge"  (infers through all models & combines their output probabilities).
    - name (str): The name of the mixer language model object (optional).

    Methods:
    - next(self, context: Iterable) -> Tuple[Any, float]: Generates the next token based on the given context.
    - next_all(self, context: Iterable) -> Tuple[List[Any], List[float]]: Generates all next
        tokens and their associated probabilities based on the given context.
    - save(self, model_path: str) -> None: saves the mixer model as a pickle file.
    - load(model_path: str) -> MixerLM: loads the mixer model from a pickle file.
    - __str__(self) -> str: Returns a string representation of the MixerLM instance.
    """

    def __init__(
        self,
        models: List[LanguageModel],
        selection_probabilities: Optional[List[float]] = None,
        unknown_token="<unk>",
        name=None,
        model_selection_strategy="choose" or "merge",
    ) -> None:
        if selection_probabilities is None:
            selection_probabilities = [1.0] * len(models)

        assert len(models) == len(selection_probabilities)
        assert all(isinstance(p, (float, int)) for p in selection_probabilities)

        super().__init__(unknown_token=unknown_token)
        self.models = models
        total = sum(selection_probabilities)
        self.selection_probabilities = [w / total for w in selection_probabilities]
        self.name = name
        assert model_selection_strategy in {"merge", "choose"}
        self.strategy = model_selection_strategy

    def next(self, context: Iterable) -> Tuple[Any, float]:
        token_and_prob = (self.unknown_token, 1.0)

        if self.strategy == "merge":
            tokens, probabilities = self.next_all(context)
            i = sample_one_index(probabilities)
            token_and_prob = tokens[i], probabilities[i]

        elif self.strategy == "choose":
            tried_models = set()
            while len(tried_models) < len(self.models):
                # select model
                model_idx = sample_one_index(self.selection_probabilities)
                if model_idx in tried_models:
                    continue
                tried_models.add(model_idx)

                # sample next token
                token, prob = self.models[model_idx].next(context)
                if token != self.models[model_idx].unknown_token:
                    token_and_prob = token, prob
                    break

        return token_and_prob

    def next_all(self, context: Iterable) -> Tuple[List[Any], List[float]]:
        """Computes probabilities for all next tokens based on the given context and returns them both.

        If model selection strategy is "choose" then selects one model and infers through it.
        If model_selection_strategy is "merge" then for each language model,
        it generates the all next tokens and probabilities. It combines the tokens and probabilities
        from all models to create a list of unique next tokens and their corresponding weighted probabilities.

        Args:
            context (Iterable): A piece of sequence like the training examples.

        Returns:
            Tuple[List[Any], List[float]]: A tuple containing a list of unique next tokens and their corresponding probabilities.
        """

        next_tokens = []
        probabilities = []
        total_weight = 1.0
        for model, weight in zip(self.models, self.selection_probabilities):
            all_token, all_probability = model.next_all(context)
            if model.unknown_token not in all_token:
                for token, prob in zip(all_token, all_probability):
                    if token not in next_tokens:
                        next_tokens.append(token)
                        probabilities.append(0)
                        i = -1
                    else:
                        i = next_tokens.index(token)
                    probabilities[i] += prob * weight
            else:
                total_weight -= weight

        if not next_tokens:
            return [self.unknown_token], [1.0]

        if total_weight != 1.0:
            probabilities = [p / total_weight for p in probabilities]
        assert (
            abs(sum(probabilities) - 1) < 0.001
        ), f"{probabilities= }\nsum = {sum(probabilities)}"

        return next_tokens, probabilities

    def save(self, model_path: str, overwrite=False) -> None:
        """
        Save the model to a file.

        Args:
            model_path (str): The path to save the model.
            overwrite (bool, optional): Whether to overwrite an existing file. Defaults to False.

        Raises:
            FileExistsError: If a file already exists at `model_path` and `overwrite` is False.
        """

        if exists(model_path) and not overwrite:
            raise FileExistsError(f"there is already a file at {model_path = }")

        with open(model_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(model_path: str) -> MixerLM:
        """
        Loads a MixerLM model from the given model path.

        Parameters:
            model_path (str): The path to the model file.

        Returns:
            MixerLM: The loaded MixerLM model.
        """

        with open(model_path, "rb") as f:
            mixer_model = pickle.load(f)
        return mixer_model

    def __str__(self) -> str:
        # don't judge my code by its looks.. it works!
        model_strs = []
        for i, model in enumerate(self.models):
            branch = "├── " if i < len(self.models) - 1 else "└── "
            model_lines = str(model).splitlines()
            for line_no, _ in enumerate(model_lines):
                model_lines[line_no] = (
                    (
                        branch
                        if line_no == 0
                        else ("│   " if branch == "├── " else "    ")
                    )
                    + model_lines[line_no]
                    + (
                        f" | prob={self.selection_probabilities[i]:.1%}"
                        if line_no == 0
                        else ""
                    )
                )
            model_strs.append("\n".join(model_lines))

        mixer_str = f"Mixer LM: {super().__str__()}[{len(self.models)}]\n" + "\n".join(
            model_strs
        )
        return mixer_str
