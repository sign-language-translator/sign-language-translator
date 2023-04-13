import os
import random
from typing import Any, Iterable, List, Tuple

from .abstract_language_model import LanguageModel


class Mixer(LanguageModel):
    def __init__(
        self,
        models: List[LanguageModel],
        selection_probabilities: List[float] = None,
        unknown_token="<unk>",
        name=None,
    ) -> None:

        if selection_probabilities is None:
            selection_probabilities = [1] * len(models)

        assert len(models) == len(selection_probabilities)
        assert all(isinstance(p, (float, int)) for p in selection_probabilities)

        super().__init__(unknown_token=unknown_token)
        self.models = models
        total = sum(selection_probabilities)
        self.selection_probabilities = [w / total for w in selection_probabilities]
        self.name = name

    def next(self, context: Iterable) -> Tuple[Iterable, float]:
        # if strategy = "merge"
        #       use finetune weighted_merge for all possible next[_all] token vectors then choose

        # if strategy == "choose"
        tried_models = set()
        while len(tried_models) < len(self.models):
            model_idx = random.choices(
                range(len(self.models)),
                weights=self.selection_probabilities,
                k=1,
            )[0]
            if model_idx in tried_models:
                continue
            else:
                tried_models.add(model_idx)

            token, prob = self.models[model_idx].next(context)
            if token != self.models[model_idx].unknown_token:
                return token, prob

        return self.unknown_token, 0.0

    def next_all(self, context: Iterable) -> Tuple[Iterable[Any], Iterable[float]]:
        return [], []

    def load(self, model_path: str) -> None:
        # folder
        pass

    def save(self, model_path: str) -> None:
        # folder
        pass

    def __str__(self) -> str:
        model_strs = []
        for i, model in enumerate(self.models):
            branch = "├── " if i < len(self.models) - 1 else "└── "
            model_lines = str(model).splitlines()
            for line_no in range(len(model_lines)):
                model_lines[line_no] = (
                    (branch if line_no == 0 else ("│   " if branch == "├── " else "    "))
                    + model_lines[line_no]
                    + (
                        f" | prob={self.selection_probabilities[i]:.1%}"
                        if line_no == 0
                        else ""
                    )
                )
            model_strs.append("\n".join(model_lines))

        mixer_str = (
            f"Mixer LM: {super().__str__()}[{len(self.models)}]\n"
            + "\n".join(model_strs)
        )
        return mixer_str
