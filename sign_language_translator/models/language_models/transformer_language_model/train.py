"""This module contains classes to train transformer language models.

Classes:
    LM_Dataset(torch.utils.data.Dataset): subclass for language model. has a process function that can convert text file into a list of tensors.
    LM_Trainer: Trainer class for language model. runs training loop, prints metrics and makes model checkpoints.
"""

from __future__ import annotations

import os
from glob import glob
from time import time
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from sign_language_translator.models.utils import (
    FullyLambdaLR,
    set_layers_trainability_,
)
from sign_language_translator.text.utils import make_ngrams

if TYPE_CHECKING:
    from sign_language_translator.models.language_models.transformer_language_model.model import (
        TransformerLanguageModel,
    )


class LM_Dataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx][..., :-1], self.data[idx][..., 1:])

    @staticmethod
    def prepare(
        file_path: str,
        text_to_token_ids: Callable[[str], List[int]],
        # padding_token_id: int,
        max_sequence_length: int = 32,
        encoding="utf-8",
        dtype=torch.int,
    ) -> List[torch.Tensor]:
        """Process a text file into list of 2d torch tensors of shape (n_examples, n_tokens).

        Args:
            file_path (str): where the input file is stored
            text_to_token_ids (Callable[[str], List[int]]): a function that can process a line from file and convert it into a list of token ids.
            max_sequence_length (int, optional): make n_grams of sequences longer than this of size max_sequence_length. Defaults to 32.
            encoding (str, optional): the encoding used in the text file. Defaults to "utf-8".
            dtype (_type_, optional): the type of returned torch tensors. check the range of values a type can contain and choose the smallest to save space. Defaults to torch.int.

        Returns:
            List[torch.Tensor]: _description_
        """

        with open(file_path, "r", encoding=encoding) as f:
            lines = f.read().splitlines()[:10000]

        def text_to_examples(text: str) -> List[List[int]]:
            ids = text_to_token_ids(text)
            # ids = ids + [padding_token_id] * max(0, max_sequence_length - len(ids) + 1)
            examples = (
                make_ngrams(ids, max_sequence_length + 1)
                if len(ids) > max_sequence_length + 1
                else [ids]
            )

            return examples  # type:ignore

        buckets = {}
        for examples in map(text_to_examples, tqdm(lines, leave=False)):
            for line in examples:
                if len(line) not in buckets:
                    buckets[len(line)] = []
                buckets[len(line)].append(line)

        data = [
            torch.Tensor(buckets[i]).type(dtype)  # torch.pad
            for i in range(2, max(buckets.keys()))  # type: ignore
            if i in buckets
        ]

        return data


class LM_Trainer:
    """class contains functions to train a language model built with pytorch.
    It is not designed to be generic rather it is specific to the TransformerLanguageModel.
    """

    def __init__(
        self,
        model: TransformerLanguageModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        epochs: int = 10,
        learning_rate: float = 1e-3,
        lr_lambda: Callable[
            [int, float, float], float
        ] = lambda epoch, base_lr, last_lr: base_lr,
        lr_update_step_count: Optional[int] = None,
        optimizer="adamw",
        seed: int = 0,
        model_output_renderer: Optional[
            Callable[[TransformerLanguageModel], str]
        ] = None,
        epoch_unfreeze_map: Optional[Dict[int, List[str]]] = None,
        class_weights: Optional[torch.Tensor] = None,
        max_gradient_norm: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = (
            torch.optim.AdamW(model.parameters(), lr=learning_rate)
            if optimizer == "adamw"
            else torch.optim.Adam(model.parameters(), lr=learning_rate)
        )
        self.scheduler = FullyLambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.seed = seed
        self.model_output_renderer = model_output_renderer

        self.lr_update_step_count = lr_update_step_count
        self.epoch_unfreeze_map = epoch_unfreeze_map
        self.class_weights = class_weights
        self.max_gradient_norm = max_gradient_norm

    def train(self, input_sequences, outputs) -> Dict[str, float]:
        """Training loop. calculates loss on non-padding tokens.
        Multiples class weights with the loss to scale it for imbalanced token distributions.
        Clips gradients with norm > max_gradient_norm to avoid exploding gradients.

        Args:
            input_sequences (_type_): batch of sequences
            outputs (_type_): batch of target sequences in which each position contains target token for the input sequence upto that position.

        Returns:
            Dict[str, float]: the metrics tracked e.g. loss
        """
        history = {}
        # inference
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model.forward(input_sequences)

        # reshape according to loss function's requirement
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = outputs.reshape(B * T)

        # calculate loss
        not_padding_mask = targets != self.model.padding_token_id
        loss = torch.nn.functional.cross_entropy(
            logits[not_padding_mask], targets[not_padding_mask], reduction="none"
        )
        history["loss"] = loss.mean().item()

        # data
        if self.class_weights is not None:
            loss_multiplier = self.class_weights[targets[not_padding_mask]]
            loss_multiplier -= loss_multiplier.mean()
            loss_multiplier /= loss_multiplier.std()
            loss_multiplier -= loss_multiplier.min() - 1
            loss *= loss_multiplier
            history["scaled_loss"] = loss.mean().item()

        # back-propagation
        loss.mean().backward()

        if self.max_gradient_norm:
            torch.nn.utils.clip_grad_norm_(  # type:ignore
                self.model.parameters(), self.max_gradient_norm
            )

        self.optimizer.step()

        return history

    def run(
        self,
        train_batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        validation_batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        early_stop: bool = False,
        checkpoint_dir: str = "",
        checkpoint_step_count: int = 1000,
        model_output_step_count: int = 100,
        start_epoch_number: int = 0,
    ) -> Dict[str, List[float]]:
        """Run the training/validation loop and generate output & checkpoints.

        Returns:
            Dict[str, List[float]]: the tracked metrics
        """

        history = dict(train_loss=[], validation_loss=[])
        model_output = ""
        torch.manual_seed(self.seed)
        for epoch in range(start_epoch_number, start_epoch_number + self.epochs):
            # configure model
            set_layers_trainability_(
                self.model,
                layers_to_unfreeze=(self.epoch_unfreeze_map or {}).get(epoch, []),
            )

            # training loop
            start_time = time()
            train_losses = []
            for i, (input_sequences, outputs) in enumerate(train_batches):
                # move batch to right device
                input_sequences = input_sequences.to(self.device)
                outputs = outputs.to(self.device)

                # forward & back propagation
                _history = self.train(input_sequences, outputs)

                # verbose
                train_losses.append(_history["loss"])
                if i % model_output_step_count == 0:
                    with torch.no_grad():
                        self.model.eval()
                        model_output = (
                            f" | {self.model_output_renderer(self.model)}"
                            if self.model_output_renderer
                            else ""
                        )
                        self.model.train()
                remaining_seconds = (
                    (len(train_batches) - i - 1) * (time() - start_time) / (i + 1)  # type: ignore
                )
                print(
                    "\r"
                    f"epoch: {epoch} | "
                    f"train loss: {train_losses[-1]:.4f} | "
                    f"batches: {i+1}/{len(train_batches)} {(i+1)/len(train_batches):.1%} | "  # type: ignore
                    f"ETA: {remaining_seconds/(60 if remaining_seconds < 60**2 else 60 ** 2):.1f} {'min' if remaining_seconds < 60**2 else 'hr'} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.8f}" + model_output,
                    end="",
                )

                # save model
                if i % checkpoint_step_count == 0:
                    self.checkpoint(
                        checkpoint_dir,
                        losses=train_losses[-checkpoint_step_count:],
                        epoch=epoch,
                        steps_fraction=i / len(train_batches),  # type: ignore
                    )

                # Learning rate update
                if (not self.lr_update_step_count and i == len(train_batches) - 1) or (  # type: ignore
                    self.lr_update_step_count
                    and i % self.lr_update_step_count == self.lr_update_step_count - 1
                ):
                    self.scheduler.step()

            # mean_train_loss = np.mean(train_losses)
            mean_train_loss = self._average(train_losses)
            history["train_loss"].extend(train_losses)

            # validation loop
            start_time = time()
            val_losses = []
            with torch.no_grad():
                for i, (input_sequences, outputs) in enumerate(validation_batches):
                    # move batch to right device
                    input_sequences = input_sequences.to(self.device)
                    outputs = outputs.to(self.device)

                    # forward propagation
                    _history = self.validate(input_sequences, outputs)

                    # verbose
                    val_losses.append(_history["loss"])
                    print(
                        "\r"
                        f"epoch: {epoch} | "
                        f"train loss: {np.mean(train_losses[-1]):.4f} | "
                        f"val loss: {val_losses[-1]:.4f} | "
                        f"batches: {i+1}/{len(validation_batches)}= {(i+1)/len(validation_batches):.1%} | "  # type: ignore
                        f"ETA: {(len(validation_batches)-i-1)*(time()-start_time)/(i+1)/60:.1f} min",  # type: ignore
                        end="",
                    )
                mean_val_loss = np.mean(val_losses)
                history["validation_loss"].extend(val_losses)
                self.model.training_history["val_epoch"].append(epoch + 1)
                self.model.training_history["val_loss"].append(mean_val_loss)

            print(
                "\r"
                f"epoch: {epoch} | "
                f"train loss: {mean_train_loss:.4f}, "
                f"std: {np.std(train_losses):.4f} | "
                f"val loss: {mean_val_loss:.4f}, "
                f"std: {np.std(val_losses):.4f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.1e}"
            )
            if self.model_output_renderer:
                print("model output:", self.model_output_renderer(self.model))

            if early_stop:
                if mean_train_loss < mean_val_loss * 0.95:
                    print("Early stopping")
                    break

        return history

    def validate(
        self, input_sequences: torch.Tensor, outputs: torch.Tensor
    ) -> Dict[str, float]:
        """the validation loop. infers the model on validation data without gradients or back propagation and calculates metrics.

        Args:
            input_sequences (torch.Tensor): batch of input tokens
            outputs (torch.Tensor): batch of target sequences

        Returns:
            Dict[str, float]: the tracked metrics
        """
        history = {}
        with torch.no_grad():
            # inference
            self.model.eval()
            logits = self.model.forward(input_sequences)

            # reshape according to loss function's requirement
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = outputs.reshape(B * T)

            # calculate loss
            not_padding_mask = targets != self.model.padding_token_id
            loss = torch.nn.functional.cross_entropy(
                logits[not_padding_mask], targets[not_padding_mask], reduction="none"
            )
            history["loss"] = loss.mean().item()

            # handle data imbalance
            if self.class_weights is not None:
                loss_multiplier = self.class_weights[targets[not_padding_mask]]
                loss_multiplier -= loss_multiplier.mean()
                loss_multiplier /= loss_multiplier.std()
                loss_multiplier -= loss_multiplier.min() - 1
                loss *= loss_multiplier
                # loss *= self.class_weights[targets[not_padding_mask]]
                history["scaled_loss"] = loss.mean().item()

        return history

    def checkpoint(self, checkpoint_dir: str, losses, epoch, steps_fraction) -> None:
        """save metrics in model and save model to disk."""

        self.model.training_history["train_epoch"].append(epoch + steps_fraction)
        avg_loss = self._average(losses)
        self.model.training_history["train_loss"].append(avg_loss)

        if os.path.isdir(checkpoint_dir):
            previous_checkpoint_paths = glob(
                os.path.join(
                    checkpoint_dir,
                    f"checkpoint_tlm_{self.model.n_parameters/1e6:.1f}M_*loss.pt",
                )
            )
            previous_best_loss = min(
                float(p.split("M_")[-1][:-7])
                for p in previous_checkpoint_paths + ["0M_inf1234567"]
            )
            if avg_loss > previous_best_loss:
                return
            for p in previous_checkpoint_paths:
                os.remove(p)
            self.model.save(
                os.path.join(
                    checkpoint_dir,
                    f"checkpoint_tlm_{self.model.n_parameters/1e6:.1f}M_{avg_loss:.4f}loss.pt",
                )
            )
        else:
            print("checkpoint failed, dir not found")

    def _average(self, values) -> float:
        return np.average(values, weights=np.geomspace(1, 20, len(values)))


__all__ = [
    "LM_Dataset",
    "LM_Trainer",
]


if __name__ == "__main__":
    # TODO: write training script (for now, see notebooks repo)
    # https://github.com/sign-language-translator/notebooks/blob/main/model_training/transformer_lm_training.ipynb
    pass
