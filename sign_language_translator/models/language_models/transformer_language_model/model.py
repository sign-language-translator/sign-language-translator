"""
This module defines a Transformer-based Language Model that can be used for text generation and language modeling.

Class:
    TransformerLanguageModel: A class that implements a Transformer-based Language Model.
"""

from __future__ import annotations

from copy import deepcopy
from os.path import exists
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from sign_language_translator.models.language_models.abstract_language_model import (
    LanguageModel,
)
from sign_language_translator.models.language_models.transformer_language_model.layers import (
    DecoderBlock,
)
from sign_language_translator.models.utils import top_p_top_k_indexes
from sign_language_translator.utils import sample_one_index


class TransformerLanguageModel(LanguageModel, torch.nn.Module):
    """Transformer-based language model for text generation.

    This class implements a Transformer-based language model for text generation tasks.
    It takes in a sequence of token IDs and generates the next token in the sequence.
    The model consists of two embedding layers, multiple decoder blocks, and a language modeling head.

    Attributes:
        - token_embedding (torch.nn.Embedding): The embedding layer for token IDs.
        - position_embedding (torch.nn.Embedding): The embedding layer for positional IDs.
        - decoder_blocks (torch.nn.Sequential): The sequence of decoder blocks.
        - final_layer_norm (torch.nn.LayerNorm): The layer normalization for the final output.
        - language_modeling_head (torch.nn.Linear): The linear layer for language modeling.
        - n_parameters (int): The total number of parameters in the model.
        - device (str): The device to run the model on.
        - training_history (Dict[str, Any]): The training history of the model such as loss and other metrics.

    Methods:
        - forward(token_ids: torch.Tensor) -> torch.Tensor: Performs a forward pass through the model.
        - next(self, context: Iterable) -> Tuple[Any, float]: generates the next token and its probability.
        - next_all(self, context: Iterable) -> Tuple[List[Any], List[float]]: returns all next tokens and their probabilities.
        - load(model_path: str) -> TransformerLanguageModel: (static_method) Deserializes the model from a pt file.
        - save(self, model_path: str, device: str | Torch.device): Serializes the model to a pt file.
        - get_model_state() -> Dict[str, Any]: Returns the model state consisting of constructor arguments and pytorch state_dict.
        - tokens_to_ids(tokens: Iterable[str]) -> List[int]: Converts tokens to IDs.
        - ids_to_tokens(ids: Iterable[int] | torch.Tensor) -> List[str]: Converts IDs to tokens.
    """

    def __init__(
        self,
        token_to_id: Dict[str, int],
        vocab_size: int,
        unknown_token="<unk>",
        padding_token="<pad>",
        start_of_sequence_token="<sos>",
        window_size: int = 64,
        embed_size: int = 768,
        hidden_size: int = 768 * 4,
        n_heads: int = 6,
        n_blocks: int = 6,
        dropout: float = 0.25,
        activation="gelu",
        device="cuda" if torch.cuda.is_available() else "cpu",
        sampling_temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.9,
        name: Optional[str] = None,
        pretrained_token_embeddings: Optional[torch.Tensor] = None,
        randomly_shift_position_embedding_during_training: bool = False,
    ) -> None:
        """Transformer-based language model for text generation.

        Args:
            - token_to_id (Dict[str, int]): A dictionary mapping token strings to their corresponding IDs.
            - vocab_size (int): The size of the vocabulary. Determine model's dimensionality.
            - unknown_token (str): The token representing unknown words.
            - padding_token (str): The token representing padding.
            - start_of_sequence_token (str): The token representing the start of a sequence.
            - window_size (int): The maximum context length of input for positional embeddings.
            - embed_size (int): The size of the token embeddings.
            - hidden_size (int): The size of the hidden layers in the decoder blocks.
            - n_heads (int): The number of self attention heads in each decoder block.
            - n_blocks (int): The number of decoder blocks.
            - dropout (float): The dropout rate for regularization.
            - activation (str): The activation function used in the decoder blocks.
            - device (str): The device to run the model on (default is 'cuda' if available, else 'cpu').
            - sampling_temperature (float): The temperature for sampling from the output distribution. High means more randomness, low is more probable. Defaults to 1.0.
            - top_k (int | None): The number of top-k candidates to consider during sampling.
            - top_p (float | None): The threshold for top cumulative probability during sampling.
            - name (str | None): The name of the model.
            - pretrained_token_embeddings (torch.Tensor | None): Pretrained token embeddings.
            - randomly_shift_position_embedding_during_training (bool): Flag to shift the position ids by a random amount during training. Defaults to False.
        """

        LanguageModel.__init__(self, unknown_token, name)
        torch.nn.Module.__init__(self)

        # tokens
        self.token_to_id = deepcopy(token_to_id)
        self.padding_token = padding_token
        self.unknown_token_id = self.token_to_id[self.unknown_token]
        self.padding_token_id = self.token_to_id[self.padding_token]
        self.start_of_sequence_token = start_of_sequence_token
        self.start_of_sequence_token_id = self.token_to_id[start_of_sequence_token]

        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        # model hyperparameters
        self.embed_size = embed_size
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.activation = activation
        self.hidden_size = hidden_size

        # sampling
        self.sampling_temperature = sampling_temperature
        self._next_all_tokens = [
            self.id_to_token.get(i, self.unknown_token) for i in range(vocab_size)
        ]
        self.top_k = top_k
        self.top_p = top_p

        # learnable parameters
        self.token_embedding = torch.nn.Embedding(
            vocab_size, self.embed_size, padding_idx=self.padding_token_id
        )
        self.position_embedding = torch.nn.Embedding(self.window_size, self.embed_size)
        self.decoder_blocks = torch.nn.Sequential(
            *[
                DecoderBlock(
                    embed_size,
                    hidden_size,
                    n_heads,
                    max_seq_len=self.window_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_layer_norm = torch.nn.LayerNorm(self.embed_size)
        self.language_modeling_head = torch.nn.Linear(self.embed_size, self.vocab_size)
        # self.language_modeling_head = torch.nn.Linear(..., bias=False) # for tied weights

        # model configuration
        self.to(device)

        self.apply(self._initialize_weights)  # gaussian(mean=0, std=0.02)
        # apply special scaled initialization to the residual projections, per GPT-2 paper
        for parameter_name, param in self.named_parameters():
            if parameter_name.endswith("projection.weight"):
                torch.nn.init.normal_(
                    param, mean=0.0, std=0.02 / ((2 * n_blocks) ** 0.5)
                )

        if pretrained_token_embeddings is not None:
            assert (
                pretrained_token_embeddings.shape == self.token_embedding.weight.shape
            )
            self.token_embedding.load_state_dict(
                {"weight": pretrained_token_embeddings}
            )
            self.token_embedding.weight.requires_grad = False

        # weight tying (transfer learning becomes difficult)
        # self.language_modeling_head.weight = self.token_embedding.weight

        # TODO: experiment with weight tying with MLP of decoder
        # self.decoder_blocks[-1].feed_forward.fully_connected_1.weight = self.token_embedding.weight
        # self.decoder_blocks[-1].feed_forward.fully_connected_1.bias = 0
        # self.decoder_blocks[-1].feed_forward.fully_connected_1.bias.requires_grad = False

        # other
        self.n_parameters = sum(p.numel() for p in self.parameters())
        self.randomly_shift_position_embedding_during_training = (
            randomly_shift_position_embedding_during_training
        )
        self.training_history = {
            "train_epoch": [],
            "train_loss": [],
            "val_epoch": [],
            "val_loss": [],
        }
        self.eval()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        This method embeds the token_ids into vectors. It also embeds their positions into vectors.
        Depending upon the training & randomly_shift flags, it may shift sequences' position by a random amount.
        The embeddings are added together and passed to transformer decoder block containing causal multi-head self attention.
        The output is passed through LayerNorm and finally to a language-modeling-head which converts the vectors into logits for each token.

        Args:
            token_ids (torch.Tensor): Tensor containing the token IDs. Shape is ([batch,] time).

        Returns:
            torch.Tensor: Tensor containing the logits. Shape is ([batch,] time, vocab_size).
        """

        # token_ids.shape = ([batch,] time)
        token_ids = token_ids[..., -self.window_size :]
        position_ids = self._make_position_ids(token_ids)

        # embed the tokens
        token_embeddings = self.token_embedding(token_ids)  # (batch, time, embed_size)
        position_embeddings = self.position_embedding(
            position_ids
        )  # (time, embed_size)
        x = token_embeddings + position_embeddings

        # apply QKV & FF layers
        x = self.decoder_blocks(x)  # (batch, time, embed_size)

        # normalization
        x = self.final_layer_norm(x)  # (batch, time, embed_size)

        # TODO: predict target token's embedding & calculate loss using MSE
        # x = self.embedding_prediction(x)
        # TODO: add separately trained MLP as language_modeling_head

        # compute logits
        logits = self.language_modeling_head(x)  # ([batch,] time, vocab_size)

        return logits

    def next(self, context: Iterable) -> Tuple[Any, float]:
        next_tokens, probabilities = self.next_all(context)

        filtered_indexes = (
            np.arange(len(probabilities))
            if len(context) == 1 and context[0] == self.start_of_sequence_token  # type: ignore
            else top_p_top_k_indexes(probabilities, top_k=self.top_k, top_p=self.top_p)
        )

        index_in_filtered_indexes = sample_one_index(
            np.array(probabilities)[filtered_indexes].tolist(),  # type: ignore
            temperature=self.sampling_temperature,
        )

        next_token_index = filtered_indexes[index_in_filtered_indexes]
        return next_tokens[next_token_index], probabilities[next_token_index]  # type: ignore

    def next_all(self, context) -> Tuple[List[Any], List[float]]:
        context = context[-self.window_size :]  # type: ignore
        last_input_token_index = len(context) - 1

        ids = self.tokens_to_ids(context)  # type: ignore
        ids = torch.Tensor(ids).type(torch.long).to(self.device)

        with torch.no_grad():
            logits = self.forward(ids)
            logits = logits[..., last_input_token_index, :]
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        return (
            self._next_all_tokens,
            probabilities.tolist(),  # .squeeze()
        )

    @staticmethod
    def load(
        model_path, device="cuda" if torch.cuda.is_available() else "cpu"
    ) -> TransformerLanguageModel:
        """
        Loads a TransformerLanguageModel from a given model path.

        Args:
            model_path (str): The path to the saved model file.
            device (str, optional): The device to load the model on. Defaults to "cuda" if a CUDA device is available, else "cpu".

        Returns:
            TransformerLanguageModel: The loaded TransformerLanguageModel object.
        """

        model_state_dict: Dict[str, Any] = torch.load(
            model_path, map_location=torch.device(device)
        )
        model = TransformerLanguageModel(
            **{
                k: v
                for k, v in model_state_dict.items()
                if k not in ["state_dict", "training_history"]
            },
        )
        model.load_state_dict(model_state_dict["state_dict"])
        model.training_history.update(model_state_dict.get("training_history", {}))

        return model

    def save(self, model_path: str, overwrite: bool = False) -> None:
        """
        Save the model to a file.

        Args:
            model_path (str): The path to save the model.
            overwrite (bool, optional): Whether to overwrite an existing file. Defaults to False.

        Raises:
            FileExistsError: If there is already a file at the specified path and overwrite is set to False.
        """

        if exists(model_path) and not overwrite:
            raise FileExistsError(f"there is already a file at {model_path = }")

        torch.save(self.get_model_state(), model_path)

    def get_model_state(self) -> Dict[str, Any]:
        """Returns the current state of the model as a dictionary.

        Returns:
            Dict[str, Any]: A dictionary mapping strings to the class arguments,
                pytorch model's state_dict and other attributes.
        """

        return {
            # torch parameters
            "state_dict": self.state_dict(),
            # tokens
            "token_to_id": self.token_to_id,
            "padding_token": self.padding_token,
            "unknown_token": self.unknown_token,
            "start_of_sequence_token": self.start_of_sequence_token,
            # model hyperparameters
            "vocab_size": self.vocab_size,
            "window_size": self.window_size,
            "embed_size": self.embed_size,
            "hidden_size": self.hidden_size,
            "n_heads": self.n_heads,
            "n_blocks": self.n_blocks,
            "dropout": self.dropout,
            "activation": self.activation,
            # sampling
            "sampling_temperature": self.sampling_temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            # other
            "name": self.name,
            "training_history": self.training_history,
            "randomly_shift_position_embedding_during_training": self.randomly_shift_position_embedding_during_training,
        }

    def tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
        """
        Convert a list of tokens into a list of corresponding token IDs.

        Args:
            tokens (Iterable[str]): A list of tokens.

        Returns:
            List[int]: A list of token IDs. If a token is not found in the token_to_id dictionary, the unknown_token_id is used instead.
        """

        return [self.token_to_id.get(token, self.unknown_token_id) for token in tokens]

    def ids_to_tokens(self, ids: Union[Iterable[int], torch.Tensor]):
        """
        Convert a sequence of token IDs to tokens.

        Args:
            ids (Iterable[int] | torch.Tensor): An iterable of token IDs.

        Returns:
            List[str]: A list of tokens corresponding to the input IDs.
        """

        if isinstance(ids, int):
            ids = [ids]
        elif isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        return [self.id_to_token.get(i, self.unknown_token) for i in ids]

    def to(self, device, *args, **kwargs):
        self.device = device
        super().to(device, *args, **kwargs)

        return self

    def _initialize_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_position_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Generates position IDs for the given token IDs (normally from 0 to seq_len-1).
        In case both model's training and randomly_shift_position_embedding_during_training attributes are True,
        the position IDs are shifted ahead ahead by a random amount according to the window_size
        except when the sequence starts with start_of_sequence token.

        Args:
            token_ids (torch.Tensor): The input token IDs.

        Returns:
            torch.Tensor: The generated position IDs.

        """
        seq_len = token_ids.shape[-1]
        position_ids = torch.arange(seq_len, device=self.device)

        # simulate that a substring can appear at any position inside the window
        # assuming train batches are bucketed by length (no extra padding in batch)
        # before: [0,1,2,3] -> [[0,1,2,3] _, _, _, ...] only initial positions get trained
        # after : [0,1,2,3] -> [..., _, _, _, [4,5,6,7] _, _, _, ...] all positions get trained
        if self.training and self.randomly_shift_position_embedding_during_training:
            batch_size = None if token_ids.dim() == 1 else token_ids.shape[0]
            shifts = torch.randint(
                0,
                self.window_size - seq_len + 1,
                (batch_size, 1) if batch_size else (1,),
                device=self.device,
            )
            # dont shift examples that start with SOS token
            shifts[token_ids[..., :1] == self.start_of_sequence_token_id] = 0

            position_ids = position_ids + shifts

        return position_ids

    def __str__(self) -> str:
        return f"Transformer LM: {super().__str__()}, window={self.window_size}, params={self.n_parameters}"


__all__ = [
    "TransformerLanguageModel",
]
