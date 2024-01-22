"""
Module for VectorLookupModel, a class extending TextEmbeddingModel,
which finds the pretrained embedding of text via hash tables.

Classes:
    - VectorLookupModel: A text embedding model that maps tokens to vectors.
"""

from collections import Counter
from typing import Callable, Iterable, List

import torch

from sign_language_translator.models.text_embedding.text_embedding_model import (
    TextEmbeddingModel,
)


class VectorLookupModel(TextEmbeddingModel):
    """
    VectorLookupModel class extends TextEmbeddingModel to provide text embedding based on pre-defined token vectors.

    Attributes:
        - index_to_token (List[str]): A list containing tokens in the same order as the vectors.
        - known_tokens (frozenset): A frozenset containing unique known tokens.
        - token_to_index (Dict[str, int]): A dictionary mapping tokens to their corresponding indices.
        - vectors (torch.Tensor): A 2D tensor representing the token vectors.

    Methods:
        - update(self, tokens: List[str], vectors: torch.Tensor) -> None: Updates existing tokens & hash-table with new vectors.
        - embed(self, text: str, pre_normalize=False, post_normalize=False,
                tokenizer: Callable[[str], Iterable[str]] = lambda x: x.split()) -> torch.Tensor:
          Returns the pretrained embedding vector for a token or average embedding of sub tokens.
        - __getitem__(self, token: str) -> torch.Tensor: Returns the vector for a specific token.
        - save(self, path: str): Saves the model state (tokens & vectors) to a file.
        - load(cls, path: str): Loads a saved model state (tokens & vectors) from a file.

    Usage:
        ```python
        from sign_language_translator.models import VectorLookupModel
        import torch

        tokens = ["example", "text"]
        vectors = torch.tensor([[1, 2, 3], [4, 5, 6]])
        model = VectorLookupModel(tokens, vectors)

        embedding = model.embed("example text")  # [2.5, 3.5, 4.5]

        model.update(["hello"], torch.tensor([[7, 8, 9]]))

        model.save("model.pt")
        loaded_model = VectorLookupModel.load("model.pt")
        ```
    """

    def __init__(self, tokens: List[str], vectors: torch.Tensor):
        """
        Initializes the VectorLookupModel.

        Args:
            tokens (List[str]): A list of tokens.
            vectors (torch.Tensor): A (2D) tensor of vectors in the same order as `tokens`.
        """

        self.__validate_init_args(tokens, vectors)
        self.index_to_token = tokens
        self.known_tokens = frozenset(tokens)
        self.token_to_index = {token: index for index, token in enumerate(tokens)}
        self.vectors = vectors

    def update(self, tokens: List[str], vectors: torch.Tensor) -> None:
        """
        Update the vector lookup model with new tokens and their corresponding vectors.

        Args:
            tokens (List[str]): The list of new tokens to be added or updated.
            vectors (torch.Tensor): The tensor of corresponding vectors for the new tokens.

        Raises:
            ValueError: If the dimensions of the new vectors do not match the dimensions of the existing vectors.
        """

        # validation
        self.__validate_init_args(tokens, vectors)
        if (stored_dims := self.vectors.shape[1:]) != (new_dims := vectors.shape[1:]):
            raise ValueError(f"Dimension mismatch: {stored_dims = }, {new_dims = }.")

        # update previously existing tokens
        _indexes = [
            (self.token_to_index[t], i)
            for i, t in enumerate(tokens)
            if t in self.known_tokens
        ]
        stored_indexes, new_indexes = list(zip(*_indexes))
        self.vectors[stored_indexes] = vectors[new_indexes]

        # filter out new tokens
        remaining_mask = torch.ones(len(tokens), dtype=torch.bool)
        remaining_mask[new_indexes] = False
        new_tokens = [tokens[i] for i in torch.arange(len(tokens))[remaining_mask]]
        old_len = len(self.index_to_token)

        # append & update dependent properties
        self.vectors = torch.cat([self.vectors, vectors[remaining_mask]], dim=0)
        self.index_to_token = self.index_to_token + new_tokens
        self.known_tokens = frozenset(self.index_to_token)
        self.token_to_index.update({t: i + old_len for i, t in enumerate(new_tokens)})

    # =============== #
    #    embedding    #
    # =============== #

    def embed(
        self,
        text: str,
        pre_normalize=False,
        post_normalize=False,
        tokenizer: Callable[[str], Iterable[str]] = lambda x: x.split(),
    ) -> torch.Tensor:
        """
        Embeds the given text into a vector representation by lookup or averaging pre-computed embeddings.

        Args:
            text (str): The input text to be embedded, (can be in the model vocabulary or be a string of tokens from the model dictionary). If unknown, returns a zero vector.
            pre_normalize (bool, optional): Whether to normalize the vectors of tokens in the text before averaging. Defaults to False.
            post_normalize (bool, optional): Whether to normalize the vector after embedding. Defaults to False.
            tokenizer (Callable[[str], Iterable[str]], optional): A callable function to tokenize the text. Only used if the text is not present in the model vocabulary. Defaults to splitting on whitespace.

        Returns:
            torch.Tensor: The embedded vector representation of the input text.
        """

        # found in the vocabulary
        if text in self.known_tokens:
            vector = self.vectors[self.token_to_index[text]].clone()
            # scale to unit length
            if pre_normalize:
                if norm := vector.norm():
                    vector = vector / norm
        # Average of known tokens
        else:
            # break text into tokens
            tokens = tokenizer(text)
            indexes = [self.token_to_index[t] for t in tokens if t in self.known_tokens]
            if len(indexes) > 0:
                vectors = self.vectors[indexes].clone()
                # scale to unit length
                if pre_normalize:
                    vectors = vectors / vectors.norm(dim=1, keepdim=True)
                # average
                vector = vectors.nanmean(dim=0)
            # unable to embed
            else:
                vector = torch.zeros(self.vectors.shape[1:])

        # scale to unit length
        if post_normalize:
            if norm := vector.norm():
                vector = vector / norm

        return vector

    def __getitem__(self, token: str) -> torch.Tensor:
        """
        Retrieve the embedding vector for a given token. If token is unknown, returns a zero vector.

        Args:
            token (str): The token for which to retrieve the embedding vector.

        Returns:
            torch.Tensor: The embedding vector for the given token.
        """

        return self.embed(token)

    # =============== #
    #    load/save    #
    # =============== #

    def save(self, path: str):
        """
        Serialize the tokens list and corresponding vectors to a file.

        Args:
            path (str): The path to save the model file.
        """

        torch.save({"tokens": self.index_to_token, "vectors": self.vectors}, path)

    @classmethod
    def load(cls, path: str):
        """
        Load a VectorLookupModel from a saved checkpoint.

        Args:
            path (str): The path to the saved checkpoint.

        Returns:
            VectorLookupModel: The loaded VectorLookupModel instance.
        """

        checkpoint = torch.load(path)
        return cls(checkpoint["tokens"], checkpoint["vectors"])

    # ============= #
    #    Helpers    #
    # ============= #

    def __validate_init_args(self, tokens: List[str], vectors: torch.Tensor):
        if len(tokens) != vectors.shape[0]:
            raise ValueError(f"Size mismatch: {len(tokens) = }, {len(vectors) = }.")
        if vectors.shape[1] == 0:
            raise ValueError("Expected at least one dimension in vectors.")
        if repeated := {t: c for t, c in Counter(tokens).items() if c > 1}:
            raise ValueError(f"Tokens must be unique. Repeated tokens: {repeated}")

    def __len__(self):
        return len(self.index_to_token)
