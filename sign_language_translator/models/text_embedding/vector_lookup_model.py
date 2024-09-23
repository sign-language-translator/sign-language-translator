"""
Module for VectorLookupModel, a class extending TextEmbeddingModel,
which finds the pretrained embedding of text via hash tables.

Classes:
    - VectorLookupModel: A text embedding model that maps tokens to vectors.
"""

from collections import Counter
from os.path import basename
from typing import Callable, Iterable, List, Optional, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
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

    Example:

    ..code-block:: python

        from sign_language_translator.models import VectorLookupModel
        import torch

        tokens = ["example", "text"]
        vectors = torch.tensor([[1, 2, 3], [4, 5, 6]])
        model = VectorLookupModel(tokens, vectors)

        embedding = model.embed("example text")  # [2.5, 3.5, 4.5]

        model.update(["hello"], torch.tensor([[7, 8, 9]]))

        model.save("model.pt")
        loaded_model = VectorLookupModel.load("model.pt")
    """

    def __init__(
        self,
        tokens: List[str],
        vectors: torch.Tensor,
        alignment_matrix: Optional[torch.Tensor] = None,
        description: str = "",
    ):
        """
        Initializes the VectorLookupModel.

        Args:
            tokens (List[str]): A list of tokens.
            vectors (torch.Tensor): A (2D) tensor of vectors in the same order as `tokens`.
        """

        self.__validate_init_args(tokens, vectors, alignment_matrix)
        self.index_to_token = tokens
        self.known_tokens = frozenset(tokens)
        self.token_to_index = {token: index for index, token in enumerate(tokens)}
        self.vectors = vectors
        self.alignment_matrix = alignment_matrix
        self.description = description

        self._normalized_vectors = None
        self._tokens_array = None

    def update(self, tokens: List[str], vectors: torch.Tensor) -> None:
        """
        Update the vector lookup model with new tokens and their corresponding vectors.

        Args:
            tokens (List[str]): The list of new tokens to be added or updated.
            vectors (torch.Tensor): The tensor of corresponding vectors for the new tokens.
            alignment_matrix (Optional[torch.Tensor], optional): A 2D Tensor to transform the final vectors. (e.g. some orthogonal matrix can be used to align the word vector to an embedding for some other language or model). Defaults to None.
            description (str, optional): A description of the model. Defaults to "".

        Raises:
            ValueError: If the dimensions of the new vectors do not match the dimensions of the existing vectors.
        """

        # validation
        self.__validate_init_args(tokens, vectors, None)
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

        if self._tokens_array is not None:
            self._tokens_array = np.concatenate(
                [self._tokens_array, np.array(new_tokens)]
            )
        if self._normalized_vectors is not None:
            _norms = vectors[remaining_mask].norm(dim=1, keepdim=True)
            _norms[_norms == 0] = 1
            self._normalized_vectors = torch.cat(
                [self._normalized_vectors, vectors[remaining_mask] / _norms]
            )

    # =============== #
    #    embedding    #
    # =============== #

    def embed(
        self,
        text: str,
        pre_normalize=False,
        post_normalize=False,
        align=False,
        tokenizer: Callable[[str], Iterable[str]] = lambda x: x.split(),
    ) -> torch.Tensor:
        """
        Embeds the given text into a vector representation by lookup or averaging pre-computed embeddings.

        Args:
            text (str): The input text to be embedded, (can be in the model vocabulary or be a string of tokens from the model dictionary). If unknown, returns a zero vector.
            pre_normalize (bool, optional): Whether to normalize the vectors of tokens in the text before averaging. Defaults to False.
            post_normalize (bool, optional): Whether to normalize the vector after embedding. Defaults to False.
            align (bool, optional): Whether to transform the final vector using the alignment matrix. Defaults to False.
            tokenizer (Callable[[str], Iterable[str]], optional): A callable function to tokenize the text. Only used if the text is not present in the model vocabulary. Defaults to splitting on whitespace.

        Returns:
            torch.Tensor: The embedded vector representation of the input text.
        """
        # TODO: Handle batches

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

        if align and self.alignment_matrix is not None:
            vector = vector @ self.alignment_matrix

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

    # ============ #
    #    Search    #
    # ============ #

    @property
    def normalized_vectors(self):
        if self._normalized_vectors is None:
            _norms = self.vectors.norm(dim=1, keepdim=True)
            _norms[_norms == 0] = 1
            self._normalized_vectors = self.vectors / _norms

        return self._normalized_vectors

    @property
    def tokens_array(self):
        if self._tokens_array is None:
            self._tokens_array = np.array(self.index_to_token)
        return self._tokens_array

    def similar(
        self, vector: torch.Tensor, k: int = 1
    ) -> Tuple[List[str], List[float]]:
        """
        Find the k most similar tokens to the given vector.

        Args:
            vector (torch.Tensor): The 1D vector for which to find similar tokens.
            k (int, optional): The number of similar tokens to return. Defaults to 1.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing the k most similar tokens and their corresponding cosine similarities.
        """

        # normalize the query vector
        _norm = vector.norm(keepdim=True)
        _norm[_norm == 0] = 1

        # calculate cosine similarities
        similarities = (vector / _norm) @ self.normalized_vectors.T
        top_k_similarities, top_k_indexes = similarities.topk(k)

        # return the top k similar tokens and their similarities
        return (
            self.tokens_array[top_k_indexes.numpy()].tolist(),
            top_k_similarities.tolist(),
        )

    # =============== #
    #    load/save    #
    # =============== #

    def save(self, path: str):
        """
        Serialize the tokens list and corresponding vectors to a file.
        If the path ends with '.zip' the file will be compressed.

        Args:
            path (str): The path to save the model file.
        """

        data = {
            "tokens": self.index_to_token,
            "vectors": self.vectors,
            "alignment": self.alignment_matrix,
            "description": self.description,
        }
        if path.endswith(".zip"):
            with ZipFile(path, "w", ZIP_DEFLATED) as archive:
                with archive.open(basename(path).replace(".zip", ".pt"), "w") as f:
                    torch.save(data, f)
        else:
            torch.save(data, path)

    @classmethod
    def load(cls, path: str):
        """
        Load a VectorLookupModel from a saved checkpoint.
        If the path ends with '.zip' the file will be decompressed.

        Args:
            path (str): The path to the saved checkpoint.

        Returns:
            VectorLookupModel: The loaded VectorLookupModel instance.
        """

        if path.endswith(".zip"):
            with ZipFile(path, "r") as archive:
                with archive.open(basename(path).replace(".zip", ".pt"), "r") as f:
                    checkpoint = torch.load(f)
        else:
            checkpoint = torch.load(path)

        return cls(
            checkpoint["tokens"],
            checkpoint["vectors"],
            alignment_matrix=checkpoint.get("alignment", None),
            description=checkpoint.get("description", ""),
        )

    # ============= #
    #    Helpers    #
    # ============= #

    def __validate_init_args(self, tokens: List[str], vectors: torch.Tensor, alignment):
        if len(tokens) != vectors.shape[0]:
            raise ValueError(f"Size mismatch: {len(tokens) = }, {len(vectors) = }.")
        if vectors.shape[1] == 0:
            raise ValueError("Expected at least one dimension in vectors.")
        if repeated := {t: c for t, c in Counter(tokens).items() if c > 1}:
            raise ValueError(f"Tokens must be unique. Repeated tokens: {repeated}")
        if alignment is not None:
            if alignment.shape[0] != vectors.shape[1]:
                raise ValueError(f"({alignment.shape[0]= }) != ({vectors.shape[1]= })")

    def __len__(self):
        return len(self.index_to_token)
