from abc import ABC, abstractmethod

import torch


class TextEmbeddingModel(ABC):
    """
    Abstract class for text embedding models.

    Methods:
        embed(text: str) -> torch.Tensor: Embeds text into a vector.
    """

    @abstractmethod
    def embed(self, text: str) -> torch.Tensor:
        """
        Embeds text into a vector.

        Args:
            text (str): Text to embed.

        Returns:
            torch.Tensor: A vector representation of a text.
        """

    # load and save methods
