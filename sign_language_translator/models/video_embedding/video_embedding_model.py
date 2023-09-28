"""This module provides an abstract base class for video embedding models that
transform a sequence of video frames into an embedding tensor.

Classes:
    - VideoEmbeddingModel: An abstract base class for video embedding models.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Union

from numpy import uint8
from numpy.typing import NDArray
from torch import Tensor


class VideoEmbeddingModel(ABC):
    """
    Abstract base class for video embedding models.

    This class defines the interface for video embedding models, which transform
    a sequence of video frames into an embedding tensor.

    Attributes:
        None

    Methods:
        embed(frame_sequence, **kwargs): Abstract method to embed a sequence of video frames.

    """

    @abstractmethod
    def embed(
        self, frame_sequence: Iterable[Union[Tensor, NDArray[uint8]]], **kwargs
    ) -> Tensor:
        """Embed a sequence of video frames into an embedding tensor.

        Args:
            frame_sequence (Iterable[Union[Tensor, NDArray[uint8]]]): A sequence of video frames,
                where each frame can be either a Tensor or a numpy array of uint8 values of shape (W, H, C).
            **kwargs: Additional keyword arguments specific to the embedding model.

        Returns:
            Tensor: An embedding tensor representing the sequence of video frames.
        """
