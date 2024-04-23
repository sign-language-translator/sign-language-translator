from __future__ import annotations

__all__ = [
    "Sign",
]

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional, Union

import torch
from numpy.typing import NDArray


class Sign(ABC):
    def __init__(self, sign, *args, **kwargs) -> None:
        """initialize the sign"""

    @staticmethod
    @abstractmethod
    def name() -> str:
        """the string code of the sign format"""

    @abstractmethod
    def show(self, **kwargs) -> None:
        """display the sign"""

    @abstractmethod
    def _from_path(self, path: str):
        """load the sign from a path"""

    @abstractmethod
    def _from_data(self, sign_data):
        """load the sign from an object"""

    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs) -> Sign:
        """read the sign from a path"""

    @classmethod
    @abstractmethod
    def load_asset(
        cls, label: str, archive_name: Optional[str] = None, **kwargs
    ) -> Sign:
        """load a sign asset from a dataset"""

    @staticmethod
    @abstractmethod
    def concatenate(objects: Iterable[Sign]) -> Sign:
        """concatenate multiple signs in time dimension"""

    @abstractmethod
    def numpy(self, *args, **kwargs) -> NDArray:
        """return the sign as a numpy array"""

    @abstractmethod
    def torch(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Union[torch.device, str, None] = None,
    ) -> torch.Tensor:
        """return the sign as a torch tensor"""

    @abstractmethod
    def transform(self, transformation: Callable):
        """apply some transformation to the sign to change its appearance"""

    @abstractmethod
    def save(self, path: str, **kwargs) -> None:
        """save the sign to a path"""
