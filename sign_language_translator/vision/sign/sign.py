from __future__ import annotations

__all__ = [
    "Sign",
]

from abc import ABC, abstractmethod
from typing import Callable, Iterable

from numpy.typing import NDArray
from torch import Tensor


class Sign(ABC):
    def __init__(self, sign, *args, **kwargs) -> None:
        """initialize the sign"""

    @staticmethod
    @abstractmethod
    def name() -> str:
        """return the name of the sign format"""

    @abstractmethod
    def show(self, **kwargs) -> None:
        """display the sign"""

    @abstractmethod
    def _from_path(self, path: str):
        """load the sign from a path"""

    # _from_dataset

    @abstractmethod
    def _from_data(self, sign_data):
        """load the sign from an object"""

    @staticmethod
    @abstractmethod
    def concatenate(objects: Iterable[Sign]) -> Sign:
        """concatenate multiple signs"""

    @abstractmethod
    def numpy(self, *args, **kwargs) -> NDArray:
        """return the sign as a numpy array"""

    @abstractmethod
    def torch(self, *args, **kwargs) -> Tensor:
        """return the sign as a torch tensor"""

    @abstractmethod
    def transform(self, transformation: Callable):
        """apply some transformation to the sign to change its appearance"""

    @abstractmethod
    def save(self, path: str, *args, **kwargs) -> None:
        """save the sign to a path"""
