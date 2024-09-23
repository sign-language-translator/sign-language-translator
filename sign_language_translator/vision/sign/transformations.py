"""
sign.transformations.py
=======================

This module contains the base class for all sign transformations available for data augmentation.
"""

__all__ = [
    "BaseSignTransformation",
]

from abc import ABC, abstractmethod


class BaseSignTransformation(ABC):
    """Base class for sign transformations"""

    @staticmethod
    @abstractmethod
    def name() -> str:
        """The string name of the transformation"""

    @abstractmethod
    def transform(self, sign, *args, **kwargs):
        """Apply the transformation to a sign"""

    def __call__(self, sign, *args, **kwargs):
        return self.transform(sign, *args, **kwargs)
