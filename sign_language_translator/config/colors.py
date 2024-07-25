__all__ = [
    "Colors",
]

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np

from sign_language_translator.utils.arrays import linear_interpolation


@dataclass(frozen=True)
class Colors:
    """A class to represent a set of predefined colors and provide utility functions related to colors."""

    BLACK: Tuple[int, int, int] = (0, 0, 0)
    NAVY_BLUE: Tuple[int, int, int] = (0, 0, 255)
    GREEN: Tuple[int, int, int] = (0, 255, 0)
    RED: Tuple[int, int, int] = (255, 0, 0)
    CYAN: Tuple[int, int, int] = (0, 255, 255)
    MAGENTA: Tuple[int, int, int] = (255, 0, 255)
    YELLOW: Tuple[int, int, int] = (255, 255, 0)
    ORANGE: Tuple[int, int, int] = (255, 127, 0)
    PURPLE: Tuple[int, int, int] = (127, 0, 255)
    PINK: Tuple[int, int, int] = (255, 127, 255)
    HOT_PINK: Tuple[int, int, int] = (255, 0, 127)
    BLUE: Tuple[int, int, int] = (0, 127, 255)
    LEAF_GREEN: Tuple[int, int, int] = (127, 255, 0)
    DARK_GREY: Tuple[int, int, int] = (63, 63, 63)
    GREY: Tuple[int, int, int] = (127, 127, 127)
    LIGHT_GREY: Tuple[int, int, int] = (191, 191, 191)

    @staticmethod
    def gradient(
        n: int,
        colors: Sequence[
            Tuple[Union[float, int], Union[float, int], Union[float, int]]
        ] = (HOT_PINK, BLUE, HOT_PINK),
        endpoint=False,
        dtype=int,
    ) -> List[Tuple[float, float, float]]:
        """
        Generate a gradient of colors by linearly interpolating between a sequence of colors.

        Args:
            n (int): The number of colors to generate in the gradient.
            colors (Sequence[Tuple[Union[float, int], Union[float, int], Union[float, int]]], optional): A sequence of RGB tuples to interpolate between. Defaults to (HOT_PINK, BLUE, HOT_PINK).
            endpoint (bool, optional): If True, the last value in the colors sequence is included. Defaults to False.
            dtype (Union[type, str], optional): The desired data type of the output colors. Defaults to int.

        Returns:
            List[Tuple[float, float, float]]: A list of RGB tuples representing the gradient.

        Example:
            >>> Colors.gradient(5, colors = [(128, 0, 0), (0, 0, 128)], endpoint=True)
            [[128, 0, 0], [96, 0, 32], [64, 0, 64], [32, 0, 96], [0, 0, 128]]
        """
        return (
            np.array(
                linear_interpolation(
                    colors, np.linspace(0, len(colors) - 1, n, endpoint=endpoint)
                )
            )
            .astype(dtype)
            .tolist()
        )
