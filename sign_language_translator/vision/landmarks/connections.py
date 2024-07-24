__all__ = [
    "BaseConnections",
    "MediapipeConnections",
    "get_connections",
]

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple

from sign_language_translator.config.colors import Colors


class Connection:
    """
    Represents a connection between landmarks.

    Attributes:
        label (str): The label of the connection.
        indexes (Sequence[int]): The indexes of the landmarks connected by this connection.
        color (Tuple[int, int, int]): The color of the connection.
    """

    def __init__(
        self, label: str, indexes: Sequence[int], color: Tuple[int, int, int]
    ) -> None:
        self.label = label
        self.indexes = indexes
        self.color = color

    def __repr__(self) -> str:
        return f"Connection(label={self.label}, indexes={self.indexes}, color={self.color})"


class BaseConnections(ABC):
    """A class containing information about the connections between landmarks generated from various models."""

    def __init__(self) -> None:
        self._line_indexes = []
        self._line_colors = []
        self._line_labels = []
        self._matplot3d_config = {}

    @staticmethod
    @abstractmethod
    def name() -> str:
        """The name of the connection format"""

    @property
    @abstractmethod
    def connections(self) -> List[Connection]:
        """indexes of landmarks that are connected"""

    @property
    @abstractmethod
    def n_landmarks(self) -> int:
        """Total number of landmarks"""

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Total number of features per landmark"""

    @property
    def line_indexes(self) -> List[Sequence[int]]:
        """list of sequence of indexes that are connected with single line"""
        return self._line_indexes

    @property
    def line_colors(self) -> List[Tuple[int, int, int]]:
        """list of colors for each connection"""
        return self._line_colors

    @property
    def line_labels(self) -> List[str]:
        """list of labels for each connection"""
        return self._line_labels

    @property
    def matplot3d_config(self) -> Dict[str, Any]:
        """Configuration arguments for 3D matplotlib plot"""
        return self._matplot3d_config

    def _parse_connections(self) -> None:
        """extract line indexes, colors and labels from connections into separate lists"""
        self._line_indexes = [connection.indexes for connection in self.connections]
        self._line_colors = [connection.color for connection in self.connections]
        self._line_labels = [connection.label for connection in self.connections]

    # left hand range
    # right hand range
    # left hand connections
    # right hand connections
    # torso (for rectification)


class MediapipeConnections(BaseConnections):
    """Represents the connections for the Mediapipe landmark model."""

    def __init__(self) -> None:
        super().__init__()

        self.scatter_color = Colors.BLACK
        self.default_color = Colors.BLACK
        self.default_color_2 = Colors.DARK_GREY
        self._n_landmarks = 75  # 33 + 21 * 2
        self._n_features = 5

        # Initialize all the connections
        self.face = Connection(
            "face", [8, 6, 5, 4, 0, 10, 9, 0, 1, 2, 3, 7], Colors.BLACK
        )
        self.torso = Connection("torso", [12, 11, 23, 24, 12], Colors.BLACK)
        self.right_arm = Connection(
            "right_arm", [12, 14, 16, 18, 20, 16, 22], Colors.MAGENTA
        )
        self.left_arm = Connection(
            "left_arm", [11, 13, 15, 17, 19, 15, 21], Colors.CYAN
        )
        self.right_leg = Connection("right_leg", [24, 26, 28, 30, 32, 28], Colors.BLACK)
        self.left_leg = Connection("left_leg", [23, 25, 27, 29, 31, 27], Colors.BLACK)
        self.left_palm = Connection(
            "left_palm", [38, 42, 46, 50, 33, 38, 34], Colors.BLACK
        )
        self.left_thumb = Connection("left_thumb", [33, 34, 35, 36, 37], Colors.GREEN)
        self.left_index = Connection("left_index", [38, 39, 40, 41], Colors.DARK_GREY)
        self.left_middle = Connection("left_middle", [42, 43, 44, 45], Colors.BLACK)
        self.left_ring = Connection("left_ring", [46, 47, 48, 49], Colors.BLACK)
        self.left_pinky = Connection("left_pinky", [50, 51, 52, 53], Colors.BLUE)
        self.right_palm = Connection(
            "right_palm", [59, 63, 67, 71, 54, 59, 55], Colors.BLACK
        )
        self.right_thumb = Connection(
            "right_thumb", [54, 55, 56, 57, 58], Colors.ORANGE
        )
        self.right_index = Connection("right_index", [59, 60, 61, 62], Colors.GREY)
        self.right_middle = Connection("right_middle", [63, 64, 65, 66], Colors.BLACK)
        self.right_ring = Connection("right_ring", [67, 68, 69, 70], Colors.BLACK)
        self.right_pinky = Connection("right_pinky", [71, 72, 73, 74], Colors.HOT_PINK)

        # Add all the connections to the list property
        self._connections = [
            self.face,
            self.torso,
            self.right_arm,
            self.left_arm,
            self.right_leg,
            self.left_leg,
            self.left_palm,
            self.left_thumb,
            self.left_index,
            self.left_middle,
            self.left_ring,
            self.left_pinky,
            self.right_palm,
            self.right_thumb,
            self.right_index,
            self.right_middle,
            self.right_ring,
            self.right_pinky,
        ]
        self._parse_connections()

        # Set the configuration for Matplotlib 3D visualization
        self._matplot3d_config = {
            "line_indexes": self.line_indexes,
            "line_colors": [tuple(v / 255 for v in c) for c in self.line_colors],
            "line_labels": self.line_labels,
            "invert_y": True,
            "invert_z": True,
            "vertical_axis": "y",
            "scatter_color": tuple(v / 255 for v in self.scatter_color),
        }

    @staticmethod
    def name() -> str:
        return "mediapipe"

    @property
    def connections(self) -> List[Connection]:
        return self._connections

    @property
    def n_landmarks(self) -> int:
        return self._n_landmarks

    @property
    def n_features(self) -> int:
        return self._n_features


def get_connections(connections: str) -> BaseConnections:
    """Create a connections object based on the given string

    Args:
        connections (str): The name of the connections format to use.

    Returns:
        BaseConnections: The connections object.

    Raises:
        ValueError: If the connections format is not recognized.
    """

    if connections in ("mediapipe-world", "mediapipe-image"):
        return MediapipeConnections()

    raise ValueError(f"Unknown connections format: {connections}")
