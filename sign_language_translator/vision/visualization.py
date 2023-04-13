"""Toolkit for ploting pose vectors

Bugs:
    - too much white space around the figures
"""

from typing import Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.video.VideoClip import DataVideoClip
from PIL.Image import fromarray as np2pil
import PIL.Image

from ..utils.landmarks_info import LandmarksInfo


def _get_new_plt_fig(
    axis_lims: List[float] = None,
    elev=15,
    azim=25,
    fig_width=10,
    fig_height=10,
    box_aspect=[1.0, 1.0, 1.0],
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Initialize an empty 3D graph

    Args:
        axis_lims (List[float], optional): the upper and lower limits of the 3D plot axis [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper]. Defaults to None.
        elev (int, optional): the elevation angle of view point in degrees. Defaults to 15.
        azim (int, optional): the azimuth angle of view point in degrees. Defaults to 25.
        fig_width (int, optional): the width of the graph in inches. Defaults to 10.
        fig_height (int, optional): the height of the graph in inches. Defaults to 10.
        box_aspect (List[float], optional): the aspect ratio of x, y, z axes

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: an empty matplotlib (fig, ax) pair
    """

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(projection="3d")

    ax.set_box_aspect(box_aspect)
    ax.view_init(elev=elev, azim=azim, vertical_axis="y")

    if axis_lims:
        ax.set_xlim(axis_lims[0], axis_lims[1])
        ax.set_ylim(axis_lims[2], axis_lims[3])
        ax.set_zlim(axis_lims[4], axis_lims[5])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    fig.tight_layout()

    ax.invert_zaxis()
    ax.invert_yaxis()

    return fig, ax


# put landmarks on  3D graph
def plot_landmarks(
    landmarks: np.ndarray,
    connections: List[List[int]] = None,
    fig: matplotlib.figure.Figure = None,
    ax: matplotlib.axes.Axes = None,
    axis_lims: List[float] = None,
    box_aspect=[1.0, 1.0, 1.0],
    elev=15,
    azim=25,
    landmarks_color: Tuple[float, float, float] = LandmarksInfo.Colors.DEFAULT_LANDMARK,
    landmark_size=6.9,
    alpha=1.0,
    return_plot_cache=False,
    fig_width=10,
    fig_height=10,
) -> Union[
    Tuple[
        matplotlib.figure.Figure,
        matplotlib.axes.Axes,
        Dict[str, object],
    ],
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes],
]:
    """Plots a single pose & hands vector in 3D.

    Args:
        landmarks (np.ndarray): A numpy array of dimention (n,3) containing n landmarks (x, y, z).
        connections (List[List[int]], optional): Indices of the landmarks array to be connected by a line. Defaults to None.
        fig (matplotlib.figure.Figure, optional): matplotlib figure. Defaults to None.
        ax (matplotlib.axes.Axes, optional): matplotlib axes. Defaults to None.
        axis_lims (List[float], optional): the upper and lower limits of the 3D plot axis [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper]. Defaults to None.
        box_aspect (List[float], optional): the aspect ratio of x, y, z axes
        elev (int, optional): Elevation angle of view point in degrees. Defaults to 15.
        azim (int, optional): Azimuth angle of view point in degrees. Defaults to 25.
        landmarks_color (Tuple[float,float,float], optional): the color as RGB values [0 to 1] of the dots representing landmarks. Defaults to LandmarksInfo.Colors.DEFAULT_LANDMARK.
        landmark_size (float, optional): Size of the dots drawn. Defaults to 6.9.
        alpha (float, optional): Opacity of the landmark. Defaults to 1.0.
        return_plot_cache (bool, optional): Wether to return the a dict of objects corresponding to lines and dots drawn. Allows the data to be updated. Defaults to False.
        fig_width (int, optional): the width of the graph in inches. Defaults to 10.
        fig_height (int, optional): the height of the graph in inches. Defaults to 10.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes [, Dict[str, object]]] : matplotlib fig, ax [, plots_cache]
    """

    landmarks = LandmarksInfo.Reshaper().fold(landmarks)

    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.numpy()

    # set up a matplotlib plot
    if fig is None or ax is None:
        fig, ax = _get_new_plt_fig(
            axis_lims=axis_lims,
            elev=elev,
            azim=azim,
            box_aspect=box_aspect,
            fig_height=fig_height,
            fig_width=fig_width,
        )

    if connections is None:
        connections = LandmarksInfo.infer_connections(len(landmarks))

    get_connection_color = (
        LandmarksInfo.Colors._get_hand_connection_color
        if connections == LandmarksInfo.HAND_CONNECTIONS
        else LandmarksInfo.Colors._get_pose_connection_color
        if connections == LandmarksInfo.POSE_CONNECTIONS
        else LandmarksInfo.Colors._get_both_hands_connection_color
        if connections == LandmarksInfo.BOTH_HANDS_CONNECTIONS
        else LandmarksInfo.Colors._get_all_connection_color
        if connections == LandmarksInfo.ALL_CONNECTIONS
        else LandmarksInfo.Colors._get_default_connection_color
    )

    def infer_label(color=(0.0, 0.0, 0.0)) -> str:
        return (
            "Left"
            if color == LandmarksInfo.Colors.LEFT
            else "Right"
            if color == LandmarksInfo.Colors.RIGHT
            else "Thumb"
            if color == LandmarksInfo.Colors.THUMB
            else "Pinky"
            if color == LandmarksInfo.Colors.PINKY
            else None
        )

    plot_cache = {"landmarks_shape": landmarks.shape, "connection_lines": dict()}

    # plot landmarks as dots
    plot_cache["scatter3D"] = ax.scatter3D(
        xs=landmarks[:, 0],
        ys=landmarks[:, 1],
        zs=landmarks[:, 2],
        color=landmarks_color,
        s=landmark_size,
        alpha=alpha,
    )

    # draw lines to connect landmarks
    for connection in connections:
        start, end = connection

        color = get_connection_color(start, end)
        label = infer_label(color)

        plot_cache["connection_lines"][(start, end)] = ax.plot3D(
            xs=[landmarks[start, 0], landmarks[end, 0]],
            ys=[landmarks[start, 1], landmarks[end, 1]],
            zs=[landmarks[start, 2], landmarks[end, 2]],
            color=color,
            linewidth=2,
            label=label,
            alpha=alpha,
        )

    # legend redundancy removal
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    if return_plot_cache:
        return fig, ax, plot_cache
    else:
        return fig, ax


def update_plotted_landmarks(
    landmarks: np.ndarray, plot_cache: Dict[str, object]
) -> None:
    """change the data in an existing plot so nothing else has to be drawn again.

    Args:
        landmarks (np.ndarray): A numpy array of dimention (n,3) containing n landmarks (x, y, z).
        plot_cache (Dict[str, object]): Maps strings to lines and dots drawn so they can be moved.
    """

    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.numpy()

    assert (
        landmarks.shape == plot_cache["landmarks_shape"]
    ), "the provided landmarks do not match the shape of the landmarks in the cache"

    plot_cache["scatter3D"]._offsets3d = (
        landmarks[:, 0],
        landmarks[:, 1],
        landmarks[:, 2],
    )

    for (start, end), line in plot_cache["connection_lines"].items():
        line[0].set_data(
            [
                [landmarks[start, 0], landmarks[end, 0]],
                [landmarks[start, 1], landmarks[end, 1]],
            ]
        )
        line[0].set_3d_properties([landmarks[start, 2], landmarks[end, 2]])


def _get_box_coord_ranges(multi_frame_landmarks: np.ndarray) -> List[float]:
    """find the range of values within which all landmarks stay in all frames

    Args:
        multi_frame_landmarks (np.ndarray): an array of dimentions ([n_frames], n_landmarks, n_axes=3)

    Returns:
        List[float]: the range of values in each dimention [x_min, x_max, y_min, y_max, z_min, z_max]
    """

    assert (
        1 < len(multi_frame_landmarks.shape) <= 3
    ), "all_landmarks should be 2 or 3 dimentional"

    return [
        np.min(multi_frame_landmarks[..., 0]),
        np.max(multi_frame_landmarks[..., 0]),
        np.min(multi_frame_landmarks[..., 1]),
        np.max(multi_frame_landmarks[..., 1]),
        np.min(multi_frame_landmarks[..., 2]),
        np.max(multi_frame_landmarks[..., 2]),
    ]


def _get_box(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    full=True,
) -> Tuple[List[List[float]], List[List[int]]]:
    """Create a square or cuboid corners coordinates and connections from range values of axes

    Args:
        x_min (float): the lowest x value
        x_max (float): the highest x value
        y_min (float): the lowest y value
        y_max (float): the highest y value
        z_min (float): the lowest z value
        z_max (float): the highest z value
        full (bool, optional): True means return connections of a cuboid and False means return only a square on y_max side. Defaults to True.

    Returns:
        Tuple[List[List[float]], List[List[int]]]: box corner coordinates, box_connections
    """

    box_coords = np.array(
        [
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max],
        ]
    )

    box_connections = (
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [7, 6],
            [7, 5],
            [7, 3],
            [1, 3],
            [3, 2],
            [2, 6],
            [6, 4],
            [4, 5],
            [5, 1],
        ]
        if full
        else [
            [7, 6],
            [6, 2],
            [2, 3],
            [3, 7],
        ]
    )

    return box_coords, box_connections


def _move_hands_apart(landmarks: np.ndarray, distance=0.2, scale=1.0) -> np.ndarray:
    """updates the hand landmarks such that left hand gets a positive shift in x-dimention and right hand gets a negative shift

    Args:
        landmarks (np.ndarray): an array with 2nd last dimention containing all landmarks of 2 hands and last dimention contains [x, ...] values
        distance (float, optional): how many units to move the hands. Defaults to 0.2.
        scale (float, optional): how much to zoom the hands. Defaults to 1.0.

    Returns:
        np.ndarray: the modified version of the landmarks array
    """

    landmarks *= scale

    if landmarks.shape[-2] == LandmarksInfo.N_BOTH_HANDS_LANDMARKS:

        landmarks[..., : LandmarksInfo.N_HAND_LANDMARKS, 0] += (
            distance / 2
        )  # left  hand
        landmarks[..., LandmarksInfo.N_HAND_LANDMARKS :, 0] -= (
            distance / 2
        )  # right hand

    return landmarks


def _move_hands_to_wrists(landmarks: np.ndarray, scale=1.0) -> np.ndarray:
    """translates the hand landmarks which have origin on their geometric center to the wrists of the body landmarks

    Args:
        landmarks (np.ndarray): an array with 2nd last dimention containing all landmarks of body and 2 hands and last dimention contains [x, ...] values
        scale (float, optional): how much to zoom the hands. Defaults to 1.0.

    Returns:
        np.ndarray: the modified version of the landmarks array
    """

    if landmarks.shape[-2] == LandmarksInfo.N_ALL_LANDMARKS:

        # Left hand
        left_wrist = landmarks[
            ...,
            LandmarksInfo.POSE_LEFT_WRIST_INDEX : LandmarksInfo.POSE_LEFT_WRIST_INDEX
            + 1,
            :,
        ]
        left_hand_landmarks = landmarks[
            ...,
            LandmarksInfo.N_POSE_LANDMARKS : LandmarksInfo.N_POSE_LANDMARKS
            + LandmarksInfo.N_HAND_LANDMARKS,
            :,
        ]
        left_hand_landmarks *= scale
        left_hand_landmarks -= left_hand_landmarks[
            ..., LandmarksInfo.HAND_WRIST_INDEX : LandmarksInfo.HAND_WRIST_INDEX + 1, :
        ]
        left_hand_landmarks += left_wrist

        # Right hand
        right_wrist = landmarks[
            ...,
            LandmarksInfo.POSE_RIGHT_WRIST_INDEX : LandmarksInfo.POSE_RIGHT_WRIST_INDEX
            + 1,
            :,
        ]
        right_hand_landmarks = landmarks[
            ...,
            LandmarksInfo.N_POSE_LANDMARKS
            + LandmarksInfo.N_HAND_LANDMARKS : LandmarksInfo.N_POSE_LANDMARKS
            + 2 * LandmarksInfo.N_HAND_LANDMARKS,
            :,
        ]
        right_hand_landmarks *= scale
        right_hand_landmarks -= right_hand_landmarks[
            ..., LandmarksInfo.HAND_WRIST_INDEX : LandmarksInfo.HAND_WRIST_INDEX + 1, :
        ]
        right_hand_landmarks += right_wrist

    return landmarks


def _move_hands(
    landmarks: np.ndarray,
    connections: List[List[int]],
    hands_location: str,
    scale=1.0,
    distance=0.2,
) -> np.ndarray:
    """move origin centered hands apart or to wrists

    Args:
        landmarks (np.ndarray): A numpy array of dimention ([n_frames], n_landmarks, n_axes=3) containing (x, y, z) coordinates
        hands_location (str): target location of hands "auto", "wrists" or "spaced". None makes no change in landmarks.
        scale (float, optional): how many times should hands be zoomed. Defaults to 1.0.
        distance (float, optional): how much should they move "apart". Defaults to 0.2.

    Raises:
        ValueError: bad value of hands_location argument

    Returns:
        np.ndarray: the modified version of the landmarks array
    """

    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.numpy()

    landmarks = landmarks.copy()

    if connections is None:
        connections = LandmarksInfo.infer_connections(landmarks.shape[-2])

    if connections == LandmarksInfo.BOTH_HANDS_CONNECTIONS:
        if hands_location is None:
            pass
        elif hands_location in ["auto", "spaced"]:
            landmarks = _move_hands_apart(landmarks, scale=scale, distance=distance)
        else:
            raise ValueError(
                'bad value of hands_location argument, should be None or "spaced" or "auto"'
            )

    elif connections == LandmarksInfo.ALL_CONNECTIONS:
        if hands_location is None:
            pass
        elif hands_location in ["auto", "wrists"]:
            landmarks = _move_hands_to_wrists(landmarks, scale=scale)
        else:
            raise ValueError(
                'bad value of hands_location argument, should be None or "wrists" or "auto"'
            )

    return landmarks


def _fig_to_npimage(fig: matplotlib.figure.Figure) -> np.ndarray:
    """convert a matplotlib figure to a numpy array

    Args:
        fig (matplotlib.figure.Figure): A matplotlib figure object

    Returns:
        np.ndarray: a 3 channel RGB image array
    """

    return mplfig_to_npimage(fig)


def landmarks_to_npimage(
    landmarks: np.ndarray,
    connections: List[List[int]] = None,
    landmarks_color: Tuple[float, float, float] = LandmarksInfo.Colors.DEFAULT_LANDMARK,
    axis_lims: List[float] = None,
    elev=15,
    azim=25,
    move_hands=True,
    hands_location="auto",
    hands_scale=1.0,
    hands_distance=0.1,
) -> np.ndarray:
    """draw single frame landmarks onto an image

    Args:
        landmarks (np.ndarray): A numpy array of dimention (n,3) containing n landmarks (x, y, z).
        connections (List[List[int]], optional): Indices of the landmarks array to be connected by a line.. Defaults to None.
        landmarks_color (Tuple[float,float,float], optional): the color as RGB values [0 to 1] of the dots representing landmarks. Defaults to LandmarksInfo.Colors.DEFAULT_LANDMARK.
        axis_lims (List[float], optional): the upper and lower limits of the 3D plot axis [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper].. Defaults to None.
        elev (int, optional): the elevation angle of view point in degrees. Defaults to 15.
        azim (int, optional): Azimuth angle of view point in degrees. Defaults to 25.
        move_hands (bool, optional): wether to move origin centered hands apart or to wrists or not. Defaults to True.
        hands_location (str, optional): where to move hand to. ignored if move_hands is false. Defaults to "auto".
        hands_scale (float, optional): how much should hands be zoomed. Defaults to 1.0.
        hands_distance (float, optional): how much should hands be shifted "apart". Defaults to 0.1.

    Returns:
        np.ndarray: a 3 channel RGB image array containing drawn landmarks
    """

    landmarks = LandmarksInfo.Reshaper().fold(landmarks)

    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.numpy()

    if connections is None:
        connections = LandmarksInfo.infer_connections(len(landmarks))

    if move_hands:
        landmarks = _move_hands(
            landmarks,
            connections,
            hands_location,
            scale=hands_scale,
            distance=hands_distance,
        )

    fig, ax = plot_landmarks(
        landmarks,
        connections,
        axis_lims=axis_lims,
        elev=elev,
        azim=azim,
        landmarks_color=landmarks_color,
    )

    body_ratios = (
        np.ptp(landmarks[:, 2]),
        np.ptp(landmarks[:, 0]),
        np.ptp(landmarks[:, 1]),
    )

    ax.set_box_aspect(body_ratios)

    image = _fig_to_npimage(fig)

    plt.clf()
    plt.close()

    return image


def plot_multi_frame_landmarks(
    multi_frame_landmarks: np.ndarray,
    connections: List[List[int]] = None,
    landmarks_color: Tuple[float, float, float] = LandmarksInfo.Colors.DEFAULT_LANDMARK,
    axis_lims: List[float] = None,
    elev=15,
    azim=25,
    move_hands=True,
    hands_location="auto",
    hands_scale=1.0,
    hands_distance=0.1,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot multiple frames on a single graph with x-shift

    Args:
        multi_frame_landmarks (np.ndarray): an array of dimentions (n_frames, n_landmarks, n_axes=3)
        connections (List[List[int]], optional): Indices of the landmarks array to be connected by a line. Defaults to None.
        landmarks_color (Tuple[float, float, float], optional): the color as RGB values [0 to 1] of the dots representing landmarks. Defaults to LandmarksInfo.Colors.DEFAULT_LANDMARK.
        axis_lims (List[float], optional): the upper and lower limits of the 3D plot axis [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper].. Defaults to None.
        elev (int, optional): the elevation angle of view point in degrees. Defaults to 15.
        azim (int, optional): Azimuth angle of view point in degrees. Defaults to 25.
        move_hands (bool, optional): wether to move origin centered hands apart or to wrists or not. Defaults to True.
        hands_location (str, optional): where to move hand to. ignored if move_hands is false. Defaults to "auto".
        hands_scale (float, optional): how much should hands be zoomed. Defaults to 1.0.
        hands_distance (float, optional): how much should hands be shifted "apart". Defaults to 0.1.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: matplotlib fig, ax pair with larmarks plotted
    """

    multi_frame_landmarks = LandmarksInfo.Reshaper().fold(multi_frame_landmarks)

    if isinstance(multi_frame_landmarks, torch.Tensor):
        multi_frame_landmarks = multi_frame_landmarks.numpy()

    if connections is None:
        connections = LandmarksInfo.infer_connections(multi_frame_landmarks.shape[-2])

    if move_hands:
        multi_frame_landmarks = _move_hands(
            multi_frame_landmarks,
            connections,
            hands_location,
            scale=hands_scale,
            distance=hands_distance,
        )

    x0, x1, y0, y1, z0, z1 = _get_box_coord_ranges(multi_frame_landmarks)
    box_coords, box_connections = _get_box(x0, x1, y0, y1, z0, z1, full=False)

    x_shift = (x1 - x0) * 1.25

    fig, ax = _get_new_plt_fig(
        axis_lims=axis_lims,
        elev=elev,
        azim=azim,
    )  # fig_width = 10*len(multi_frame_landmarks))

    # Make graph wider
    n_frames = len(multi_frame_landmarks)
    body_ratios = (z1 - z0, x_shift * n_frames, y1 - y0)
    ax.set_box_aspect(body_ratios)

    for i, landmarks in enumerate(multi_frame_landmarks.copy()):
        box_coords[:, 0] += x_shift if i > 0 else 0
        landmarks[:, 0] += x_shift * i

        fig, ax = plot_landmarks(
            box_coords,
            box_connections,
            fig=fig,
            ax=ax,
            axis_lims=axis_lims,
            elev=elev,
            azim=azim,
            landmarks_color=LandmarksInfo.Colors._get_default_connection_color(),
            landmark_size=0,
            alpha=0.5,
        )
        fig, ax = plot_landmarks(
            landmarks,
            connections,
            fig=fig,
            ax=ax,
            axis_lims=axis_lims,
            elev=elev,
            azim=azim,
            landmarks_color=landmarks_color,
        )

    return fig, ax


def multi_frame_landmarks_to_npimage(
    multi_frame_landmarks: np.ndarray,
    connections: List[List[int]] = None,
    landmarks_color: Tuple[float, float, float] = LandmarksInfo.Colors.DEFAULT_LANDMARK,
    axis_lims: List[float] = None,
    elev=15,
    azim=25,
    move_hands=True,
    hands_location="auto",
    hands_scale=1.0,
    hands_distance=0.1,
) -> np.ndarray:
    """Draw multiple frames on a single graph image with x-shift

    Args:
        multi_frame_landmarks (np.ndarray): an array of dimentions (n_frames, n_landmarks, n_axes=3)
        connections (List[List[int]], optional): Indices of the landmarks array to be connected by a line. Defaults to None.
        landmarks_color (Tuple[float, float, float], optional): the color as RGB values [0 to 1] of the dots representing landmarks. Defaults to LandmarksInfo.Colors.DEFAULT_LANDMARK.
        axis_lims (List[float], optional): the upper and lower limits of the 3D plot axis [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper].. Defaults to None.
        elev (int, optional): the elevation angle of view point in degrees. Defaults to 15.
        azim (int, optional): Azimuth angle of view point in degrees. Defaults to 25.
        move_hands (bool, optional): wether to move origin centered hands apart or to wrists or not. Defaults to True.
        hands_location (str, optional): where to move hand to. ignored if move_hands is false. Defaults to "auto".
        hands_scale (float, optional): how much should hands be zoomed. Defaults to 1.0.
        hands_distance (float, optional): how much should hands be shifted "apart". Defaults to 0.1.

    Returns:
        np.ndarray: a 3 channel RGB image array containing drawn landmarks
    """

    multi_frame_landmarks = LandmarksInfo.Reshaper().fold(multi_frame_landmarks)

    if isinstance(multi_frame_landmarks, torch.Tensor):
        multi_frame_landmarks = multi_frame_landmarks.numpy()

    fig, ax = plot_multi_frame_landmarks(
        multi_frame_landmarks,
        connections=connections,
        landmarks_color=landmarks_color,
        axis_lims=axis_lims,
        elev=elev,
        azim=azim,
        move_hands=move_hands,
        hands_location=hands_location,
        hands_scale=hands_scale,
        hands_distance=hands_distance,
    )
    image = _fig_to_npimage(fig)

    plt.clf()
    plt.close()

    return image

def multi_frame_landmarks_to_pil_image(
    multi_frame_landmarks: np.ndarray,
    connections: List[List[int]] = None,
    landmarks_color: Tuple[float, float, float] = LandmarksInfo.Colors.DEFAULT_LANDMARK,
    axis_lims: List[float] = None,
    elev=15,
    azim=25,
    move_hands=True,
    hands_location="auto",
    hands_scale=1.0,
    hands_distance=0.1,
) -> PIL.Image:
    """Draw multiple frames on a single graph image with x-shift

    Args:
        multi_frame_landmarks (np.ndarray): an array of dimentions (n_frames, n_landmarks, n_axes=3)
        connections (List[List[int]], optional): Indices of the landmarks array to be connected by a line. Defaults to None.
        landmarks_color (Tuple[float, float, float], optional): the color as RGB values [0 to 1] of the dots representing landmarks. Defaults to LandmarksInfo.Colors.DEFAULT_LANDMARK.
        axis_lims (List[float], optional): the upper and lower limits of the 3D plot axis [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper].. Defaults to None.
        elev (int, optional): the elevation angle of view point in degrees. Defaults to 15.
        azim (int, optional): Azimuth angle of view point in degrees. Defaults to 25.
        move_hands (bool, optional): wether to move origin centered hands apart or to wrists or not. Defaults to True.
        hands_location (str, optional): where to move hand to. ignored if move_hands is false. Defaults to "auto".
        hands_scale (float, optional): how much should hands be zoomed. Defaults to 1.0.
        hands_distance (float, optional): how much should hands be shifted "apart". Defaults to 0.1.

    Returns:
        PIL.Image: a 3 channel RGB image array containing drawn landmarks
    """

    np_image = multi_frame_landmarks_to_npimage(
        multi_frame_landmarks = multi_frame_landmarks,
        connections=connections,
        landmarks_color=landmarks_color,
        axis_lims=axis_lims,
        elev=elev,
        azim=azim,
        move_hands=move_hands,
        hands_location=hands_location,
        hands_scale=hands_scale,
        hands_distance=hands_distance,
    )
    pil_image = np2pil(np_image)

    return pil_image


def multi_frame_landmarks_to_video(
    multi_frame_landmarks: np.ndarray,
    connections: List[List[int]] = None,
    landmarks_color: Tuple[float, float, float] = LandmarksInfo.Colors.DEFAULT_LANDMARK,
    axis_lims: List[float] = None,
    elev=15.0,
    azim=25.0,
    fps=24.0,
    rotate=True,
    move_hands=True,
    hands_location="auto",
    hands_scale=1.0,
    hands_distance=0.1,
    fig_width=10,
    fig_height=10,
) -> DataVideoClip:
    """Animate a pose vector's motion in 3D

    Args:
        multi_frame_landmarks (np.ndarray): an array of dimentions (n_frames, n_landmarks, n_axes=3)
        connections (List[List[int]], optional): Indices of the landmarks array to be connected by a line. Defaults to None.
        landmarks_color (Tuple[float, float, float], optional): the color as RGB values [0 to 1] of the dots representing landmarks. Defaults to LandmarksInfo.Colors.DEFAULT_LANDMARK.
        axis_lims (List[float], optional): the upper and lower limits of the 3D plot axis [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper].. Defaults to None.
        elev (int, optional): the elevation angle of view point in degrees. Defaults to 15.
        azim (int, optional): Azimuth angle of view point in degrees. Defaults to 25.
        fps (float, optional): the video frame rate per second. Defaults to 24.0.
        rotate (bool, optional): wether to add a cinematic effect to the animation. Defaults to True.
        move_hands (bool, optional): wether to move origin centered hands apart or to wrists or not. Defaults to True.
        hands_location (str, optional): where to move hand to. ignored if move_hands is false. Defaults to "auto".
        hands_scale (float, optional): how much should hands be zoomed. Defaults to 1.0.
        hands_distance (float, optional): how much should hands be shifted "apart". Defaults to 0.1.
        fig_width (int, optional): the width of the graph in inches. Defaults to 10.
        fig_height (int, optional): the height of the graph in inches. Defaults to 10.

    Returns:
        moviepy.video.VideoClip.DataVideoClip: a video of pose vector's motion in 3D
    """

    multi_frame_landmarks = LandmarksInfo.Reshaper().fold(multi_frame_landmarks)

    if isinstance(multi_frame_landmarks, torch.Tensor):
        multi_frame_landmarks = multi_frame_landmarks.numpy()

    if connections is None:
        connections = LandmarksInfo.infer_connections(multi_frame_landmarks.shape[-2])

    if move_hands:
        multi_frame_landmarks = _move_hands(
            multi_frame_landmarks,
            connections,
            hands_location,
            scale=hands_scale,
            distance=hands_distance,
        )

    x0, x1, y0, y1, z0, z1 = _get_box_coord_ranges(multi_frame_landmarks)

    if axis_lims is None:
        scale = 0.05
        axis_lims = [
            x0 - abs(x0 * scale),
            x1 + abs(x1 * scale),
            y0 - abs(y0 * scale),
            y1 + abs(y1 * scale),
            z0 - abs(z0 * scale),
            z1 + abs(z1 * scale),
        ]

    fig, ax, plot_cache = plot_landmarks(
        multi_frame_landmarks[0],
        connections,
        axis_lims=axis_lims,
        elev=elev,
        azim=azim,
        landmarks_color=landmarks_color,
        return_plot_cache=True,
        fig_width=fig_width,
        fig_height=fig_height,
    )
    ax.set_box_aspect((z1 - z0, x1 - x0, y1 - y0))

    def data_to_frame(t):
        if rotate:
            ax.view_init(
                elev=elev - elev * t * 1 / n_frames,
                azim=azim - azim * t * 2 / n_frames,
                vertical_axis="y",
            )

        if t > 0:
            update_plotted_landmarks(multi_frame_landmarks[t], plot_cache)

        image = _fig_to_npimage(fig)

        if t == len(multi_frame_landmarks) - 1:
            plt.clf()
            plt.close()

        return image

    n_frames = len(multi_frame_landmarks)
    timesteps = list(range(n_frames))
    clip = DataVideoClip(timesteps, data_to_frame, fps=fps)

    return clip
