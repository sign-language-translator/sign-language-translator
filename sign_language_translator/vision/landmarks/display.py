import math
import re
from functools import partial
from itertools import zip_longest
from typing import List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Line3D, Path3DCollection
from numpy.typing import NDArray

from sign_language_translator.config.colors import Colors


class MatPlot3D:

    @staticmethod
    def new_figure(
        x_limits: Tuple[float, float],
        y_limits: Tuple[float, float],
        z_limits: Tuple[float, float],
        vertical_axis: Literal["x", "y", "z"] = "z",
        figure_scale: Optional[float] = 5,
        style: Literal["dark_background", "default"] = "default",
        layout: Literal["constrained", "compressed", "tight", "none"] = "compressed",
        subplots: Tuple[int, int] = (1, 1),
    ) -> Tuple[Figure, List[Axes]]:
        """Creates a new 3D figure with the specified subplots and settings."""
        # figure size
        if figure_scale and figure_scale > 0:
            sizes: List = np.stack([x_limits, y_limits, z_limits]).ptp(1).tolist()
            ratio = sizes.pop("xyz".index(vertical_axis.lower()))
            ratio = ratio / np.linalg.norm(sizes)
            ratio = 0.9 * ratio  # make it wider
            figsize = (
                figure_scale * max(1, 1 / ratio) * subplots[1],
                figure_scale * max(1, ratio) * subplots[0],
            )
        else:
            figsize = None

        plt.style.use(style)  # todo: set only for this figure
        fig = plt.figure(figsize=figsize, layout=layout)
        axes = [
            fig.add_subplot(*subplots, i + 1, projection="3d")
            for i in range(np.prod(subplots))
        ]

        return fig, axes

    @staticmethod
    def initialize_Axes3D(
        ax: Axes,
        x_limits: Tuple[float, float],
        y_limits: Tuple[float, float],
        z_limits: Tuple[float, float],
        ticks_scale: float = 1.0,
        azimuth: float = 20,
        elevation: float = 15,
        roll: float = 0,
        vertical_axis: str = "y",
        invert_x: bool = False,
        invert_y: bool = False,
        invert_z: bool = False,
        show_grid: bool = True,
        show_axis: bool = True,
    ) -> None:
        """Initializes a 3D Axes object with specified limits, ticks, and settings.

        Args:
            ax (Axes): The 3D Axes object to be initialized.
            x_limits (Tuple[float, float]): The range of the x-axis from minimum to maximum value.
            y_limits (Tuple[float, float]): The range of the y-axis from minimum to maximum value.
            z_limits (Tuple[float, float]): The range of the z-axis from minimum to maximum value.
        """
        ax.set(xlim3d=x_limits, xlabel="X")
        ax.set(ylim3d=y_limits, ylabel="Y")
        ax.set(zlim3d=z_limits, zlabel="Z")

        if not ticks_scale:
            ticks_scale = _nearest_scale(
                np.array([x_limits, y_limits, z_limits]).ptp(1).min()
            )

        ax.xaxis.set_major_locator(MultipleLocator(ticks_scale))
        ax.yaxis.set_major_locator(MultipleLocator(ticks_scale))
        ax.zaxis.set_major_locator(MultipleLocator(ticks_scale))  # type: ignore

        if invert_x:
            ax.invert_xaxis()
        if invert_y:
            ax.invert_yaxis()
        if invert_z:
            ax.invert_zaxis()  # type: ignore

        ax.set_box_aspect(
            np.array([x_limits, y_limits, z_limits]).ptp(1)[
                # change aspect ratio order according to vertical axis
                {"x": [1, 2, 0], "y": [2, 0, 1], "z": [0, 1, 2]}[vertical_axis.lower()]
            ]
        )

        ax.view_init(  # type: ignore
            azim=azimuth, elev=elevation, roll=roll, vertical_axis=vertical_axis
        )
        ax.grid(show_grid)
        ax.axis("on" if show_axis else "off")

    @staticmethod
    def placeholder_scatter_and_lines(
        ax: Axes,
        n_lines: int,
        line_colors: Sequence[Union[Tuple[float, float, float], None]] = (),
        line_labels: Sequence[Union[str, None]] = (),
        scatter_color: Tuple[float, float, float] = (0, 0, 0),
        scatter_size: float = 2,
    ) -> Tuple[Path3DCollection, List[Line3D]]:
        """
        Update a 3D plot with empty Path3DCollection (scatter) and Line3D objects.

        Args:
            ax (Axes): The 3D axes object to plot on.
            n_lines (int): The number of placeholder lines to create.
            line_colors (Sequence[Union[Tuple[float, float, float], None]], optional): The colors of the lines. If not provided, a gradient of colors will be used.
            line_labels (Sequence[Union[str, None]], optional): The labels for the lines.
            scatter_color (Tuple[float, float, float], optional): The RGB color of the scatter points normalized to [0.0, 1.0] range. Defaults to black.
            scatter_size (float, optional): The size of the scatter points. Defaults to 2.

        Returns:
            Tuple[Path3DCollection, List[Line3D]]: A tuple containing the scatter plot and the list of lines.
        """
        n_lines = max(round(n_lines), 0)
        if not line_colors:
            line_colors = np.array(Colors.gradient(n_lines)) / 255  # type: ignore

        scatter = ax.scatter3D(xs=[], ys=[], zs=[], color=scatter_color, s=scatter_size)  # type: ignore
        lines = [
            ax.plot3D(xs=[], ys=[], zs=[], color=color, label=label)[0]  # type: ignore
            for i, color, label in zip_longest(range(n_lines), line_colors, line_labels)
            if i < n_lines
        ]
        return scatter, lines

    @staticmethod
    def set_frame_data(
        points: Union[Sequence[Tuple[float, float, float]], NDArray],
        scatter: Path3DCollection,
        lines: Sequence[Line3D],
        line_indexes: Sequence[Sequence[int]] = (),
        ax: Optional[Axes] = None,
        azimuth_delta: float = 0,
        elevation_delta: float = 0,
        roll_delta: float = 0,
    ) -> List[Union[Path3DCollection, Line3D]]:
        """Sets the frame data for visualization.

        Args:
            points (Union[Sequence[Tuple[float, float, float]], NDArray]): A collection of tuples or a 2D NDArray representing the (x, y, z) points.
            scatter (Path3DCollection): Object representing the scatter plot.
            lines (Sequence[Line3D]): A sequence of Line3D objects representing the lines to be plotted.
            line_indexes (Sequence[Sequence[int]], optional): indexes of points to connect with lines. Defaults to ().
            ax (Optional[Axes], optional): An optional Axes object to update the view. Defaults to None.

        Returns:
            List[Union[Path3DCollection, Line3D]]: A list containing the updated scatter plot and lines objects.
        """
        if len(line_indexes) == 0 and lines:
            line_indexes = _indexes_to_connect(len(points))

        # update view
        if ax is not None:
            if ax.get_title():
                ax.set_title(
                    re.sub(
                        r"\[t_-?\d+\]",
                        lambda timestamp: f"[t_{int(timestamp.group()[3:-1])+1}]",
                        ax.get_title(),
                    )
                )
            if azimuth_delta:
                ax.azim = ax.azim + azimuth_delta  # type: ignore
            if elevation_delta:
                ax.elev = ax.elev + elevation_delta  # type: ignore
            if roll_delta:
                ax.roll = ax.roll + roll_delta  # type: ignore

        # update data
        points = np.array(points)
        scatter._offsets3d = (  # pylint: disable=protected-access
            points[..., 0].ravel(),
            points[..., 1].ravel(),
            points[..., 2].ravel(),
        )
        for idx_track, line in zip(line_indexes, lines):
            line.set_data_3d(points[..., idx_track, :3].T)

        return [scatter] + list(lines)

    @classmethod
    def animate(
        cls,
        frames: Union[Sequence[Sequence[Tuple[float, float, float]]], NDArray],
        line_indexes: Optional[Sequence[Sequence[int]]] = None,
        line_colors: Sequence[Union[Tuple[float, float, float], None]] = (),
        line_labels: Sequence[Union[str, None]] = (),
        scatter_color: Tuple[float, float, float] = (0, 0, 0),
        scatter_size: float = 2,
        title: Optional[str] = "{frame_number}",
        vertical_axis: Literal["x", "y", "z"] = "z",
        ticks_scale: Optional[float] = None,
        azimuth: float = 20,
        elevation: float = 15,
        roll: float = 0,
        azimuth_delta: float = 0,
        elevation_delta: float = 0,
        roll_delta: float = 0,
        invert_x: bool = False,
        invert_y: bool = False,
        invert_z: bool = False,
        show_grid: bool = True,
        show_axis: bool = True,
        figure_scale: Optional[float] = None,
        style: Literal["dark_background", "default"] = "default",
        layout: Literal["constrained", "compressed", "tight", "none"] = "none",
        interval: Union[float, int] = 37,
        repeat_delay: Union[float, int] = 100,
        blit: bool = True,
    ) -> FuncAnimation:
        """
        Animates the given frames representing 3D coordinates with 3D scatter plot and lines connecting those points.

        Args:
            frames (Union[Sequence[Sequence[Tuple[float, float, float]]], NDArray]): The frames to animate, represented as a sequence of collection of 3D coordinates.
            line_indexes (Optional[Sequence[Sequence[int]]]): The indexes of the points in a frame to connect in lines.  If not provided, connects the points in a cycle [0, 1, 2, ..., n-1, 0].
            line_colors (Sequence[Union[Tuple[float, float, float], None]]): The colors of the lines in RGB format normalized to [0.0, 1.0] range. If not provided, default to a gradient of blue to pink to blue.
            line_labels (Sequence[Union[str, None]]): The labels of the lines.
            scatter_color (Tuple[float, float, float]): The color of the scatter points in RGB format normalized to [0.0, 1.0] range. Default is black.
            title (Optional[str]): The title of the animation. Can include the placeholder "{frame_number}" to display the frame number. Defaults to "{frame_number}".
            vertical_axis (Literal["x", "y", "z"]): The vertical axis in the plot. Default is "z".
            ticks_scale (Optional[float]): The scale of the ticks. Defaults to the nearest power of 10 under the range in data.
            layout (Literal["constrained", "compressed", "tight", "none"]): The layout of the plot. Default is "none".
            interval (Union[float, int]): The interval between frames in milliseconds. Default is 37.
            repeat_delay (Union[float, int]): The delay between replays in milliseconds. Default is 100.
            blit (bool): Whether to use blitting for faster updates. Default is True.

        Returns:
            FuncAnimation: The animation object.
        """
        frames = np.array(frames)
        limits = np.stack([frames.min(axis=(0, 1))[:3], frames.max(axis=(0, 1))[:3]]).T

        if line_indexes is None:
            line_indexes = _indexes_to_connect(len(frames[0]))
        if not ticks_scale:
            ticks_scale = _nearest_scale(limits.ptp(1).min())

        fig, [ax] = cls.new_figure(
            *limits,
            vertical_axis=vertical_axis,
            figure_scale=figure_scale,
            style=style,
            layout=layout,
            subplots=(1, 1),
        )
        cls.initialize_Axes3D(
            ax,
            *limits,
            ticks_scale=ticks_scale,
            azimuth=azimuth,
            elevation=elevation,
            roll=roll,
            vertical_axis=vertical_axis,
            invert_x=invert_x,
            invert_y=invert_y,
            invert_z=invert_z,
            show_grid=show_grid,
            show_axis=show_axis,
        )

        scatter, lines = cls.placeholder_scatter_and_lines(
            ax,
            len(line_indexes),
            line_colors=line_colors,
            line_labels=line_labels,
            scatter_color=scatter_color,
            scatter_size=scatter_size,
        )

        if line_labels:
            ax.legend(loc="center right")
        if title:
            # TODO: why is -4 required here to show t_0 in the fist frame???
            ax.set_title(title.format(frame_number="[t_-4]"))

        animation = FuncAnimation(
            fig,
            partial(
                cls.set_frame_data,
                scatter=scatter,
                lines=lines,
                line_indexes=line_indexes,
                ax=ax,
                azimuth_delta=azimuth_delta,
                elevation_delta=elevation_delta,
                roll_delta=roll_delta,
            ),
            frames=frames,
            interval=interval,
            blit=blit,
            repeat_delay=repeat_delay,
        )

        # plt.close(fig)  # todo: handle trailing figure

        return animation

    @classmethod
    def frames_grid(
        cls,
        frames: Union[Sequence[Sequence[Tuple[float, float, float]]], NDArray],
        subplots: Tuple[int, int],
        line_indexes: Optional[Sequence[Sequence[int]]] = None,
        line_colors: Sequence[Union[Tuple[float, float, float], None]] = (),
        line_labels: Sequence[Union[str, None]] = (),
        scatter_color: Tuple[float, float, float] = (0, 0, 0),
        scatter_size: float = 2,
        title: Optional[str] = "{frame_number}",
        figure_title: Optional[str] = None,
        figure_title_font_size: float = 20,
        vertical_axis: Literal["x", "y", "z"] = "z",
        ticks_scale: Optional[float] = None,
        azimuth: float = 20,
        elevation: float = 15,
        roll: float = 0,
        azimuth_delta: float = 0,
        elevation_delta: float = 0,
        roll_delta: float = 0,
        invert_x: bool = False,
        invert_y: bool = False,
        invert_z: bool = False,
        show_grid: bool = True,
        show_axis: bool = True,
        figure_scale: Optional[float] = 4,
        style: Literal["dark_background", "default"] = "default",
        layout: Literal["constrained", "compressed", "tight", "none"] = "tight",
    ) -> Figure:
        """
        Generates a grid of frames with 3D scatter plots and lines connecting the points.

        Args:
            frames (Union[Sequence[Sequence[Tuple[float, float, float]]], NDArray]): The frames containing the 3D coordinates of the points.
            subplots (Tuple[int, int]): The number of rows and columns in the figure. Each cell is a 3D plot containing one frame.
            line_indexes (Optional[Sequence[Sequence[int]]]): The indexes of points to be connected with lines.
            line_colors (Sequence[Union[Tuple[float, float, float], None]]): The colors of the lines connecting the points. Color should be in RGB format and in range [0.0, 1.0].
            line_labels (Sequence[Union[str, None]]): The labels for the lines connecting the points.
            scatter_color (Tuple[float, float, float]): The color of the scatter points. Color should be in RGB format and in range [0.0, 1.0].
            scatter_size (float): The size of the scatter points.
            title (Optional[str]): The title of each subplot. Can include the placeholder "{frame_number}" to display the frame number.
            figure_title (Optional[str]): The title of the entire figure.
            figure_title_font_size (float): The font size of the figure title.
            vertical_axis (Literal["x", "y", "z"]): The vertical axis in the 3D plots.
            azimuth (float): The azimuth angle (rotation around the vertical axis) of the initial view in the plot. Value must be in degrees.
            elevation (float): The elevation angle (amount of rise from the horizontal plane) of the initial view in the plot.  Value must be in degrees.
            roll (float): The roll angle (rotation around the line of sight) of the initial view in the plot. Value must be in degrees.
            azimuth_delta (float): The change in azimuth angle for each subplot. Value must be in degrees.
            elevation_delta (float): The change in elevation angle for each subplot. Value must be in degrees.
            roll_delta (float): The change in roll angle for each subplot. Value must be in degrees.
            invert_x (bool): Whether to invert the x-axis.
            invert_y (bool): Whether to invert the y-axis.
            invert_z (bool): Whether to invert the z-axis.
            show_grid (bool): Whether to show the grid lines on the axes.
            show_axis (bool): Whether to show the axis lines.
            figure_scale (Optional[float]): The size of the entire figure.
            style (Literal["dark_background", "default"]): The color theme of the plot.
            layout (Literal["constrained", "compressed", "tight", "none"]): The spacing between the subplots.

        Returns:
            Figure: The generated matplotlib figure.
        """
        frames = np.array(frames)
        limits = np.stack([frames.min(axis=(0, 1))[:3], frames.max(axis=(0, 1))[:3]]).T
        indexes = np.linspace(0, len(frames) - 1, np.prod(subplots)).round().astype(int)

        if line_indexes is None:
            line_indexes = _indexes_to_connect(len(frames[0]))
        if not ticks_scale:
            ticks_scale = _nearest_scale(limits.ptp(1).min())

        fig, axes = cls.new_figure(
            *limits,
            vertical_axis=vertical_axis,
            figure_scale=figure_scale,
            style=style,
            layout=layout,
            subplots=subplots,
        )

        for i, ax in enumerate(axes):
            cls.initialize_Axes3D(
                ax,
                *limits,
                ticks_scale=ticks_scale,
                azimuth=azimuth + i * azimuth_delta,
                elevation=elevation + i * elevation_delta,
                roll=roll + i * roll_delta,
                vertical_axis=vertical_axis,
                invert_x=invert_x,
                invert_y=invert_y,
                invert_z=invert_z,
                show_grid=show_grid,
                show_axis=show_axis,
            )

            scatter, lines = cls.placeholder_scatter_and_lines(
                ax,
                len(line_indexes),
                line_colors=line_colors,
                line_labels=line_labels,
                scatter_color=scatter_color,
                scatter_size=scatter_size,
            )

            if title:
                ax.set_title(title.format(frame_number=f"[t_{indexes[i]}]"))

            cls.set_frame_data(
                frames[indexes[i]], scatter, lines, line_indexes=line_indexes
            )

        if line_labels:
            fig.legend(handles=axes[0].lines, loc="center right")

        if figure_title:
            fig.suptitle(figure_title, fontsize=figure_title_font_size)

        return fig


def _indexes_to_connect(n: int) -> List[List[int]]:
    return [[i, (i + 1) % n] for i in range(n)]


def _nearest_scale(x: float, base: int = 10) -> float:
    return float(base ** math.floor(math.log(x, base)))


def _reset_counter_in_animation_title(animation: FuncAnimation) -> None:
    if isinstance(animation, FuncAnimation):
        ax = animation._fig.gca()  # type: ignore  # pylint: disable=protected-access
        ax.set_title(re.sub(r"\[t_-?\d+\]", "[t_-2]", ax.get_title()))


# todo: class Plotly3D:
# todo: class Avatar3D: # uses open3d and custom avatar
