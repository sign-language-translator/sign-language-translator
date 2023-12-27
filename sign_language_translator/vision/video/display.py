from itertools import islice
from os.path import isfile
from time import sleep, time
from typing import Iterable
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import Animation, ArtistAnimation
from numpy import uint8
from numpy.typing import NDArray

from sign_language_translator.utils import in_jupyter_notebook


class VideoDisplay:
    @staticmethod
    def display_ipython_video_in_jupyter(
        path: str,
        html_attributes="loop autoplay controls preload muted",
        width=640,
        height=480,
    ) -> None:
        if not in_jupyter_notebook():
            warn(f"Attempted to use IPython.core.display.Video outside jupyter: {path}")
            return

        from IPython.display import Video, display  # type: ignore

        if isfile(path):
            display(
                Video(path, width=width, height=height, html_attributes=html_attributes)
            )
        else:
            raise FileNotFoundError(f"No video file found at {path}")

    @staticmethod
    def frames_to_matplotlib_animation(
        frames: Iterable[NDArray[uint8]], fps: float = 30.0, close_plot=True
    ) -> Animation:
        fig, ax = plt.subplots()
        fig.set_facecolor("black")
        ax.axis("off")
        fig.tight_layout(pad=0)
        ims = []

        for frame in frames:
            im = ax.imshow(frame, animated=True)
            ims.append([im])

        animation = ArtistAnimation(fig, ims, interval=(1000 / fps), blit=True)

        if close_plot:
            plt.close()

        return animation

    @staticmethod
    def display_frames(
        frames: Iterable[NDArray[uint8]],
        fps: float = 30.0,
        inline_player="jshtml" or "html5",
        max_frames=None,
    ) -> None:
        if isinstance(frames, np.ndarray) and frames.ndim == 3:
            frames = [frames]

        if max_frames is not None:
            frames = islice(frames, max_frames)

        if in_jupyter_notebook():
            if (hasattr(frames, "__len__") and len(frames) == 1) or max_frames == 1:  # type: ignore
                VideoDisplay.display_frames_as_ipython_image(
                    [next(iter(frames))], fps=float("inf")
                )
            else:
                from IPython.display import HTML, display  # type: ignore

                animation = VideoDisplay.frames_to_matplotlib_animation(
                    frames, fps, close_plot=True
                )
                html = (
                    animation.to_html5_video()
                    if inline_player == "html5"
                    else animation.to_jshtml()
                )
                display(HTML(html))
        else:
            if (hasattr(frames, "__len__") and len(frames) == 1) or max_frames == 1:  # type: ignore
                VideoDisplay.show_image_plot(next(iter(frames)))  # type: ignore
            else:
                # TODO: Improve fps of the video player.
                # TODO: Add seek bar to the video player.
                animation = VideoDisplay.frames_to_matplotlib_animation(
                    frames, fps, close_plot=False
                )
            plt.show(block=False)  # TODO: window name
            plt.pause((len(frames) / fps) * 1.02 if hasattr(frames, "__len__") else 10)  # type: ignore
            plt.close()

    @staticmethod
    def display_frames_as_ipython_image(
        frames: Iterable[NDArray[uint8]], fps: float = 30.0
    ):
        # BUG: randomly prints None or shows Image

        from IPython.display import Image, display  # type: ignore

        display_handle = display(None, display_id=True)
        for frame in frames:
            start_time = time()
            # RGB to BGR because opencv is weird
            _, frame = cv2.imencode(".jpeg", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            display_handle.update(Image(data=frame.tobytes()))  # type: ignore

            elapsed_time = (start_time - time()) if start_time is not None else 0
            sleep(max(0, 1 / fps - elapsed_time))

    @staticmethod
    def show_image_plot(frame: NDArray[uint8]) -> None:
        plt.figure(facecolor="black")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.imshow(frame)
