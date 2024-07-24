import os

from sign_language_translator.config.assets import Assets
from sign_language_translator.vision.landmarks.landmarks import Landmarks


def test_landmarks_frames_grid():
    # load
    asset_id = "xx-shapes-1_square.landmarks-testmodel.csv"
    landmarks = Landmarks.load_asset(asset_id)

    # save
    landmarks.save_frames_grid(
        (grid_path := "temp/grid_sq.jpg"),
        2,
        2,
        overwrite=True,
        elevation_delta=10,
    )
    assert os.path.exists(grid_path)

    # show
    landmarks.show_frames_grid(3, 1)


def test_landmarks_animation():
    # load
    path = Assets.download("landmarks/test-landmarks.mediapipe-world.csv")[0]
    landmarks = Landmarks.load(path)

    # save
    landmarks.save_animation(
        (animation_path := "temp/animation.gif"),
        overwrite=True,
        style="dark_background",
        roll_delta=-1,
        writer="pillow",
    )
    assert os.path.exists(animation_path)

    # show
    landmarks.show()
