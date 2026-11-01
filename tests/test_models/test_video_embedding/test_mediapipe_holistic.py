import sys

import pytest

from sign_language_translator.config.assets import Assets
from sign_language_translator.models import MediaPipeHolisticModel
from sign_language_translator.vision.utils import read_frames_with_opencv


def test_mediapipe_holistic_embedding():
    if sys.version_info >= (3, 12):
        pytest.skip("MediaPipe is not supported in Python >3.11")

    Assets.download("videos/wordless_wordless.mp4")
    file_path = Assets.get_path("videos/wordless_wordless.mp4")[0]

    frames = read_frames_with_opencv(file_path)

    model = MediaPipeHolisticModel()
    landmarks = model.embed(frames, landmark_type="all")

    with pytest.raises(ValueError):
        landmarks = model.embed(frames, landmark_type="3D")

    n_values = (33 + 21 * 2) * 5
    assert landmarks.shape[0] == len(frames)
    assert landmarks.shape[-1] == n_values * 2
    assert not (landmarks == 0).all()

    # world landmarks come first; Holistic gives no world landmarks for hands
    world, image = landmarks[:, :n_values], landmarks[:, n_values:]
    assert not (world[:, : 33 * 5] == 0).all()
    assert (world[:, 33 * 5 :] == 0).all()
    assert not (image[:, : 33 * 5] == 0).all()

    image_only = model.embed(frames, landmark_type="image")
    assert image_only.shape == (len(frames), n_values)
    assert (image_only == image).all()


def test_mediapipe_holistic_model_complexity():
    if sys.version_info >= (3, 12):
        pytest.skip("MediaPipe is not supported in Python >3.11")

    with pytest.raises(ValueError):
        MediaPipeHolisticModel(model_complexity=3)
