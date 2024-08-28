import sys

import pytest

from sign_language_translator.config.assets import Assets
from sign_language_translator.models import MediaPipeLandmarksModel
from sign_language_translator.vision.utils import read_frames_with_opencv


def test_mediapipe_embedding():
    if sys.version_info >= (3, 12):
        pytest.skip("MediaPipe is not supported in Python >3.11")

    Assets.download("videos/wordless_wordless.mp4")
    file_path = Assets.get_path("videos/wordless_wordless.mp4")[0]

    frames = read_frames_with_opencv(file_path)

    model = MediaPipeLandmarksModel()
    landmarks = model.embed(frames, landmark_type="all")

    with pytest.raises(ValueError):
        landmarks = model.embed(frames, landmark_type="3D")

    assert landmarks.shape[0] == len(frames)
    assert landmarks.shape[-1] == (33 + 21 * 2) * 5 * 2
    assert not (landmarks == 0).all()
