import os

from sign_language_translator import Settings
from sign_language_translator.models import MediaPipeLandmarksModel
from sign_language_translator.utils import download_resource
from sign_language_translator.vision.utils import read_frames_with_opencv


def test_mediapipe_embedding():
    download_resource("videos/wordless_wordless.mp4")
    file_path = os.path.join(
        Settings.RESOURCES_ROOT_DIRECTORY, "videos", "wordless_wordless.mp4"
    )

    frames = read_frames_with_opencv(file_path)

    model = MediaPipeLandmarksModel()
    landmarks = model.embed(frames, landmark_type="all")

    assert landmarks.shape[0] == len(frames)
    assert landmarks.shape[-1] == (33 + 21 * 2) * 5 * 2
    assert not (landmarks == 0).all()
