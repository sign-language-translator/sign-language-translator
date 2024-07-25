import cv2
import numpy as np
import pytest

from sign_language_translator.config.enums import SignFormats
from sign_language_translator.vision._utils import get_sign_wrapper_class
from sign_language_translator.vision.utils import iter_frames_with_opencv


def test_image_iteration():
    # create and save image
    image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    image_path = "temp/test_image_iteration.png"
    cv2.imwrite(image_path, image)

    # test iteration
    for frame in iter_frames_with_opencv(image_path):
        assert frame is not None
        assert frame.shape == image.shape

    frames = list(iter_frames_with_opencv(image_path))
    assert len(frames) == 1


def test_get_sign_wrapper_class():
    assert get_sign_wrapper_class("video").name() == SignFormats.VIDEO.value
    assert get_sign_wrapper_class("landmarks").name() == SignFormats.LANDMARKS.value

    with pytest.raises(ValueError):
        get_sign_wrapper_class("unknown")
