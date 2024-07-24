import numpy as np

from sign_language_translator.vision.landmarks.landmarks import Landmarks


def test_lambda_landmarks_transform():
    data = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[4, 5, 6], [7, 8, 9], [0, 1, 2]],
    ]
    data = np.array(data)
    landmarks = Landmarks(data)

    landmarks.transform(lambda x: x + 5, inplace=True)
    new_landmarks = landmarks.transform(lambda x: x + 5, inplace=False)
    assert (np.array(new_landmarks) == (data + 10)).all()
    assert (np.array(landmarks) == (data + 5)).all()
