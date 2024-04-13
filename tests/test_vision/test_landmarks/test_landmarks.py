import os

import numpy as np
import torch

from sign_language_translator.config.assets import Assets
from sign_language_translator.config.enums import SignFormats
from sign_language_translator.vision.landmarks.landmarks import Landmarks


def test_landmarks_initialization_from_data():

    data = [
        [[0, 0], [1, 1], [2, 2], [3, 3]],
        [[0, 1], [1, 2], [3, 3], [4, 4]],
    ]
    # test __init__ arg types
    landmarks = Landmarks(data)
    _ = Landmarks(np.array(data))
    _ = Landmarks(torch.tensor(data))
    _ = Landmarks([np.array(data[0])])
    _ = Landmarks([torch.tensor(data[0])])

    # test casting
    assert landmarks.tolist() == data
    assert (landmarks.numpy() == np.array(data)).all()
    assert (landmarks.torch() == torch.tensor(data)).all()

    # test properties
    assert Landmarks.name() == SignFormats.LANDMARKS.value
    assert landmarks.n_frames == 2
    assert landmarks.n_landmarks == 4
    assert landmarks.n_coordinates == 2
    assert landmarks.ndim == 3
    assert landmarks.shape == (2, 4, 2)

    frame = [[[1, 0], [2, 1], [3, 2], [4, 3]]]
    landmarks.data = np.array(data + frame)
    assert landmarks.n_frames == 3
    assert landmarks.n_landmarks == 4
    assert landmarks.n_coordinates == 2
    assert (landmarks.data == np.array(data + frame)).all()

    # test indexing / getitem
    assert landmarks[0].data.tolist() == data[0:1]
    assert landmarks[2:, :2, 0].tolist() == [[[1], [2]]]
    assert landmarks[:1, 2:, 1].shape == (1, 2, 1)
    assert landmarks[-1, torch.tensor([1, 2]), :].ndim == 3
    assert landmarks[2, np.array([1, 2]), :].ndim == 3
    landmarks.data = landmarks.torch()
    assert landmarks[-1, torch.Tensor([1, 2]), :].shape == (1, 2, 2)
    assert landmarks[2, np.array([1, 2]), :].shape == (1, 2, 2)

    # test iteration
    for i, frame_ in enumerate(landmarks[:2]):  # drops a dimension
        assert frame_.tolist() == data[i]


def test_landmarks_load_and_save():
    data = [
        [[0, 0], [1, 1], [2, 2], [3, 3]],
        [[0, 1], [1, 2], [3, 3], [4, 4]],
        [[1, 0], [2, 1], [3, 2], [4, 3]],
    ]
    data = np.array(data)
    landmarks = Landmarks(data)

    # CSV
    landmarks.save(csv_path := os.path.join("temp", "landmarks.csv"), overwrite=True)

    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        assert header == "x0,y0,x1,y1,x2,y2,x3,y3", f"wrong header: {header}"
        assert sum(1 for _ in f) == 3

    loaded_landmarks = Landmarks.load(csv_path)
    assert (loaded_landmarks.data == data).all()

    # without header (use file name)
    with open(
        path := os.path.join("temp", "landmarks.mediapipe-world.csv"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(("1," * 75 * 5)[:-1] + "\n")
        f.write(("2," * 75 * 5)[:-1] + "\n")
        f.write(("3," * 75 * 5)[:-1] + "\n")

    loaded_landmarks = Landmarks.load(path)
    assert loaded_landmarks.n_frames == 3
    assert loaded_landmarks.n_landmarks == 75
    assert loaded_landmarks.n_features == 5

    # from asset
    path = Assets.download("landmarks/test-landmarks.mediapipe-all.csv")[0]
    loaded_landmarks = Landmarks.load(path)
    assert loaded_landmarks.n_landmarks == 150
    assert loaded_landmarks.n_features == 5

    restructured = Landmarks(loaded_landmarks.numpy().reshape((-1, 5, 150)))
    restructured.save(path := os.path.join("temp", "landmarks.csv"), overwrite=True)
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        assert header[-1][-1] == "4"
        assert header[27] == "ab0"

    # NPY
    landmarks.save(npy_path := os.path.join("temp", "landmarks.npy"), overwrite=True)
    loaded_landmarks = Landmarks.load(npy_path)
    assert (loaded_landmarks.data == data).all()

    # PT
    landmarks.save(pt__path := os.path.join("temp", "landmarks.pt"), overwrite=True)
    loaded_landmarks = Landmarks.load(pt__path)
    assert (np.array(loaded_landmarks.data) == data).all()


def test_landmarks_concatenation():
    data = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[4, 5, 6], [7, 8, 9], [0, 1, 2]],
    ]
    landmarks_1 = Landmarks(data[:1])
    landmarks_2 = Landmarks(data[1:])

    assert Landmarks.concatenate([landmarks_1, landmarks_2]).data.tolist() == data


def test_landmarks_validation():
    data = [
        [[0, 0], [1, 1], [2, 2], [3, 3]],
        [[0, 1], [1, 2], [3, 3], [4, 4]],
        [[1, 0], [2, 1], [3, 2], [4, 3]],
    ]
    data = np.array(data)
    landmarks = Landmarks(data)

    # SAVE
    # file without header or model name in filename
    with open(path := os.path.join("temp", "landmark.csv"), "w", encoding="utf-8") as f:
        f.write("0,0,1,1,2,2,3,3\n")
        f.write("0,1,1,2,3,3,4,4\n")
        f.write("1,0,2,1,3,2,4,3\n")
        f.write("1,0,2,1,3,2,4,3\n")

    try:
        landmarks.save(path, overwrite=False)
    except FileExistsError:
        pass

    try:
        landmarks.save("landmarks.tar.gz")
    except ValueError:  # unsupported extension
        pass

    # LOAD

    try:
        _ = Landmarks(path)
    except ValueError:  # no header or model name in filename
        pass

    try:
        _ = Landmarks("landmarks.tar.gz")
    except ValueError:  # unsupported extension
        pass

    try:
        _ = Landmarks({"frame_1": [1, 2, 3], "frame_2": [4, 5, 6]})  # type: ignore
    except TypeError:
        pass

    try:
        _ = Landmarks([["1,2,3", "2,3,4"]])  # type: ignore
    except ValueError:  # unknown format
        pass
    try:
        _ = Landmarks(
            [
                {"landmark_1": "1,2,3", "landmark_2": "4,2,3"},
                {"landmark_1": "2,3,4", "landmark_2": "5,2,3"},  # type: ignore
            ]
        )
    except ValueError:
        pass

    four_d_data = np.array([[[[1, 2, 3], [2, 3, 4]], [[1, 2, 3], [2, 3, 4]]]])
    try:
        _ = Landmarks(four_d_data)  # type: ignore
    except ValueError:
        pass
    try:
        landmarks.data = four_d_data
    except ValueError:
        pass

    try:
        setattr(landmarks, "_data", None)
        getattr(landmarks, "data")
    except ValueError:  # no data
        pass
