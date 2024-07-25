import os
from copy import copy

import numpy as np
import pytest
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
    x = Landmarks([torch.tensor(data[0])])
    assert ((x.torch(torch.float32) - torch.Tensor([data[0]])).abs() < 1e-6).all()

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

    # test .connections
    with pytest.raises(ValueError) as exc_info:
        landmarks.connections
    assert "defined" in str(exc_info.value).lower()

    with pytest.raises(TypeError):
        landmarks.connections = []  # type: ignore

    with pytest.raises(ValueError) as exc_info:
        landmarks.connections = "mediapipe-world"
    assert "expected" in str(exc_info.value).lower()

    with pytest.raises(ValueError):
        landmarks.connections = "unknown"

    # test copy
    landmarks_copy = copy(landmarks)
    assert landmarks_copy.data.tolist() == landmarks.data.tolist()
    assert landmarks_copy._connections == landmarks._connections

    landmarks_copy.data[:] = 0
    assert (landmarks_copy.data == 0).all()
    assert not (landmarks.data == 0).all()


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

    # .load_asset
    loaded_landmarks = Landmarks.load_asset(
        fname := "xx-shapes-1_square.landmarks-testmodel.csv", overwrite=True
    )
    assert loaded_landmarks.n_landmarks == 4
    assert loaded_landmarks.n_features == 3
    assert os.path.exists(os.path.join(Assets.ROOT_DIR, "landmarks", fname))

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

    with pytest.raises(ValueError):
        Landmarks.concatenate([])

    path = Assets.download("landmarks/test-landmarks.mediapipe-world.csv")[0]
    landmarks_3 = Landmarks.load(path)

    with pytest.raises(ValueError) as exc_info:
        Landmarks.concatenate([landmarks_1, landmarks_3])
    assert "same connections" in str(exc_info.value).lower()


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

    with pytest.raises(FileExistsError):
        landmarks.save(path, overwrite=False)

    with pytest.raises(ValueError) as exc_info:  # unsupported extension
        landmarks.save("landmarks.tar.gz")
    assert "unsupported file format" in str(exc_info.value).lower()

    # LOAD

    with pytest.raises(ValueError):  # no header or model name in filename
        _ = Landmarks(path)

    with pytest.raises(ValueError) as exc_info:  # unsupported extension
        _ = Landmarks("landmarks.tar.gz")
    assert "unsupported file format" in str(exc_info.value).lower()

    with pytest.raises(TypeError):
        _ = Landmarks({"frame_1": [1, 2, 3], "frame_2": [4, 5, 6]})  # type: ignore

    with pytest.raises(ValueError):  # unknown format
        _ = Landmarks([["1,2,3", "2,3,4"]])  # type: ignore

    with pytest.raises(ValueError):
        _ = Landmarks(
            [
                {"landmark_1": "1,2,3", "landmark_2": "4,2,3"},
                {"landmark_1": "2,3,4", "landmark_2": "5,2,3"},  # type: ignore
            ]
        )

    four_d_data = np.array([[[[1, 2, 3], [2, 3, 4]], [[1, 2, 3], [2, 3, 4]]]])
    with pytest.raises(ValueError):
        _ = Landmarks(four_d_data)  # type: ignore

    with pytest.raises(ValueError):
        landmarks.data = four_d_data

    with pytest.raises(ValueError):  # no data
        setattr(landmarks, "_data", None)
        getattr(landmarks, "data")

    with pytest.warns(UserWarning) as record:
        Landmarks.load_asset(r"xx-shapes-1_.*.landmarks-testmodel.csv", overwrite=True)
    assert len(record) == 1
    assert "multiple" in str(record[0].message).lower()

    with pytest.raises(FileNotFoundError):
        Landmarks.load_asset(r"xx-shapes-1_.^.landmarks-testmodel.csv", overwrite=True)
