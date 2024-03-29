import numpy as np
import torch

from sign_language_translator.utils.arrays import (
    ArrayOps,
    adjust_vector_angle,
    align_vectors,
    linear_interpolation,
)


def test_linear_interpolation():
    array = np.array(
        [
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            [
                [12, 13, 14, 15],
                [16, 17, 18, 19],
                [20, 21, 22, 23],
            ],
        ]
    )

    # test intermediate frames
    indices = [0, 0.2, 0.5]
    new_array = linear_interpolation(array, new_indexes=indices, dim=0)
    expected_array = np.array(
        [
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            [
                [2.4, 3.4, 4.4, 5.4],
                [6.4, 7.4, 8.4, 9.4],
                [10.4, 11.4, 12.4, 13.4],
            ],
            [
                [6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0],
                [14.0, 15.0, 16.0, 17.0],
            ],
        ]
    )
    assert (np.abs(new_array - expected_array) < 1e-4).all()

    # test intermediate rows
    old_x = np.array([3, 5, 9])
    new_x = [4, 7.7, 9]
    new_array = linear_interpolation(torch.Tensor(array), old_x=old_x, new_x=new_x, dim=1)  # type: ignore
    assert new_array.isclose(  # type: ignore
        torch.Tensor(
            [
                [
                    [2.0, 3.0, 4.0, 5.0],
                    [6.7, 7.7, 8.7, 9.7],
                    [8.0, 9.0, 10.0, 11.0],
                ],
                [
                    [14.0, 15.0, 16.0, 17.0],
                    [18.7, 19.7, 20.7, 21.7],
                    [20.0, 21.0, 22.0, 23.0],
                ],
            ]
        )
    ).all()

    try:
        linear_interpolation(array, new_indexes=indices, old_x=old_x, dim=2)  # type: ignore
    except ValueError:
        pass
    try:
        linear_interpolation(array, new_indexes=indices, dim=6)  # type: ignore
    except ValueError:
        pass


def test_array_ops():
    tensor = torch.arange(10)
    array = np.arange(10)

    assert (tensor == ArrayOps.floor(tensor + 0.1)).all()
    assert (array == ArrayOps.floor(array + 0.1)).all()

    assert ArrayOps.norm(torch.Tensor([1, 1])) == torch.Tensor([2]).sqrt()

    # test ArrayOps.top_k
    tensor = torch.tensor([1, 2, 3, 4, 5])
    _, indices = ArrayOps.top_k(tensor, 3)
    assert set(indices.tolist()) == {4, 3, 2}

    array = np.array([3, 5, 1, 2, 4])
    _, indices = ArrayOps.top_k(array, 3)
    assert set(indices.tolist()) == {1, 4, 0}


def test_adjust_vector_angle():
    v1 = np.array([1, 1])
    v2 = np.array([1, -1])

    #   |  ,· (1, 1)           [v1]
    #   | /,· (1, 1/sqrt(3))   [new_v1]
    # --+---+---+-->
    #   | \`· (1, -1/sqrt(3))  [new_v2]
    #   |  `· (1, -1)          [v2]

    height = v1[1] - 0
    target_height = 1 / np.sqrt(3)  # height at 30 degrees (width=1)

    v1_weight = (target_height + height) / (2 * height)

    new_v1, new_v2 = adjust_vector_angle(v1, v2, v1_weight, post_normalize=True)

    # chect norm
    assert np.isclose(np.linalg.norm(new_v1), 1)
    assert np.isclose(np.linalg.norm(new_v2), 1)

    # check angle
    assert np.isclose(new_v1, np.array([np.sqrt(3) / 2, 0.5])).all()
    assert np.isclose(new_v2, np.array([np.sqrt(3) / 2, -0.5])).all()

    def rotation_matrix_2d(x: float):
        """get the 2d rotation matrix based on radian angle theta

        Args:
            theta (float): clockwise angle in radians
        """
        return np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])

    new_v1, new_v2 = adjust_vector_angle(v1, v2, v1_weight, post_normalize=False)

    R = rotation_matrix_2d(-15 / 180 * np.pi)
    assert np.isclose(new_v1, R @ v1).all()

    R = rotation_matrix_2d(15 / 180 * np.pi)
    assert np.isclose(new_v2, R @ v2).all()


def test_align_vectors():
    # Define source and target matrices
    source_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    target_matrix = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]])

    # Call the function
    alignment = align_vectors(source_matrix, target_matrix, pre_normalize=True)

    # Check if the result matrix has the correct shape
    assert tuple(alignment.shape) == (source_matrix.shape[1], target_matrix.shape[1])

    # Check if the result matrix is orthogonal (U @ V should be an identity matrix)
    assert np.allclose(alignment @ alignment.T, np.eye(2), rtol=1e-5)

    # Check if the transformation is a reflection in y=x
    assert np.allclose(alignment, np.array([[0, 1], [1, 0]]), rtol=1e-5)

    # Check if the applied transformation is correct
    assert np.allclose(source_matrix @ alignment, target_matrix, rtol=1e-5)

    # Check for torch tensors
    source_matrix = torch.tensor(source_matrix, dtype=torch.float32)
    target_matrix = torch.tensor(target_matrix, dtype=torch.float32)
    alignment = align_vectors(source_matrix, target_matrix, pre_normalize=True)
    assert tuple(alignment.shape) == (2, 2)
    assert alignment.allclose(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), atol=1e-6)  # type: ignore
