import numpy as np
import torch

from sign_language_translator.utils.arrays import (
    ArrayOps,
    linear_interpolation,
    adjust_vector_angle,
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
