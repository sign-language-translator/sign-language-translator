import numpy as np
import torch

from sign_language_translator.utils import ArrayOps, linear_interpolation


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
    assert new_array.isclose(
        torch.Tensor(  # type: ignore
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

    ArrayOps.floor(tensor)
    ArrayOps.floor(array)
