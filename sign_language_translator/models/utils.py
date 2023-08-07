"""Utility functions and classes for PyTorch models.

This module contains various utility functions and classes to assist with PyTorch models and their training.

Functions:
    top_p_top_k_indexes(probabilities, top_p=None, top_k=None):
        Perform top-p (nucleus) and top-k filtering based on the given probabilities.

    plot_lr_scheduler(*args, lr_scheduler_class=None, lr_scheduler_object=None, initial_lr=1e-3, n_steps=20, parameter_group_number=0, **kwargs):
        Plot the learning rate of a specific parameter group across training steps.

    downwards_wave(n_waves, n_steps_per_wave=9, start=1e-3, end=1e-7, amplitude=0.25):
        Generate a downwards wave pattern with a combination of sine wave and linear function.

    set_layers_trainability_(model, layers_to_unfreeze=None, layers_to_freeze=None):
        Set the trainability of specified layers in the given PyTorch model.

Classes:
    FullyLambdaLR(torch.optim.lr_scheduler.LRScheduler):
        Sets the learning rate of each parameter group to a given function that takes step_num, base_lr & last_lr as args.

Example usage:
    See the docstrings of individual functions and classes for usage examples.
"""


from typing import Callable, Iterable, List, Type

import numpy
import torch


def top_p_top_k_indexes(
    probabilities: Iterable[float],
    top_p: float | None = None,
    top_k: int | None = None,
) -> List[int]:
    """Perform top-p (nucleus) and top-k filtering based on the given probabilities.
    Top-k returns the indices of the top-k elements.
    Top-p returns the indices of the top elements whose sum does not exceed a certain value.

    Args:
        probs (Iterable[float]): A 1-D iterable containing the probabilities of each element. Probabilities must sum to 1.
        top_p (float or None): The threshold probability for top-p sampling (0 to 1). If None, top-p sampling is not applied.
        top_k (int or None): The maximum number of elements to keep for top-k sampling. If None, top-k sampling is not applied.

    Returns:
        torch.Tensor: The indices of the selected elements.

    Example usage:
        >>> selected_indices = top_p_top_k_indexes(
        >>>    probabilities=[0.1, 0.2, 0.15, 0.05, 0.3, 0.2],
        >>>    top_p=0.75,
        >>>    top_k=3,
        >>> )
        [4, 1, 5]

        You can then use the `selected_indices` to gather the actual elements from the original tensor.
        >>> sampled_elements = probs[selected_indices]
        [0.3, 0.2, 0.2]
    """

    if top_p is None and top_k is None:
        return list(range(len(probabilities)))  # type: ignore

    probs = torch.Tensor(probabilities)

    assert probs.dim() == 1, "The input probabilities iterable must be 1-dimensional."
    assert torch.allclose(  # pylint: disable = no-member
        probs.sum(), torch.Tensor([1])
    ), "probabilities must sum to 1."
    assert top_p is None or 0 <= top_p <= 1, "top_p must be between 0 and 1, or None."
    assert top_k is None or top_k > 0, "top_k must be greater than 0, or None."

    sorted_probs, sorted_indices = torch.sort(  # pylint: disable = no-member
        probs, descending=True
    )

    cumulative_probs = torch.cumsum(sorted_probs, dim=0)  # pylint: disable = no-member

    if top_p is not None:
        # Find indices of the elements whose cumulative probability is below top_p
        sorted_indices_to_remove_mask = cumulative_probs > top_p
        # [False, False, ..., True, True, ...] -> True means the probability too low and is below top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove_mask[1:] = sorted_indices_to_remove_mask[:-1].clone()
        # if the first element itself has probability above top_p, shifting step wouldn't help it from removal.
        sorted_indices_to_remove_mask[0] = False
        # Zero out the probabilities of elements whose index is marked for removal
        sorted_probs[sorted_indices_to_remove_mask] = 0.0

    if top_k is not None:
        # Zero out the probabilities of elements beyond top_k
        sorted_probs[top_k:] = 0.0

    # Get the indices of the selected elements
    selected_indices = sorted_indices[sorted_probs > 0]

    return selected_indices.tolist()


class FullyLambdaLR(torch.optim.lr_scheduler.LRScheduler):
    """Sets the learning rate of each parameter group to a given function
    that takes step_num, base_lr, and last_lr as parameters.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function): A function which computes the learning rate
            given an integer parameter step_num, initial learning rate and previous learning rate
            for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> scheduler = FullyLambdaLR(
        >>>    optimizer,
        >>>    lambda step_num, base_lr, last_lr: last_lr * (1.08 if step_num%2 == 0 else 0.8)
        >>> )
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
        >>>     print(scheduler.get_last_lr()[0])
    """

    def __init__(
        self,
        optimizer,
        lr_lambda: Callable[[int, float, float], float],
        last_epoch=-1,
        verbose=False,
    ):
        self.optimizer = optimizer
        self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [
            lmbda(self.last_epoch, base_lr, group["lr"])
            for lmbda, base_lr, group in zip(
                self.lr_lambdas,
                self.base_lrs,
                self.optimizer.param_groups,
            )
        ]


def plot_lr_scheduler(
    *args,
    lr_scheduler_class: Type[torch.optim.lr_scheduler.LRScheduler] | None = None,
    # lr_lambda: Callable[[Any], float] | None = None,
    lr_scheduler_object: torch.optim.lr_scheduler.LRScheduler | None = None,
    initial_lr: float = 1e-3,
    n_steps: int = 20,
    parameter_group_number: int = 0,
    **kwargs,
):
    """Plot the learning rate of a specific parameter group across training steps.

    This function generates a plot to visualize how the learning rate of a specified
    parameter group changes across training steps. It requires either an existing `lr_scheduler_object`
    or a combination of `lr_scheduler_class`, `args`, and `kwargs` to create a new learning rate scheduler.

    Args:
        lr_scheduler_class (Type[torch.optim.lr_scheduler._LRScheduler], optional):
            The class of the learning rate scheduler. Defaults to None.
        lr_scheduler_object (torch.optim.lr_scheduler._LRScheduler, optional):
            An existing object of a learning rate scheduler class. Defaults to None.
        initial_lr (float, optional): The initial learning rate for the new optimizer object
            needed in case lr_scheduler_object is None. Defaults to 1e-3.
        n_steps (int, optional): The number of epochs/steps to visualize the learning rate changes. Defaults to 20.
        parameter_group_number (int, optional): The index for the optimizer's parameter group to plot the learning rate for. Defaults to 0.
        *args: Additional arguments to pass to the lr_scheduler_class when creating a new scheduler.
        **kwargs: Additional keyword arguments to pass to the lr_scheduler_class when creating a new scheduler.

    Example:
        ```
        # Using an existing learning rate scheduler object
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        plot_lr_scheduler(lr_scheduler_object=lr_scheduler, n_steps=100)

        # Creating a new learning rate scheduler object
        plot_lr_scheduler(
            lr_scheduler_class=torch.optim.lr_scheduler.ExponentialLR,
            initial_lr=0.01,
            gamma=0.9,
            n_steps=50,
        )
        ```
    """

    import matplotlib.pyplot as plt

    optimizer = None
    if lr_scheduler_object is None:

        class TemporaryModel(torch.nn.Module):
            def __init__(self):
                super(TemporaryModel, self).__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = TemporaryModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
        if lr_scheduler_class is None:
            lr_scheduler_class = torch.optim.lr_scheduler.LambdaLR
        lr_scheduler_object = lr_scheduler_class(optimizer, *args, **kwargs)  # type: ignore

    x_values = list(range(n_steps))
    y_values = []
    for _ in x_values:
        y_values.append(lr_scheduler_object.get_lr()[parameter_group_number])  # type: ignore
        if optimizer:
            optimizer.step()
        lr_scheduler_object.step()

    plt.plot(
        x_values,
        y_values,
        ".-" if len(x_values) < 50 else "-",
        label=f"lr, param_group_no:{parameter_group_number}",
    )
    plt.xlabel("steps")
    plt.legend()
    plt.show()
    plt.close()


def downwards_wave(
    n_waves: int,
    n_steps_per_wave: int = 9,
    start: float = 1e-3,
    end: float = 1e-7,
    amplitude: float = 0.25,
) -> numpy.ndarray:
    """
    Generate a downwards wave pattern with a combination of sine wave and linear function.

    The function generates a sequence of points forming a downward wave pattern, which
    consists of a combination of a sine wave and a linear function. The sine wave is
    modulated by the linear function to create a gradual decrease in axis of the
    waves.

    Parameters:
        n_waves (int): Number of peaks/cycles to generate.
        n_steps_per_wave (int, optional): Number of steps per wave. Default is 9.
        start (float, optional): Starting value for the linear function. Default is 1e-3.
        end (float, optional): Ending value for the linear function. Default is 1e-7.
        amplitude (float, optional): Amplitude of the sine wave. Default is 0.25.

    Returns:
        numpy.ndarray: Array containing the y-axis values of downwards wave pattern.
    """

    x = numpy.linspace(0, n_waves, n_waves * n_steps_per_wave)

    downwards_line = numpy.linspace(start, end, len(x))

    wave = numpy.sin(x * 2 * numpy.pi - numpy.pi / 2) + 1
    wave *= abs(start - end) / n_waves * amplitude

    final = wave + downwards_line
    return final


def set_layers_trainability_(
    model: torch.nn.Module,
    layers_to_unfreeze: List[str] | None = None,
    layers_to_freeze: List[str] | None = None,
):
    """
    Set the trainability of specified layers in the given PyTorch model.

    This function allows you to selectively freeze or unfreeze specific layers of a PyTorch model by setting their
    `requires_grad` attribute accordingly.

    Args:
        model (torch.nn.Module): The PyTorch model whose layers' requires_grad will be modified.
        layers_to_unfreeze (List[str] | None, optional): A list of layer names or prefixes for layers
            that you want to unfreeze. If None, no layers will be unfrozen. If [""], all layers will be unfrozen. Default is None.
        layers_to_freeze (List[str] | None, optional): A list of layer names or prefixes for layers
            that you want to freeze. If None, no layers will be frozen. If [""], all layers will be frozen. Default is None.

    Returns:
        None: This function modifies the model in-place. It does not return anything.

    Note:
        - If both `layers_to_unfreeze` and `layers_to_freeze` are None or empty, no action will be taken,
          and the function will return immediately.
        - The layers' names or prefixes specified in the lists should match the names as returned by
          `model.named_parameters()`.

    Examples:
        1. To freeze all layers in the model:
        set_layers_trainability_(model, layers_to_freeze=[""])

        2. To unfreeze the layers with names starting with 'classifier' and 'fc':
        set_layers_trainability_(model, layers_to_unfreeze=["classifier", "fc"])

        3. To unfreeze all layers:
        set_layers_trainability_(model, layers_to_unfreeze=[""])
    """

    if not (layers_to_unfreeze or layers_to_freeze):
        return

    for layer_name, param in model.named_parameters():
        for given_name in layers_to_freeze or []:
            if layer_name.startswith(given_name):
                param.requires_grad = False
        for given_name in layers_to_unfreeze or []:
            if layer_name.startswith(given_name):
                param.requires_grad = True


__all__ = [
    "top_p_top_k_indexes",
    "FullyLambdaLR",
    "plot_lr_scheduler",
    "downwards_wave",
    "set_layers_trainability_",
]
