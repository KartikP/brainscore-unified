"""Tensor coordinates shared by activity recording and intervention."""
from typing import Any
from brainscore_core.events import Selection


def selection_view(tensor: Any, selection: Selection) -> tuple[Any, int]:
    """Return the selected coordinate system before indexing units.

    None flattens non-batch axes. An integer preserves the original tensor.
    The caller must reshape a modified flat view back to the original shape.
    """
    if tensor.ndim == 0 or (tensor.ndim < 2 and selection.axis is None):
        raise ValueError('Unit selection requires a batch axis and a feature axis.')
    if selection.expected_shape is not None and tuple(tensor.shape[1:]) != selection.expected_shape:
        raise ValueError(f'Selection expected non-batch shape {selection.expected_shape}, '
                         f'got {tuple(tensor.shape[1:])}; recording and intervention layouts differ.')
    if selection.axis is None:
        return tensor.reshape(tensor.shape[0], -1), -1
    axis = selection.axis
    if not -tensor.ndim <= axis < tensor.ndim or (tensor.ndim > 1 and axis % tensor.ndim == 0):
        raise ValueError('Selection axis must address a non-batch tensor dimension.')
    return tensor, axis
