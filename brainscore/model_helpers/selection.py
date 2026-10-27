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


INPUT_SITE = ':input'


def split_site(layer: str) -> tuple[str, str]:
    """``'a.b:input'`` -> ``('a.b', 'input')``; a bare path is the module output.

    The input site addresses a module's first positional input, for values
    computed inline that no module outputs (e.g. the gated MLP activation
    entering a transformer's ``down_proj``). Recording and intervention both
    parse layer names here, so one string names the same tensor in both.
    """
    if layer.endswith(INPUT_SITE):
        return layer[:-len(INPUT_SITE)], 'input'
    if ':' in layer:
        raise ValueError(f"Unknown layer site in {layer!r}; only '{INPUT_SITE}' is supported.")
    return layer, 'output'
