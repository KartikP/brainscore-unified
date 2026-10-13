"""Paired-input attribution and exact activity replacement for PyTorch models."""
from contextlib import contextmanager, ExitStack
import copy
import hashlib
import json
import math
import operator
from typing import Any, Callable, Iterator

import numpy as np


def _tensor(output):
    import torch
    value = output[0] if isinstance(output, tuple) and output else output
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.ndim == 0:
        raise TypeError('Activity must be a floating tensor or the first tensor in a tuple')
    return value


def _replace(output, value):
    return (value, *output[1:]) if isinstance(output, tuple) else value


def _location(location):
    if location is None:
        return None
    if not isinstance(location, (tuple, list)) or any(isinstance(i, bool) for i in location):
        raise ValueError('location must contain integer coordinates, excluding the unit axis')
    return tuple(operator.index(i) for i in location)


def _view(tensor, location, indices):
    if location is not None:
        if len(location) != tensor.ndim - 1:
            raise ValueError('location must index every axis except the last unit axis')
        tensor = tensor[location]
    if indices is not None:
        if not indices or len(set(indices)) != len(indices):
            raise ValueError('Select nonempty, distinct unit indices')
        tensor = tensor[..., list(indices)]
    return tensor


def _array(tensor):
    import torch
    value = tensor.detach().cpu()
    if value.dtype in (torch.bfloat16, torch.float16):
        value = value.float()
    result = value.numpy().copy()
    if not np.isfinite(result).all():
        raise ValueError('Activity and attribution values must be finite')
    return result


def reference_digest(reference: dict) -> str:
    """Identify the coordinates, units and values used for an exact patch."""
    header = {key: value for key, value in reference.items() if key != 'values'}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode())
    for name, array in sorted(reference['values'].items()):
        array = np.asarray(array)
        digest.update(json.dumps([name, array.shape, str(array.dtype)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _selections(provider, operation, targets):
    from .tools import _selection
    targets = tuple(targets)
    provider.validate(operation, targets)
    selections = [_selection(target) for target in targets]
    for selection in selections:
        if selection.indices is not None:
            indices = list(selection.indices)
            if not indices or len(set(indices)) != len(indices) or any(isinstance(i, bool) for i in indices):
                raise ValueError('Select nonempty, distinct unit indices')
    return selections


def _capture(provider, selections, call, objective, location, *, gradients):
    import torch
    tensors, values, shapes, dtypes = {}, {}, {}, {}
    with ExitStack() as stack:
        for selection in selections:
            def hook(module, inputs, output, selection=selection):
                name = selection.layer
                if name in tensors:
                    raise ValueError('Each attribution target must run exactly once per call')
                tensor = _tensor(output)
                values[name] = _array(_view(tensor, location, selection.indices))
                shapes[name], dtypes[name] = list(tensor.shape), str(tensor.dtype)
                if gradients:
                    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
                        raise ValueError('Attribution needs a forward call with gradients enabled')
                    # Frozen weights still allow derivatives with respect to activity.
                    if not tensor.requires_grad:
                        tensor = tensor.detach().requires_grad_(True)
                    tensors[name] = tensor
                    # Downstream in-place operations must not rewrite the captured value.
                    return _replace(output, tensor.clone())
                tensors[name] = tensor
            handle = provider.targets[selection.layer].register_forward_hook(hook)
            stack.callback(handle.remove)
        with torch.enable_grad() if gradients else torch.no_grad():
            score = objective(call())
        if set(tensors) != {s.layer for s in selections}:
            raise ValueError('Every attribution target must run exactly once per call')
        if (not isinstance(score, torch.Tensor) or score.ndim != 0
                or not score.is_floating_point() or not torch.isfinite(score)):
            raise ValueError('objective must return one finite scalar tensor')
        if gradients and not score.requires_grad:
            raise ValueError(
                'objective requires gradients; do not use detached outputs or inference-only calls'
            )
        return score, tensors, values, shapes, dtypes


def attribute(
    provider: Any,
    targets: Any,
    *,
    reference_call: Callable[[], Any],
    comparison_call: Callable[[], Any],
    objective: Callable[[Any], Any],
    location: tuple[int, ...] | None = None,
    normalize: bool = True,
) -> dict:
    """Estimate reference-minus-comparison patches using comparison gradients.

    Calls must be deterministic, stateless forwards on the same evaluation-mode
    model. ``objective`` returns the same scalar preference in both calls.
    Gradients are with respect to activity only; parameter gradients are not
    accumulated. Returned arrays are detached snapshots suitable for run records.
    """
    import torch
    selections = _selections(provider, 'attribute', targets)
    location = _location(location)
    if not isinstance(normalize, bool):
        raise TypeError('normalize must be True or False')
    if not all(callable(fn) for fn in (reference_call, comparison_call, objective)):
        raise TypeError('Provide two forward callbacks and a scalar objective')
    if any(module.training for module in provider.targets.values()):
        raise ValueError('Use model.eval() before attribution')
    if torch.is_inference_mode_enabled():
        raise ValueError('Attribution cannot run inside torch.inference_mode()')
    clean, _, clean_values, shapes, dtypes = _capture(
        provider, selections, reference_call, objective, location, gradients=False,
    )
    changed, tensors, changed_values, changed_shapes, changed_dtypes = _capture(
        provider, selections, comparison_call, objective, location, gradients=True,
    )
    if shapes != changed_shapes or dtypes != changed_dtypes:
        raise ValueError('Paired activity shapes and dtypes must match; align inputs explicitly')
    difference = float(clean.detach()) - float(changed.detach())
    if normalize and (not math.isfinite(difference) or abs(difference) <= 1e-8):
        raise ValueError('Cannot normalize an absent or near-zero objective difference')
    gradients = torch.autograd.grad(changed, list(tensors.values()), allow_unused=True)
    attributions, disconnected = {}, []
    for name, gradient in zip(tensors, gradients):
        selection = next(s for s in selections if s.layer == name)
        if gradient is None:
            disconnected.append(name)
            attributions[name] = np.zeros_like(clean_values[name])
            continue
        gradient = _array(_view(gradient, location, selection.indices))
        estimate = (clean_values[name] - changed_values[name]) * gradient
        attributions[name] = estimate / difference if normalize else estimate
        if not np.isfinite(attributions[name]).all():
            raise ValueError('Attribution overflow; choose a stable objective or precision')
    reference = {
        'schema_version': 1,
        'location': location,
        'indices': {s.layer: None if s.indices is None else [int(i) for i in s.indices] for s in selections},
        'shapes': shapes,
        'dtypes': dtypes,
        'values': clean_values,
    }
    return {
        'method': 'attribution_patching',
        'reference_value': float(clean.detach()),
        'comparison_value': float(changed.detach()),
        'normalized': normalize,
        'attributions': attributions,
        'disconnected_targets': disconnected,
        'reference': reference,
    }


def validate_reference(provider: Any, targets: Any, reference: dict) -> None:
    """Check a saved reference before installing any replacement hooks."""
    selections = _selections(provider, 'patch', targets)
    keys = {'schema_version', 'location', 'indices', 'shapes', 'dtypes', 'values'}
    if not isinstance(reference, dict) or set(reference) != keys or reference['schema_version'] != 1:
        raise ValueError('Use a reference returned by TorchInstrumentation.attribute')
    location = _location(reference['location'])
    for selection in selections:
        name = selection.layer
        if any(name not in reference[key] for key in ('indices', 'shapes', 'dtypes', 'values')):
            raise ValueError(f'Reference does not contain {name}')
        shape = reference['shapes'][name]
        if not shape or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape):
            raise ValueError('Reference shapes must have positive integer dimensions')
        if location is not None:
            if len(location) != len(shape) - 1 or any(not -n <= i < n for i, n in zip(location, shape)):
                raise ValueError('Reference location is outside its recorded shape')
        recorded = reference['indices'][name]
        array = np.asarray(reference['values'][name])
        count = shape[-1] if recorded is None else len(recorded)
        expected = tuple(shape[:-1] if location is None else ()) + (count,)
        if array.shape != expected or array.dtype.kind != 'f' or not np.isfinite(array).all():
            raise ValueError('Reference values must be finite floating arrays with the recorded shape')
        population = list(range(shape[-1])) if recorded is None else recorded
        if (not population or len(set(population)) != len(population)
                or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < shape[-1]
                       for i in population)):
            raise ValueError('Reference unit indices are invalid')
        requested = list(range(shape[-1])) if selection.indices is None else list(selection.indices)
        if not set(requested).issubset(population):
            raise ValueError('Requested units were not captured in the reference')


@contextmanager
def patch(provider: Any, targets: Any, reference: dict) -> Iterator[None]:
    """Replace selected outputs on exactly one forward call; always remove hooks."""
    import torch
    selections = _selections(provider, 'patch', targets)
    validate_reference(provider, selections, reference)
    reference = copy.deepcopy(reference)
    location = _location(reference['location'])
    seen = set()
    with ExitStack() as stack:
        for selection in selections:
            def hook(module, inputs, output, selection=selection):
                name = selection.layer
                if name in seen:
                    raise ValueError('Each patch target must run exactly once per scope')
                seen.add(name)
                tensor = _tensor(output)
                if (list(tensor.shape) != reference['shapes'][name]
                        or str(tensor.dtype) != reference['dtypes'][name]):
                    raise ValueError('Patch shape and dtype must match the reference')
                recorded = reference['indices'][name]
                indices = (list(range(tensor.shape[-1])) if selection.indices is None
                           else list(selection.indices))
                if recorded is None:
                    columns = indices
                else:
                    lookup = {unit: column for column, unit in enumerate(recorded)}
                    columns = [lookup[i] for i in indices]
                replacement = torch.as_tensor(
                    reference['values'][name][..., columns], dtype=tensor.dtype, device=tensor.device,
                )
                result = tensor.clone()
                view = result if location is None else result[location]
                view[..., indices] = replacement
                return _replace(output, result)
            handle = provider.targets[selection.layer].register_forward_hook(hook)
            stack.callback(handle.remove)
        yield
        if seen != {s.layer for s in selections}:
            raise ValueError('Every patch target must run exactly once per scope')
