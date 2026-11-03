"""Scoped, coordinate-preserving transformations of selected PyTorch activity."""
from contextlib import contextmanager, ExitStack
from typing import Any, Callable, Iterator


@contextmanager
def transform(provider: Any, targets: Any, apply: Callable[[Any], Any]) -> Iterator[None]:
    import torch
    from .tools import _selection
    from .attribution import _tensor, _replace
    from brainscore.model_helpers.selection import selection_view

    targets = tuple(targets)
    provider.validate('transform', targets)
    if not callable(apply):
        raise TypeError('A transform must supply an apply(tensor) callable')
    seen = set()
    with ExitStack() as stack:
        for target in targets:
            selection = _selection(target)

            def hook(module, inputs, output, selection=selection):
                tensor = _tensor(output)
                view, axis = selection_view(tensor, selection)
                view = torch.movedim(view, axis, -1)
                if selection.indices is None:
                    selected = view
                    indices = None
                else:
                    if not selection.indices or len(set(selection.indices)) != len(selection.indices):
                        raise ValueError('Transform selections require distinct, nonempty unit indices')
                    if max(selection.indices) >= view.shape[-1]:
                        raise ValueError('Selected units are outside the actual activity shape')
                    indices = torch.tensor(selection.indices, device=tensor.device, dtype=torch.long)
                    selected = view.index_select(-1, indices)
                changed = apply(selected.clone())
                if (not isinstance(changed, torch.Tensor) or changed.shape != selected.shape
                        or changed.dtype != tensor.dtype or changed.device != tensor.device):
                    raise ValueError('A transform must preserve selected shape, dtype and device')
                if not torch.isfinite(changed).all():
                    raise ValueError('An activity transform produced nonfinite values')
                result = changed if indices is None else view.clone().index_copy(-1, indices, changed)
                result = torch.movedim(result, -1, axis).reshape(tensor.shape)
                seen.add(selection.layer)
                return _replace(output, result)

            handle = provider.targets[selection.layer].register_forward_hook(hook)
            stack.callback(handle.remove)
        yield
        if seen != {_selection(target).layer for target in targets}:
            raise RuntimeError('An activity transform observed no forward activity; cached-only responses are unsupported')
