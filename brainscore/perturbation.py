"""Helpers for building ``state_change_fn`` callables.

Lives in ``unified`` (not ``core``) because the implementations depend on
``torch``. ``brainscore_core`` itself stays dependency-light per project
convention.

The single helper today is :func:`build_pytorch_ablation_fn`, which converts
a torch module into a state_change_fn that installs a forward-hook to zero
out (or scale) units at a specified layer. The hook is registered when
``process(StateChange)`` is called and removed when ``model.reset()`` is
called or the returned cleanup callable is invoked directly.

Usage::

    from brainscore.perturbation import build_pytorch_ablation_fn
    from brainscore_core.model_interface import (
        BrainScoreModel, StateChange, Selection, Perturbation,
    )

    bs_model = BrainScoreModel(
        identifier='clip-with-ablation',
        model=clip_model,
        ...,
        state_change_fn=build_pytorch_ablation_fn(clip_model),
    )

    # Apply: zero units 0-99 at the last text-encoder block
    applied = bs_model.process(StateChange(
        kind='ablation',
        target=Selection(layer='text_model.encoder.layers.11',
                         indices=list(range(100))),
        perturbation=Perturbation(kind='zero'),
    ))

    # ... run process(stim) here, observe the lesioned behavior ...

    bs_model.reset()  # forward hook removed; model back to baseline
"""

from typing import Any, Callable, Tuple
import uuid

from brainscore_core.model_interface import (
    Perturbation,
    PerturbationApplied,
    Selection,
    StateChange,
)


def build_pytorch_ablation_fn(model: Any) -> Callable:
    """Build a ``state_change_fn`` that ablates units via a torch forward-hook.

    The returned closure understands two perturbation kinds:

    - ``'zero'``: zero out the selected units (or the entire layer if
      ``indices`` is None).
    - ``'scale'``: multiply the selected units by ``perturbation.scale``.

    For ``kind='ablation'`` (the canonical case), the hook is installed on
    the layer named by ``state_change.target.layer``. The layer path is
    resolved by ``getattr``-walking the model: ``'a.b.c'`` becomes
    ``getattr(getattr(getattr(model, 'a'), 'b'), 'c')``. Standard PyTorch
    ``Module.named_modules()`` paths work directly.

    The hook is a pre-output hook: it modifies the forward output, so
    downstream layers see the lesioned activations. For multi-tensor
    outputs (some attention layers return tuples), the hook ablates the
    first element and leaves the rest untouched — adequate for typical
    transformer blocks.

    :param model: The torch ``nn.Module`` whose layers will be ablated.
    :returns: A ``state_change_fn`` suitable for
        ``BrainScoreModel(state_change_fn=...)``.
    """
    import torch  # local import: keeps this module's import-time cheap

    def _resolve_layer(layer_path: str) -> torch.nn.Module:
        layer = model
        for part in layer_path.split('.'):
            try:
                idx = int(part)
                layer = layer[idx]
            except (ValueError, TypeError):
                if not hasattr(layer, part):
                    raise AttributeError(
                        f"Cannot resolve layer path {layer_path!r} on model "
                        f"{type(model).__name__}: missing attribute {part!r} "
                        f"on {type(layer).__name__}."
                    )
                layer = getattr(layer, part)
        return layer

    def _make_hook(selection: Selection, perturbation: Perturbation):
        indices = selection.indices
        kind = perturbation.kind
        scale = perturbation.scale

        def hook(_module, _inputs, output):
            # Some layers return tuples (e.g., MultiheadAttention).
            # We modify the leading tensor only; the rest pass through.
            if isinstance(output, tuple):
                head, *rest = output
                head = _apply(head)
                return (head, *rest)
            return _apply(output)

        def _apply(tensor):
            if not isinstance(tensor, torch.Tensor):
                return tensor  # don't touch non-tensors
            if indices is None:
                # ablate the whole layer
                if kind == 'zero':
                    return torch.zeros_like(tensor)
                elif kind == 'scale':
                    return tensor * scale
                else:
                    raise ValueError(
                        f"Unsupported Perturbation kind for whole-layer "
                        f"ablation: {kind!r}. Expected 'zero' or 'scale'.")
            # ablate at the last (feature) axis at the named indices
            modified = tensor.clone()
            idx = torch.tensor(indices, dtype=torch.long, device=tensor.device)
            if kind == 'zero':
                modified.index_fill_(-1, idx, 0)
            elif kind == 'scale':
                vals = modified.index_select(-1, idx) * scale
                # in-place scatter back
                modified.index_copy_(-1, idx, vals)
            else:
                raise ValueError(
                    f"Unsupported Perturbation kind for index-ablation: "
                    f"{kind!r}. Expected 'zero' or 'scale'.")
            return modified

        return hook

    def state_change_fn(
        state_change: StateChange,
    ) -> Tuple[PerturbationApplied, Callable[[], None]]:
        if state_change.target is None or state_change.perturbation is None:
            raise ValueError(
                f"StateChange(kind={state_change.kind!r}) for "
                f"build_pytorch_ablation_fn requires both target and "
                f"perturbation to be set."
            )
        layer = _resolve_layer(state_change.target.layer)
        hook = _make_hook(state_change.target, state_change.perturbation)
        handle = layer.register_forward_hook(hook)

        applied = PerturbationApplied(
            handle_id=str(uuid.uuid4()),
            target=state_change.target,
            perturbation=state_change.perturbation,
        )

        def cleanup() -> None:
            handle.remove()

        return applied, cleanup

    return state_change_fn
