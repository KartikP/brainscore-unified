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

    The hook modifies the layer's forward output, so downstream layers see
    the lesioned activations. Set ``Selection.metadata['site'] = 'input'`` to
    modify the layer's first input instead (a forward pre-hook) -- needed for
    values computed inline that no module outputs, such as the gated MLP
    activation entering a transformer's ``down_proj``. For multi-tensor
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
        amount = perturbation.amount

        def pre_hook(_module, args):
            # site='input': perturb the first positional input, e.g. the gated
            # MLP activation entering down_proj, which no module outputs.
            return (_apply(args[0]), *args[1:])

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
            from brainscore.model_helpers.selection import selection_view
            view, axis = selection_view(tensor, selection)
            if indices is None:
                # ablate the whole layer
                if kind == 'zero':
                    return torch.zeros_like(tensor)
                elif kind == 'scale':
                    return tensor * scale
                elif kind == 'drive':
                    return tensor + amount
                else:
                    raise ValueError(
                        f"Unsupported Perturbation kind for whole-layer "
                        f"ablation: {kind!r}. Expected 'zero', 'scale', or 'drive'.")
            # Use the same coordinates as recording; never reinterpret indices.
            modified = view.clone()
            idx = torch.tensor(indices, dtype=torch.long, device=tensor.device)
            if kind == 'zero':
                modified.index_fill_(axis, idx, 0)
            elif kind == 'scale':
                vals = modified.index_select(axis, idx) * scale
                # in-place scatter back
                modified.index_copy_(axis, idx, vals)
            elif kind == 'drive':
                vals = modified.index_select(axis, idx) + amount
                modified.index_copy_(axis, idx, vals)
            else:
                raise ValueError(
                    f"Unsupported Perturbation kind for index-ablation: "
                    f"{kind!r}. Expected 'zero', 'scale', or 'drive'.")
            return modified.reshape(tensor.shape)

        return pre_hook if selection.metadata.get('site') == 'input' else hook

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
        # Validate indices at apply-time, not cryptically mid-forward-pass.
        indices = state_change.target.indices
        if indices is not None:
            import operator
            norm = []
            for i in indices:
                try:
                    ii = operator.index(i)  # accepts int + numpy ints, rejects float/str
                except TypeError:
                    raise ValueError(
                        f"Selection.indices must be integers, got {i!r} "
                        f"({type(i).__name__})."
                    )
                if ii < 0:
                    raise ValueError(
                        f"Selection.indices must be non-negative, got {ii}."
                    )
                norm.append(ii)
            # Index perturbation targets the LAST output axis: the feature axis
            # for Linear and (N, L, D) transformer-block outputs. It is NOT the
            # channel axis for conv (out_channels is axis 1, not last), so we do
            # not range-check conv here — a conv channel index would otherwise be
            # validated against the wrong axis. Layers with no last-axis width
            # attr are validated at forward time.
            width = ('in_features' if state_change.target.metadata.get('site') == 'input'
                     else 'out_features')
            n_units = (getattr(layer, width, None)
                       if state_change.target.axis == -1 else None)
            if n_units is not None and norm and max(norm) >= n_units:
                raise ValueError(
                    f"Selection index {max(norm)} is out of range for layer "
                    f"'{state_change.target.layer}' with {n_units} units "
                    f"(valid 0..{n_units - 1})."
                )
        hook = _make_hook(state_change.target, state_change.perturbation)
        if state_change.target.metadata.get('site', 'output') not in ('input', 'output'):
            raise ValueError(f"Selection.metadata['site'] must be 'input' or 'output', "
                             f"got {state_change.target.metadata['site']!r}.")
        handle = (layer.register_forward_pre_hook(hook)
                  if state_change.target.metadata.get('site') == 'input'
                  else layer.register_forward_hook(hook))

        applied = PerturbationApplied(
            handle_id=str(uuid.uuid4()),
            target=state_change.target,
            perturbation=state_change.perturbation,
        )

        def cleanup() -> None:
            handle.remove()

        return applied, cleanup

    return state_change_fn
