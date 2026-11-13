"""Build the ``intervention`` part of a PyTorch subject.

Lives in ``unified`` (not ``core``) because it depends on ``torch``.
:func:`build_pytorch_ablation_fn` turns a torch module into an intervention
part that installs a forward hook to zero, scale or replace units at a layer.
The hook is installed when the subject receives the intervention and removed
when the intervention is undone by its handle, or by ``reset()``.

Usage::

    from brainscore import BrainScoreModel
    from brainscore.perturbation import build_pytorch_ablation_fn
    from brainscore_core import Perturbation, Selection, StateChange
    from brainscore_core.streaming_helpers import apply_state_change

    subject = BrainScoreModel(
        identifier='clip-with-ablation',
        model=clip_model,
        ...,
        intervention=build_pytorch_ablation_fn(clip_model),
    )

    # Apply: zero units 0-99 at the last text-encoder block
    applied = apply_state_change(subject, StateChange(
        kind='ablation',
        target=Selection(layer='text_model.encoder.layers.11',
                         indices=list(range(100))),
        perturbation=Perturbation(kind='zero'),
    ))

    # ... run a session here and measure the changed subject ...

    # Undo this intervention only; subject.reset() undoes all of them.
    apply_state_change(subject, StateChange(kind='reset', handle_id=applied.handle_id))
"""

from typing import Any, Callable, Tuple
import dataclasses
import uuid

import numpy as np

from brainscore.model_helpers.selection import split_site
from brainscore_core.model_interface import (
    Perturbation,
    PerturbationApplied,
    Selection,
    StateChange,
)


def build_pytorch_ablation_fn(model: Any) -> Callable:
    """Build an ``intervention`` part that changes units via a torch forward hook.

    The returned closure understands these perturbation kinds:

    - ``'zero'``: zero out the selected units (or the entire layer if
      ``indices`` is None).
    - ``'scale'``: multiply the selected units by ``perturbation.scale``.
    - ``'drive'``: add ``perturbation.amount``. A Python number is added
      as is. A NumPy array or torch tensor is cast to the activity's dtype
      and device first. With ``indices=None`` the array may be a vector, e.g.
      a residual-space direction, but it must broadcast to exactly the
      activity's shape: it never adds batch, token or feature dimensions.
      With ``indices`` set, the amount must be a scalar (a number or a
      zero-dimensional array).

    Two more kinds reach beyond activations, both undone exactly:

    - ``kind='weights'``: edit a parameter. ``target.layer`` is its path
      (``'layers.3.mlp.down_proj.weight'``); ``indices`` pick slices along
      ``axis``, counted from the end (-1 columns, -2 rows). Undo restores a
      saved copy.
    - ``kind='attention_heads'``: silence or scale heads of an attention module
      (``target.layer``; ``indices`` are head numbers) by editing each head's
      slice of the input to its output projection.

    For ``kind='ablation'`` (the canonical case), the hook is installed on
    the layer named by ``state_change.target.layer``. The layer path is
    resolved by ``getattr``-walking the model: ``'a.b.c'`` becomes
    ``getattr(getattr(getattr(model, 'a'), 'b'), 'c')``. Standard PyTorch
    ``Module.named_modules()`` paths work directly.

    The hook modifies the layer's forward output, so downstream layers see
    the lesioned activations. Suffix the layer with ``':input'`` (e.g.
    ``'model.layers.14.mlp.down_proj:input'``) to modify the layer's first
    input instead (a forward pre-hook) -- needed for values computed inline
    that no module outputs, such as the gated MLP activation entering a
    transformer's ``down_proj``. Recording accepts the same suffix. For multi-tensor
    outputs (some attention layers return tuples), the hook ablates the
    first element and leaves the rest untouched — adequate for typical
    transformer blocks.

    :param model: The torch ``nn.Module`` whose layers will be ablated.
    :returns: An intervention part for ``BrainScoreModel(intervention=...)``.
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

        def _amount_like(tensor):
            if not isinstance(amount, torch.Tensor):
                return amount  # a Python number: unchanged scalar arithmetic
            value = amount.to(device=tensor.device, dtype=tensor.dtype)
            try:
                fits = torch.broadcast_shapes(value.shape, tensor.shape) == tensor.shape
            except RuntimeError:
                fits = False
            if not fits:
                raise ValueError(
                    f"drive amount of shape {tuple(value.shape)} does not broadcast to "
                    f"the activity's shape {tuple(tensor.shape)} without changing it.")
            return value

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
                    return tensor + _amount_like(tensor)
                else:
                    raise ValueError(
                        f"Unsupported Perturbation kind for whole-layer "
                        f"ablation: {kind!r}. Expected 'zero', 'scale', or 'drive'.")
            # Use the same coordinates as recording; never reinterpret indices.
            # Some devices (MPS) silently ignore out-of-range indices; fail everywhere.
            if max(indices) >= view.shape[axis]:
                raise ValueError(
                    f"Selection index {max(indices)} is out of range for axis {axis} of "
                    f"'{selection.layer}' (size {view.shape[axis]}); check the selection's "
                    f"axis and expected_shape match how its units were recorded.")
            modified = view.clone()
            idx = torch.tensor(indices, dtype=torch.long, device=tensor.device)
            if kind == 'zero':
                modified.index_fill_(axis, idx, 0)
            elif kind == 'scale':
                vals = modified.index_select(axis, idx) * scale
                # in-place scatter back
                modified.index_copy_(axis, idx, vals)
            elif kind == 'drive':
                vals = modified.index_select(axis, idx) + _amount_like(tensor)
                modified.index_copy_(axis, idx, vals)
            else:
                raise ValueError(
                    f"Unsupported Perturbation kind for index-ablation: "
                    f"{kind!r}. Expected 'zero', 'scale', or 'drive'.")
            return modified.reshape(tensor.shape)

        return pre_hook if split_site(selection.layer)[1] == 'input' else hook

    def _normalized(state_change: StateChange) -> Perturbation:
        # Arrays become tensors once, here, so a NumPy amount neither promotes
        # the activity's dtype nor breaks gradients or devices at forward time.
        perturbation = state_change.perturbation
        amount = perturbation.amount
        if perturbation.kind != 'drive' or isinstance(amount, (int, float)):
            return perturbation
        if isinstance(amount, (np.ndarray, np.generic)):
            amount = torch.tensor(np.asarray(amount))  # a copy: later edits do not leak in
        elif not isinstance(amount, torch.Tensor):
            raise TypeError(
                f"drive amount must be a number, NumPy array or torch tensor; "
                f"got {type(amount).__name__}.")
        if state_change.target.indices is not None and amount.ndim != 0:
            raise ValueError(
                "drive with Selection.indices takes a scalar amount; a vector amount "
                "is only supported for the whole layer (indices=None).")
        return dataclasses.replace(perturbation, amount=amount)

    def _apply(tensor, perturbation):
        """The perturbation's effect on a tensor (or a slice of one), in place."""
        if perturbation.kind == 'zero':
            tensor.zero_()
        elif perturbation.kind == 'scale':
            tensor.mul_(perturbation.scale)
        elif perturbation.kind == 'drive':
            tensor.add_(torch.as_tensor(perturbation.amount, dtype=tensor.dtype, device=tensor.device))
        elif perturbation.kind == 'replace':
            tensor.copy_(torch.as_tensor(perturbation.replacement, dtype=tensor.dtype, device=tensor.device))
        else:
            raise ValueError(f"Unknown perturbation kind {perturbation.kind!r}")

    def _change_weights(state_change):
        """Edit a parameter: ``target.layer`` is its path (``'layers.3.mlp.down_proj.weight'``),
        ``indices`` pick slices along ``axis``, counted from the end (-1: columns,
        -2: rows of a matrix). Undo restores an exact copy."""
        module_path, _, name = state_change.target.layer.rpartition('.')
        module = _resolve_layer(module_path) if module_path else model
        parameter = getattr(module, name, None)
        if not isinstance(parameter, torch.nn.Parameter):
            raise ValueError(f"{state_change.target.layer!r} is not a parameter of the model.")
        saved = parameter.detach().clone()
        indices, axis = state_change.target.indices, state_change.target.axis
        with torch.no_grad():
            if indices is None:
                _apply(parameter.data, state_change.perturbation)
            else:
                index = torch.tensor(list(indices), device=parameter.device)
                axis = parameter.dim() - 1 if axis in (None, -1) else axis
                part = parameter.data.index_select(axis, index)
                _apply(part, state_change.perturbation)
                parameter.data.index_copy_(axis, index, part)

        def cleanup():
            with torch.no_grad():
                parameter.data.copy_(saved)
        return PerturbationApplied(handle_id=str(uuid.uuid4()), target=state_change.target,
                                   perturbation=state_change.perturbation), cleanup

    def _change_heads(state_change):
        """Silence or scale attention heads: ``target.layer`` is the attention module,
        ``indices`` the heads. Edits each head's slice of the output projection's input."""
        attention = _resolve_layer(state_change.target.layer)
        head_dim = getattr(attention, 'head_dim', None)
        projection = getattr(attention, 'o_proj', None) or getattr(attention, 'out_proj', None)
        if head_dim is None or projection is None:
            raise ValueError(f"{state_change.target.layer!r} needs head_dim and an o_proj or out_proj "
                             "to edit heads; target the projection's input with kind='ablation' instead.")
        heads = list(state_change.target.indices or ())
        if not heads:
            raise ValueError("kind='attention_heads' needs Selection.indices: the heads to change.")

        def hook(_module, args):
            hidden = args[0].clone()
            for head in heads:
                _apply(hidden[..., head * head_dim:(head + 1) * head_dim], state_change.perturbation)
            return (hidden, *args[1:])
        handle = projection.register_forward_pre_hook(hook)
        return PerturbationApplied(handle_id=str(uuid.uuid4()), target=state_change.target,
                                   perturbation=state_change.perturbation), handle.remove

    def intervention(
        state_change: StateChange,
    ) -> Tuple[PerturbationApplied, Callable[[], None]]:
        if state_change.target is None or state_change.perturbation is None:
            raise ValueError(
                f"StateChange(kind={state_change.kind!r}) for "
                f"build_pytorch_ablation_fn requires both target and "
                f"perturbation to be set."
            )
        if state_change.kind == 'weights':
            return _change_weights(state_change)
        if state_change.kind == 'attention_heads':
            return _change_heads(state_change)
        path, site = split_site(state_change.target.layer)
        layer = _resolve_layer(path)
        perturbation = _normalized(state_change)
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
            width = 'in_features' if site == 'input' else 'out_features'
            n_units = (getattr(layer, width, None)
                       if state_change.target.axis == -1 else None)
            if n_units is not None and norm and max(norm) >= n_units:
                raise ValueError(
                    f"Selection index {max(norm)} is out of range for layer "
                    f"'{state_change.target.layer}' with {n_units} units "
                    f"(valid 0..{n_units - 1})."
                )
        hook = _make_hook(state_change.target, perturbation)
        handle = (layer.register_forward_pre_hook(hook) if site == 'input'
                  else layer.register_forward_hook(hook))

        applied = PerturbationApplied(
            handle_id=str(uuid.uuid4()),
            target=state_change.target,
            perturbation=state_change.perturbation,
        )

        def cleanup() -> None:
            handle.remove()

        return applied, cleanup

    return intervention
