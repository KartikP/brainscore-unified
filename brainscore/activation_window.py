"""ActivationWindow — capture layer-output activations during any process() loop.

`PerceptWindow` taps a model's INPUT (a `forward_pre_hook`); `ActivationWindow`
taps its LAYER OUTPUTS (a `forward_hook`), so you can collect hidden activations
while the model runs — crucially, **including inside an embodied or multi-agent
rollout**, where `process(EnvironmentStep)` dispatches to `action_fn` and never
touches the `start_recording → NeuroidAssembly` path. It is non-invasive (hooks,
no extra forward pass), strided/capped, and decoupled from the dispatch path:
the hook fires whenever the layer's `forward()` runs.

    with ActivationWindow(agent, layers=['blocks.20']) as rec:
        run_interaction(agent, ...)          # each process() forward fires the hooks
    acts = rec.stack('blocks.20', reduce='mean')   # (n_calls, hidden) — one vector per tick

`target` may be a torch Module, an activations wrapper (`._model`), or a
BrainScoreModel (its wrapped module is resolved automatically — same logic as
PerceptWindow). `layers` are dotted submodule paths; omit to hook the outermost
module's output.
"""
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .percept_window import _resolve_targets


def _first_tensor_out(out):
    """The activation tensor from a layer's output — handles a bare Tensor, a
    tuple/list (many transformer blocks return tuples), or an HF-style object
    exposing ``last_hidden_state`` / ``hidden_states``."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            if isinstance(o, torch.Tensor):
                return o
    for attr in ('last_hidden_state', 'hidden_states'):
        v = getattr(out, attr, None)
        if isinstance(v, torch.Tensor):
            return v
        if isinstance(v, (tuple, list)) and v and isinstance(v[-1], torch.Tensor):
            return v[-1]
    return None


def _get_submodule(root, path):
    try:
        return root.get_submodule(path)
    except Exception:
        return dict(root.named_modules()).get(path)


@dataclass
class ActivationCapture:
    """One captured layer output."""
    index: int
    layer: str
    call: int                       # per-layer call counter (aligns to ticks)
    tensor: Any                     # torch.Tensor (detached, CPU)
    shape: Tuple[int, ...] = ()
    dtype: str = ''


class ActivationWindow:
    def __init__(self, target, layers: Optional[List[str]] = None, every: int = 1,
                 max_captures: int = 1024, select_output: Optional[Callable] = None,
                 label: Optional[str] = None):
        roots = _resolve_targets(target)
        if not roots:
            raise ValueError("ActivationWindow found no torch module on the target.")
        self.label = label or 'activations'
        self.every = max(1, int(every))
        self.max_captures = max_captures
        self.select_output = select_output or _first_tensor_out
        self.targets: List[Tuple[str, torch.nn.Module]] = []
        for rootname, root in roots:
            if layers:
                for lp in layers:
                    sub = _get_submodule(root, lp)
                    if sub is None:
                        raise ValueError(
                            f"layer {lp!r} not found under {rootname}; available e.g. "
                            f"{list(dict(root.named_modules()).keys())[:8]}")
                    name = f'{rootname}:{lp}' if len(roots) > 1 else lp
                    self.targets.append((name, sub))
            else:
                self.targets.append((rootname, root))
        self.captures: List[ActivationCapture] = []
        self._handles: List[Any] = []
        self._calls: Dict[str, int] = {}

    def __enter__(self):
        def make_hook(label):
            self._calls.setdefault(label, 0)

            def hook(module, inp, out):
                self._calls[label] += 1
                if (self._calls[label] - 1) % self.every != 0:
                    return
                if len(self.captures) >= self.max_captures:
                    return
                t = self.select_output(out)
                if not isinstance(t, torch.Tensor):
                    return
                t = t.detach().cpu()
                self.captures.append(ActivationCapture(
                    index=len(self.captures), layer=label, call=self._calls[label] - 1,
                    tensor=t, shape=tuple(t.shape), dtype=str(t.dtype)))
            return hook

        for label, mod in self.targets:
            self._handles.append(mod.register_forward_hook(make_hook(label)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    # -- access -----------------------------------------------------------
    def by_layer(self) -> Dict[str, List[Any]]:
        """dict ``layer -> [activation array per call, in order]``."""
        out: Dict[str, List[Any]] = {}
        for c in self.captures:
            out.setdefault(c.layer, []).append(c.tensor.numpy())
        return out

    def stack(self, layer: str, reduce: Optional[str] = None):
        """Stack a layer's per-call activations into one array of shape
        ``(n_calls, *feature)``. ``reduce='mean'`` first mean-pools each call over
        all but the last (feature) dim — useful when seq-length varies tick to
        tick, and the form you want for inter-agent RSA."""
        arrs = self.by_layer().get(layer)
        if not arrs:
            raise KeyError(f"no captures for layer {layer!r}")
        if reduce == 'mean':
            arrs = [a.reshape(-1, a.shape[-1]).mean(0) for a in arrs]
        shapes = {a.shape for a in arrs}
        if len(shapes) != 1:
            raise ValueError(
                f"layer {layer!r} has heterogeneous shapes {shapes}; pass "
                f"reduce='mean' to pool each call to a per-call feature vector")
        return np.stack(arrs)

    def summary(self) -> Dict[str, Any]:
        by: Dict[str, int] = {}
        for c in self.captures:
            by[c.layer] = by.get(c.layer, 0) + 1
        return {'label': self.label, 'n_captures': len(self.captures), 'by_layer': by}

    def save(self, out_dir: str) -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)
        man: Dict[str, Any] = {'summary': self.summary(), 'layers': {}}
        for layer, arrs in self.by_layer().items():
            safe = layer.replace('/', '_').replace(':', '_').replace('.', '_')
            try:
                arr = self.stack(layer)
                p = os.path.join(out_dir, f'act_{safe}.npy')
                np.save(p, arr)
                man['layers'][layer] = {'file': p, 'shape': list(arr.shape)}
            except ValueError:                          # ragged across calls
                p = os.path.join(out_dir, f'act_{safe}.npz')
                np.savez(p, *arrs)
                man['layers'][layer] = {'file': p, 'n': len(arrs), 'ragged': True}
        with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
            json.dump(man, f, indent=2, default=str)
        return man
