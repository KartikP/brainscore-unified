"""PerceptWindow — reconstruct what the model *actually* saw.

The Witness records what was *presented* (a stimulus path, a montage, a frame)
because that is all that exists at the ``process()`` seam — preprocessing
(resize / crop / normalize / tokenize / resample) happens one layer deeper,
inside the activations wrapper, right before the torch ``forward()``. The
PerceptWindow taps *that* point: a ``register_forward_pre_hook`` on the wrapped
module captures the exact tensor the network ingested, and reconstruction
inverts the preprocessing into a viewable artifact.

This is the model's literal percept — it shows the aspect-ratio squish, the
borders the crop discarded, a wrong-normalization color cast, a transcript
truncated at ``max_length``. It is strictly more faithful than the presented
stimulus, and it costs **zero extra forward passes**: the hook tees the tensor
that the model's own forward already computed.

    from brainscore.percept_window import PerceptWindow
    with PerceptWindow(model, denorm='auto') as eye:   # any model, any modality
        benchmark(model)
    eye.save('/tmp/percept')      # reconstructed images / decoded text / waveforms

Reconstruction is modality-specific:
  * vision — de-normalize ``x*std + mean`` (exact, given mean/std)
  * text   — ``tokenizer.decode(ids)`` (exact, incl. special tokens / truncation)
  * audio  — the resampled waveform (raw-waveform models); a spectrogram image
             for spectrogram models (sound inversion would be lossy)

Heavy renderers (PIL / soundfile / matplotlib) are imported lazily in ``save``;
the recorder and the pure reconstruction functions need only numpy + torch.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ── Pure reconstruction functions (no model, no hooks — unit-testable) ──

def infer_modality(tensor: torch.Tensor) -> str:
    """Best-effort guess of what a captured input tensor represents.

    Integer tensor -> 'text' (token ids); float with a channel-shaped 3rd-from-
    last dim (1 or 3) -> 'vision'; otherwise a 1-2D float -> 'audio'. Heuristic;
    pass an explicit ``modality`` to PerceptWindow to override.
    """
    if tensor.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8, torch.long):
        return 'text'
    nd = tensor.dim()
    if nd >= 3 and tensor.shape[-3] in (1, 3):
        return 'vision'
    if nd <= 2:
        return 'audio'
    return 'unknown'


def denormalize_image(tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Invert ``(img - mean) / std`` to recover viewable uint8 images.

    Accepts ``(C, H, W)`` or ``(B, C, H, W)`` (numpy or torch). Returns
    ``(B, H, W, C)`` uint8 — exactly the resized/cropped image the model saw
    (cropped-away borders are not recoverable; the model didn't see them either).
    """
    arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    arr = arr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[None]                              # -> (1, C, H, W)
    if arr.ndim != 4:
        raise ValueError(f"expected (C,H,W) or (B,C,H,W), got shape {arr.shape}")
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)
    if mean.shape[1] != arr.shape[1] or std.shape[1] != arr.shape[1]:
        raise ValueError(
            f"mean/std length ({mean.shape[1]}/{std.shape[1]}) must match the "
            f"channel count ({arr.shape[1]})")
    img = arr * std + mean                           # invert normalization
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).round().astype(np.uint8)
    return np.transpose(img, (0, 2, 3, 1))           # (B, H, W, C)


def minmax_image(tensor) -> np.ndarray:
    """Per-image min-max visualization when mean/std are unknown — NOT a faithful
    color reconstruction (flagged as such in the manifest), just a way to see
    structure. Same I/O shape as :func:`denormalize_image`."""
    arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    arr = arr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[None]
    out = np.empty_like(arr)
    for b in range(arr.shape[0]):
        x = arr[b]
        lo, hi = float(x.min()), float(x.max())
        out[b] = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    out = (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
    return np.transpose(out, (0, 2, 3, 1))


def decode_tokens(ids, tokenizer, skip_special_tokens: bool = False) -> List[str]:
    """Detokenize captured input ids back to strings — exactly what the model
    read, including special tokens and any truncation. Accepts ``(T,)`` or
    ``(B, T)``."""
    arr = ids.detach().cpu().numpy() if isinstance(ids, torch.Tensor) else np.asarray(ids)
    if arr.ndim == 1:
        arr = arr[None]
    return [tokenizer.decode(list(map(int, row)), skip_special_tokens=skip_special_tokens)
            for row in arr]


def as_waveform(tensor) -> np.ndarray:
    """Captured audio input as ``(B, samples)`` float — the resampled/truncated
    waveform the model ingested. Accepts ``(S,)``, ``(B, S)``, or ``(B, 1, S)``."""
    arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    return arr


def find_normalization(obj: Any) -> Optional[Tuple[List[float], List[float]]]:
    """Best-effort introspection of vision normalization constants.

    Handles HuggingFace image processors (``image_mean`` / ``image_std``) and
    torchvision ``Compose`` pipelines containing a ``Normalize``. Returns
    ``(mean, std)`` or ``None`` if it can't be found — in which case the caller
    falls back to a (flagged) min-max visualization.
    """
    if obj is None:
        return None
    mean = getattr(obj, 'image_mean', None)
    std = getattr(obj, 'image_std', None)
    if mean is not None and std is not None:
        return list(map(float, mean)), list(map(float, std))
    # torchvision Compose: look for a Normalize transform
    transforms = getattr(obj, 'transforms', None)
    if transforms:
        for t in transforms:
            m, s = getattr(t, 'mean', None), getattr(t, 'std', None)
            if m is not None and s is not None:
                return list(map(float, m)), list(map(float, s))
    return None


# ── Capture record + target resolution ─────────────────────────────────

@dataclass
class PerceptCapture:
    """One captured input tensor entering a hooked module."""
    index: int
    module: str
    modality: str
    tensor: Any                                      # torch.Tensor (detached, CPU)
    shape: Tuple[int, ...] = ()
    dtype: str = ''


def _first_tensor(args, kwargs):
    """The first torch.Tensor among positional args then kwarg values — handles
    both ``forward(x)`` and ``forward(pixel_values=x)`` / ``forward(input_ids=x)``."""
    for a in args:
        if isinstance(a, torch.Tensor):
            return a
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            return v
    return None


def _resolve_targets(target) -> List[Tuple[str, torch.nn.Module]]:
    """Resolve a PerceptWindow target into ``[(name, module), ...]``.

    Accepts a torch Module, a list of Modules, a wrapper exposing ``._model``,
    or a BrainScoreModel (taps ``_activations_model._model`` plus any
    wrapper-valued preprocessors). De-duplicates by module identity.
    """
    found: List[Tuple[str, torch.nn.Module]] = []
    seen: set = set()

    def add(name, mod):
        if isinstance(mod, torch.nn.Module) and id(mod) not in seen:
            seen.add(id(mod))
            found.append((name, mod))

    if isinstance(target, torch.nn.Module):
        add(getattr(target, '__class__').__name__, target)
        return found
    if isinstance(target, (list, tuple)):
        for i, t in enumerate(target):
            for nm, m in _resolve_targets(t):
                add(f"{i}:{nm}", m)
        return found
    # a wrapper with ._model
    inner = getattr(target, '_model', None)
    if isinstance(inner, torch.nn.Module):
        add(getattr(target, 'identifier', 'model'), inner)
    # a BrainScoreModel: activations_model + preprocessors
    am = getattr(target, '_activations_model', None)
    if am is not None:
        for nm, m in _resolve_targets(am):
            add(f"activations:{nm}", m)
    preprocs = getattr(target, '_preprocessors', None)
    if isinstance(preprocs, dict):
        for modality, pp in preprocs.items():
            for nm, m in _resolve_targets(pp):
                add(f"{modality}:{nm}", m)
    return found


# ── PerceptWindow ───────────────────────────────────────────────────────

class PerceptWindow:
    """Capture and reconstruct the tensors a model actually ingests.

    :param target: a torch Module, a list of them, an activations wrapper, or a
        BrainScoreModel (auto-resolves the wrapped module(s) to hook).
    :param modality: force a modality for every capture ('vision'|'text'|
        'audio'); default ``None`` infers per tensor.
    :param denorm: ``(mean, std)`` for vision de-normalization, ``'auto'`` to
        introspect from the target's preprocessing, or ``None``.
    :param every: capture every Nth forward call (stride) to bound memory.
    :param max_captures: hard cap on stored captures.
    :param select_input: optional ``callable(args, kwargs) -> tensor`` to choose
        which input to capture; default is the first tensor found.
    """

    def __init__(self, target, modality: Optional[str] = None,
                 denorm: Any = None, every: int = 1, max_captures: int = 64,
                 select_input: Optional[Callable] = None, label: Optional[str] = None):
        self.targets = _resolve_targets(target)
        if not self.targets:
            raise ValueError(
                "PerceptWindow found no torch module to hook on the target. Pass "
                "a torch.nn.Module, a wrapper exposing ._model, or a "
                "BrainScoreModel with an activations_model.")
        self.modality = modality
        self._denorm_spec = denorm
        self.every = max(1, int(every))
        self.max_captures = max_captures
        self.select_input = select_input or _first_tensor
        self.label = label or 'percept'
        self.captures: List[PerceptCapture] = []
        self._handles: List[Any] = []
        self._calls = 0
        # resolve denorm now if possible (explicit tuple or auto-introspection)
        self.mean: Optional[List[float]] = None
        self.std: Optional[List[float]] = None
        if isinstance(denorm, (tuple, list)) and len(denorm) == 2:
            self.mean, self.std = list(map(float, denorm[0])), list(map(float, denorm[1]))
        elif denorm == 'auto':
            self._auto_denorm_from(target)

    def _auto_denorm_from(self, target):
        candidates = []
        preprocs = getattr(target, '_preprocessors', None)
        if isinstance(preprocs, dict):
            candidates.extend(preprocs.values())
        for attr in ('_image_processor', 'image_processor', 'preprocess', '_preprocess'):
            candidates.append(getattr(target, attr, None))
        for c in candidates:
            ms = find_normalization(c)
            if ms is not None:
                self.mean, self.std = ms
                return

    # -- context management ------------------------------------------------
    def __enter__(self):
        def make_hook(name):
            def hook(module, args, kwargs):
                self._calls += 1
                if (self._calls - 1) % self.every != 0:
                    return
                if len(self.captures) >= self.max_captures:
                    return
                t = self.select_input(args, kwargs)
                if t is None or not isinstance(t, torch.Tensor):
                    return
                t = t.detach().cpu()
                modality = self.modality or infer_modality(t)
                self.captures.append(PerceptCapture(
                    index=len(self.captures), module=name, modality=modality,
                    tensor=t, shape=tuple(t.shape), dtype=str(t.dtype)))
            return hook

        for name, module in self.targets:
            self._handles.append(
                module.register_forward_pre_hook(make_hook(name), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    # -- reconstruction ----------------------------------------------------
    def reconstruct(self, mean=None, std=None, tokenizer=None) -> List[Dict[str, Any]]:
        """Reconstruct every capture into a viewable artifact descriptor.

        Returns a list of dicts ``{index, module, modality, kind, data, ...}``.
        Vision uses ``mean``/``std`` (or those resolved at construction; else a
        flagged min-max fallback). Text needs ``tokenizer``.
        """
        mean = mean if mean is not None else self.mean
        std = std if std is not None else self.std
        out = []
        for cap in self.captures:
            rec: Dict[str, Any] = {'index': cap.index, 'module': cap.module,
                                   'modality': cap.modality, 'shape': list(cap.shape),
                                   'dtype': cap.dtype}
            if cap.modality == 'vision':
                if mean is not None and std is not None:
                    rec['kind'] = 'image'
                    rec['data'] = denormalize_image(cap.tensor, mean, std)
                    rec['denorm'] = 'mean_std'
                else:
                    rec['kind'] = 'image'
                    rec['data'] = minmax_image(cap.tensor)
                    rec['denorm'] = 'minmax-fallback'   # NOT true color
            elif cap.modality == 'text':
                if tokenizer is None:
                    rec['kind'] = 'token_ids'
                    rec['data'] = cap.tensor.numpy()
                else:
                    rec['kind'] = 'text'
                    rec['data'] = decode_tokens(cap.tensor, tokenizer)
            elif cap.modality == 'audio':
                rec['kind'] = 'waveform'
                rec['data'] = as_waveform(cap.tensor)
            else:
                rec['kind'] = 'raw'
                rec['data'] = cap.tensor.numpy()
            out.append(rec)
        return out

    def summary(self) -> Dict[str, Any]:
        by_mod: Dict[str, int] = {}
        for c in self.captures:
            by_mod[c.modality] = by_mod.get(c.modality, 0) + 1
        return {'label': self.label, 'n_captures': len(self.captures),
                'modules': sorted({c.module for c in self.captures}),
                'by_modality': by_mod,
                'denorm_resolved': self.mean is not None}

    def save(self, out_dir: str, mean=None, std=None, tokenizer=None,
             sample_rate: int = 16000) -> Dict[str, Any]:
        """Write reconstructed artifacts (PNG / txt / wav) + a manifest.json."""
        os.makedirs(out_dir, exist_ok=True)
        recs = self.reconstruct(mean=mean, std=std, tokenizer=tokenizer)
        manifest = []
        for rec in recs:
            entry = {k: rec[k] for k in ('index', 'module', 'modality', 'kind',
                                         'shape', 'dtype') if k in rec}
            if 'denorm' in rec:
                entry['denorm'] = rec['denorm']
            paths = self._write_artifact(out_dir, rec, sample_rate)
            entry['artifacts'] = paths
            manifest.append(entry)
        with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
            json.dump({'summary': self.summary(), 'captures': manifest}, f,
                      indent=2, default=str)
        return {'manifest': os.path.join(out_dir, 'manifest.json'),
                'n_captures': len(recs), 'summary': self.summary()}

    def _write_artifact(self, out_dir, rec, sample_rate) -> List[str]:
        i = rec['index']
        paths: List[str] = []
        if rec['kind'] == 'image':
            from PIL import Image
            for b, img in enumerate(rec['data']):
                arr = img[:, :, 0] if img.shape[-1] == 1 else img
                p = os.path.join(out_dir, f'percept_{i:04d}_{b}.png')
                Image.fromarray(arr).save(p)
                paths.append(p)
        elif rec['kind'] == 'text':
            p = os.path.join(out_dir, f'percept_{i:04d}.txt')
            with open(p, 'w') as f:
                f.write('\n'.join(rec['data']))
            paths.append(p)
        elif rec['kind'] == 'token_ids':
            p = os.path.join(out_dir, f'percept_{i:04d}_ids.npy')
            np.save(p, rec['data'])
            paths.append(p)
        elif rec['kind'] == 'waveform':
            for b, wav in enumerate(rec['data']):
                p = os.path.join(out_dir, f'percept_{i:04d}_{b}.wav')
                try:
                    import soundfile as sf
                    sf.write(p, wav, sample_rate)
                except Exception:
                    p = os.path.join(out_dir, f'percept_{i:04d}_{b}.npy')
                    np.save(p, wav)
                paths.append(p)
        else:
            p = os.path.join(out_dir, f'percept_{i:04d}.npy')
            np.save(p, rec['data'])
            paths.append(p)
        return paths
