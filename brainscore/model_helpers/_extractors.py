"""Plumbing the extractors in this folder share.

Each function does one job that every extractor needs: find a layer by its
dotted path, copy a hooked output to numpy, hold hooks for one forward pass,
record while another caller runs the model, run inputs in batches, key the
activation cache, and copy stimulus columns onto an assembly.
"""
import inspect
import sys
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Dict, List

import numpy as np
from tqdm.auto import tqdm

from brainscore_core.assembly_builder import make_assembly, make_layer_neuroid_coords
from brainscore_core.extraction_cache import extraction_fingerprint, wrapper_config
from brainscore.model_helpers.selection import split_site


def get_layer(root, layer: str):
    """The module that ``layer`` names under ``root``.

    ``layer`` is a dotted path; a numeric part also indexes a container. A
    trailing ``':input'`` names the module's first input (see ``split_site``);
    the module is the same.
    """
    module = root
    for part in split_site(layer)[0].split('.'):
        found = getattr(module, part, None)
        if found is None and part.isdigit() and hasattr(module, '__getitem__'):
            try:
                found = module[int(part)]
            except (IndexError, KeyError, TypeError):
                found = None
        if found is None:
            raise ValueError(f"Layer '{layer}' not found at part '{part}'")
        module = found
    return module


def as_array(output) -> np.ndarray:
    """A hooked output as numpy: the first element of a tuple, floats as float32.

    numpy has no bfloat16, and large models load in it, so every float width is
    promoted on the way out.
    """
    activity = _tensor(output).cpu()
    if activity.dtype.is_floating_point:
        activity = activity.float()
    return activity.numpy()


def as_numpy(output) -> np.ndarray:
    """A hooked output as numpy in its own dtype: the first element of a tuple.

    Only a float type that numpy cannot hold (bfloat16) becomes float32. The
    audio extractor keeps the precision of its network up to its temporal
    reduction.
    """
    activity = _tensor(output).cpu()
    try:
        return activity.numpy()
    except TypeError:
        if not activity.dtype.is_floating_point:
            raise
        return activity.float().numpy()


def as_float32(output) -> np.ndarray:
    """A hooked output as float32 numpy, whatever its type: the first element of a tuple."""
    return _tensor(output).cpu().float().numpy()


def _tensor(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.detach()


def _hook(module, layer: str, keep: Callable):
    """Hook ``module``; ``keep(value)`` gets its output, or its first input when ``layer``
    ends in ``':input'`` (the tensor an ``':input'`` intervention changes)."""
    if split_site(layer)[1] == 'input':
        def on_input(_module, args):
            keep(args[0])
        return module.register_forward_pre_hook(on_input)

    def on_output(_module, _inputs, output):
        keep(output)
    return module.register_forward_hook(on_output)


@contextmanager
def hooked(root, layers: List[str], convert: Callable):
    """Hook ``layers`` until the block exits; yields ``{layer: convert(value)}``.

    Entries appear in the order the forward pass reaches the layers; a module
    that runs twice keeps its last value. Every hook is removed on exit, also
    when a later layer is not found.
    """
    captured: Dict[str, np.ndarray] = OrderedDict()
    handles = []
    try:
        for layer in layers:
            handles.append(_hook(get_layer(root, layer), layer,
                                 lambda value, layer=layer: captured.__setitem__(layer, convert(value))))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


class LiveRecording:
    """Record ``layers`` while another caller runs ``model``, one presentation at a time.

    An extractor's ``record(layers)`` returns one; ``BrainScoreModel`` uses it to
    record regions during the forward pass that produces an answer, so a
    stimulus is shown once for both measurements. Unknown layers fail here,
    before any compute; the hooks are on while the ``with`` block runs.
    ``take(row)`` gives ``take(first, call, row)``: each layer's value from the
    first forward pass since the previous take (passed through ``convert``),
    the ``(args, kwargs)`` of that pass's call to ``model``, and the stimulus
    row presented (None when the caller has no row). For a generating model
    the first pass is the prompt pass.

    ``check(first_call, call)``, when given, runs before every pass reaches
    ``model`` (``first_call`` is None for the first pass of a presentation). It
    raises for a pass the extractor cannot record completely, before that pass
    computes, so a recording is never a part of what the passive path records.
    """

    def __init__(self, model, root, layers: List[str], take: Callable, convert: Callable,
                 check: Callable = None):
        self.layers, self._model, self._take, self._convert = list(layers), model, take, convert
        self._check = check
        self._modules = {layer: get_layer(root, layer) for layer in self.layers}
        self._first: Dict[str, np.ndarray] = {}
        self._call = None
        self._handles = []

    def __enter__(self):
        try:
            self._handles.append(self._model.register_forward_pre_hook(self._watch_call,
                                                                       with_kwargs=True))
            for layer, module in self._modules.items():
                self._handles.append(_hook(module, layer, lambda value, layer=layer:
                                           self._keep_first(layer, value)))
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exception):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _keep_first(self, layer, value):
        if layer not in self._first:            # a later pass (a generated token) is not kept
            self._first[layer] = self._convert(value)

    def _watch_call(self, _module, args, kwargs):
        if self._check is not None:
            self._check(self._call, (args, kwargs))
        if self._call is None:
            self._call = (args, kwargs)

    def take(self, row=None):
        """The first forward pass since the last take, packaged by the extractor."""
        missing = [layer for layer in self.layers if layer not in self._first]
        if missing:
            raise ValueError(f'No forward pass reached {missing} since the last presentation.')
        result = self._take({layer: self._first[layer] for layer in self.layers}, self._call, row)
        self._first.clear()
        self._call = None
        return result


def refuse_other_input(first_call, call) -> None:
    """A ``LiveRecording`` check: every pass of a presentation shows the input of the first.

    A later pass with other input (a later chunk or window of the stimulus)
    would be missing from a recording that keeps the first pass, so it is
    refused before it runs. A repeat of the same input (a second scoring of
    the same prompt) is recorded once.
    """
    if first_call is not None and not _same(first_call, call):
        raise ValueError(
            'A recording during a task keeps one forward pass per presentation, and a later '
            'pass showed other input than the first (for example a second chunk or window of '
            'the stimulus). Present each stimulus in one pass, or record it in a session '
            'without a task.')


def _same(a, b) -> bool:
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.shape == b.shape and a.device == b.device and bool(torch.equal(a, b)))
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b)))
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                and all(_same(a[k], b[k]) for k in a))
    try:
        return bool(a == b)
    except Exception:                                       # noqa: BLE001
        return a is b


class LayerDtypesDiffer(ValueError):
    """Layers of one call have different dtypes, so one assembly cannot hold them."""


def one_dtype(layer_activations) -> None:
    """Refuse to join layers of different dtypes: numpy would promote them, and a
    layer's values would then differ from a call for that layer alone."""
    dtypes = {layer: np.asarray(values).dtype for layer, values in layer_activations.items()
              if not layer.startswith('_')}
    if len(set(dtypes.values())) > 1:
        raise LayerDtypesDiffer(
            f'The layers {sorted(dtypes)} have different dtypes '
            f'({", ".join(f"{l}: {d}" for l, d in dtypes.items())}); one assembly holds one '
            'dtype, so they need separate calls. A session records them apart under '
            'BRAINSCORE_REGIONS_PER_PASS=one; a recording during a task cannot.')


def grid_thw(call):
    """The ``grid_thw`` a VLM vision network was called with (keyword or second argument), or None."""
    args, kwargs = call if call is not None else ((), {})
    return kwargs.get('grid_thw', args[1] if len(args) > 1 else None)


def run_batches(items, batch_size: int, run: Callable, desc: str):
    """``run`` each batch of ``items``; returns ``{layer: array}`` with all items on the first axis."""
    total = len(items)
    stacked = None
    for start in tqdm(range(0, total, batch_size), unit_scale=batch_size, desc=desc):
        end = min(start + batch_size, total)
        batch = run(items[start:end])
        if stacked is None:
            stacked = OrderedDict((layer, np.empty((total,) + value.shape[1:], dtype=value.dtype))
                                  for layer, value in batch.items())
        for layer, value in batch.items():
            stacked[layer][start:end] = value
    return stacked


def layer_assembly(identifier: str, layer: str, values: np.ndarray):
    """One layer's activations as an assembly: ``(presentation, neuroid)``, or
    ``(presentation, time_bin, neuroid)`` for three axes. Stimulus ids are positions."""
    dims = ['presentation', 'neuroid'] if values.ndim == 2 else ['presentation', 'time_bin', 'neuroid']
    coords = {
        'stimulus_id': ('presentation', list(range(values.shape[0]))),
        **make_layer_neuroid_coords(identifier, layer, values.shape[-1],
                                    include=('neuroid_id', 'neuroid_num', 'model', 'layer')),
    }
    return make_assembly(values, coords=coords, dims=dims)


def cache_config(wrapper, fields):
    """The wrapper's cache configuration, with this module's source in its implementation.

    The source of the wrapper class is part of the cache key, so an edit to the
    extraction code recomputes cached activations. The code here is extraction
    code too.
    """
    config = wrapper_config(wrapper, fields)
    config['implementation'].append((__name__, inspect.getsource(sys.modules[__name__])))
    return config


def cache_signature(wrapper, stored: str, inputs, stimuli_identifier):
    """The cache fingerprint for one extraction, or None when the cache does not apply.

    ``stored`` names the wrapper's ``@store_xarray`` method; ``inputs`` is what
    the fingerprint reads for the stimuli (the texts, or the files behind paths).
    """
    if not (wrapper.backbone_id and stimuli_identifier):
        return None
    return extraction_fingerprint(
        {'configuration': wrapper.cache_config(), 'inputs': inputs},
        cache_identifier=(f'{type(wrapper).__module__}.{type(wrapper).__name__}'
                          f'.{stored}/identifier={wrapper.backbone_id}'),
    )


def attach_stimulus_set(assembly, stimuli):
    """Copy every stimulus column onto the presentation axis.

    The presentation index is reset first, so that ``stimulus_id`` does not
    collide with a level of a MultiIndex built at construction. A MultiIndex
    becomes one coordinate per level. A plain index holds only positions, which
    ``stimulus_id`` replaces, so it is dropped.
    """
    stimulus_ids = list(stimuli['stimulus_id'].values)
    if assembly.sizes['presentation'] != len(stimulus_ids):
        raise ValueError(
            f"Mismatch between activations ({assembly.sizes['presentation']}) "
            f"and stimulus set ({len(stimulus_ids)}) on the presentation axis.")
    if 'presentation' in assembly.indexes:
        multi = getattr(assembly.indexes['presentation'], 'nlevels', 1) > 1
        assembly = assembly.reset_index('presentation', drop=not multi)
    assembly = assembly.assign_coords(stimulus_id=('presentation', stimulus_ids))
    for column in stimuli.columns:
        if column != 'stimulus_id':
            assembly = assembly.assign_coords({column: ('presentation', list(stimuli[column].values))})
    return assembly
