"""Plumbing the extractors in this folder share.

Each function does one job that every extractor needs: find a layer by its
dotted path, copy a hooked output to numpy, hold hooks on the forward pass that
the extractor or another caller runs, run inputs in batches, key the
activation cache, and copy stimulus columns onto an assembly.
"""
import inspect
import sys
from collections import OrderedDict
from typing import Callable, List

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


class LiveRecording:
    """Hooks on ``layers`` of ``root`` for the forward passes of ``model``; one presentation per ``take``.

    The one hook lifecycle of the extractors: ``__call__`` runs the network
    inside one, and ``record(layers, stimuli)`` returns one while another part
    (``reply``, ``answer``, ``likelihood``, ``policy``) runs it. Unknown layers
    fail here, before any compute. Each layer keeps its value, through
    ``convert``, from the first pass since the previous take (the last value of
    a module that runs twice in it); later passes (generated tokens) are not kept.

    ``inputs(args, kwargs)`` runs in the forward pre-hook of that first pass;
    ``kwargs`` also holds the positional arguments, named by ``forward``. Its
    pass may give a positional argument after the first only where ``forward``
    names it with one of ``_INPUTS``, and no keyword naming another mask than
    ``attention_mask``; another pass is refused before it computes, since the
    mask of the recorded tokens would be unknown. It returns, as values, what packaging reads from the inputs (an attention mask,
    patch counts), or raises for a pass the extractor cannot package as
    ``__call__`` would, before the pass computes. With ``one_input``, a later
    pass of the presentation must show the values of the first (a repeat is
    recorded once); other input (a later chunk or window) is refused before it runs.

    ``take(row)`` returns ``package(activity, inputs, row)``: ``activity`` in the
    order the pass reached the layers, ``row`` the stimulus row presented (None
    when there is none). Without ``package`` it returns ``(activity, inputs)``.
    """

    def __init__(self, model, root, layers: List[str], convert: Callable, package: Callable = None,
                 inputs: Callable = None, one_input: bool = False):
        self.layers, self._model, self._convert = list(layers), model, convert
        self._package = package or (lambda activity, read, _row: (activity, read))
        self._inputs, self._one_input, self._reads = inputs or (lambda args, kwargs: None), one_input, bool(inputs)
        self._modules = {layer: get_layer(root, layer) for layer in self.layers}
        self._names = [p.name for p in inspect.signature(model.forward).parameters.values()
                       if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        self._handles = []
        self._clear()

    def _clear(self):
        self._activity, self._read, self._shown, self._passes = OrderedDict(), None, None, 0

    def __enter__(self):
        try:
            self._handles.append(self._model.register_forward_pre_hook(self._before, with_kwargs=True))
            self._handles.append(self._model.register_forward_hook(self._after))
            for layer, module in self._modules.items():
                self._handles.append(_hook(module, layer, lambda value, layer=layer: self._keep(layer, value)))
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exception):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _before(self, _module, args, kwargs):
        if self._passes == 0:
            named = {**dict(zip(self._names, args)), **kwargs}
            unknown = [f'positional argument {i}' for i, arg in enumerate(args[1:], 1) if arg is not None
                       and (i >= len(self._names) or self._names[i] not in _INPUTS)]
            unknown += [key for key, arg in named.items() if 'mask' in key and key != 'attention_mask'
                        and arg is not None]
            if self._reads and unknown:
                raise ValueError(f'A recording reads the inputs {_INPUTS} of a pass, and this pass also '
                                 f'gives {unknown}, which may hold a mask of the recorded tokens; give '
                                 f'the model its inputs by these names.')
            self._read = self._inputs(args, named)
            self._shown = _copy((args, kwargs)) if self._one_input else None
        elif self._one_input and not _same(self._shown, (args, kwargs)):
            raise ValueError(
                'A recording keeps one forward pass per presentation, and a later pass showed '
                'other input than the first (for example a second chunk or window of the '
                'stimulus). Present each stimulus in one pass, or record it in a session '
                'without a task.')

    def _after(self, *_):
        self._passes += 1

    def _keep(self, layer, value):
        if self._passes == 0:
            self._activity[layer] = _copy(self._convert(value))     # the network may reuse the buffer

    def take(self, row=None):
        """The first forward pass since the last take, packaged by the extractor."""
        missing = [layer for layer in self.layers if layer not in self._activity]
        if missing:
            raise ValueError(f'No forward pass reached {missing} since the last presentation.')
        activity, read = self._activity, self._read
        self._clear()
        return self._package(activity, read, row)


def _copy(value):
    """``value`` with every tensor and array copied: a snapshot that later in-place changes miss."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (tuple, list)):
        return type(value)(_copy(item) for item in value)
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    return value


def _same(a, b) -> bool:
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.shape == b.shape and a.device == b.device and bool(torch.equal(a, b)))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
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


def value(tensor):
    """A tensor as a numpy copy (None stays None): what a pre-hook keeps of an input."""
    return None if tensor is None else np.array(tensor.detach().cpu())


# The inputs an extractor reads from a pass (``LiveRecording(inputs=...)``): the
# token or sample input of text and sound, and the one mask read.
_INPUTS = ('input_ids', 'inputs_embeds', 'input_values', 'input_features', 'attention_mask')


def attention_mask(args, kwargs):
    """A ``LiveRecording`` input reader: the attention mask a pass was given, as values."""
    return value(kwargs.get('attention_mask'))


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
            'dtype, so they need separate calls (a session makes them).')


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
