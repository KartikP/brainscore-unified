"""Plumbing the extractors in this folder share.

Each function does one job that every extractor needs: find a layer by its
dotted path, copy a hooked output to numpy, hold hooks on the forward pass that
the extractor or another caller runs, run inputs in batches, key the
activation cache, and copy stimulus columns onto an assembly.

The activation cache key of a wrapper is explicit (``cache_config``): it names
the ``_VERSION`` in the body of the wrapper's class, not source text. A change
that moves the values a wrapper extracts raises that ``_VERSION`` by hand. This holds for
code outside the wrapper classes too: this module, and what the wrappers call
(``video_wrapper._window_features``, ``_shared_clock``, ``_edges_from_centers``,
``brainscore.temporal.window_plan``, ``brainscore_core.assembly_builder``). A
change here raises the ``_VERSION`` of every wrapper class that uses the
changed code. ``tests/test_model_helpers/test_wrapper_cache_guard.py`` fails when values move
under an unchanged ``_VERSION``.
"""
import inspect
import warnings
from collections import OrderedDict
from typing import Callable, List

import numpy as np
from result_caching import is_enabled
from tqdm.auto import tqdm

from brainscore_core.assembly_builder import make_assembly, make_layer_neuroid_coords, make_neuroid_ids
from brainscore_core.extraction_cache import extraction_fingerprint, file_inputs, model_config
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
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


def hook(module, layer: str, keep: Callable):
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

    The activity is what the network computed from the input of that pass. A
    task recording and the passive call reduce and package it with the same
    functions, so they are equal when the network receives the same input; when
    the answering part prepares the stimulus in its own way (a chat processor),
    the recording holds what the network received from that part.
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
                self._handles.append(hook(module, layer, lambda value, layer=layer: self._keep(layer, value)))
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
        return value.copy(order='K')     # keep the memory layout: a later reduction rounds by it
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


# The one value set of ``layer_aggregation``. Each wrapper takes the values that
# fit its outputs: ``mean`` over the positions (tokens, time steps, patches),
# the ``first`` or ``last`` position, or the whole ``sequence`` of positions.
LAYER_AGGREGATIONS = ('mean', 'first', 'last', 'sequence')


def checked_aggregation(value: str, allowed) -> str:
    """``value`` when it is one of ``allowed`` (a subset of ``LAYER_AGGREGATIONS``); raises otherwise."""
    if value not in allowed:
        raise ValueError(f'layer_aggregation must be one of {tuple(allowed)}; got {value!r}.')
    return value


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


def kind(wrapper) -> str:
    """The wrapper's class as ``<module>.<Class>``: the store path of its cache entries."""
    cls = type(wrapper)
    return f'{cls.__module__}.{cls.__qualname__}'


def cache_config(wrapper, fields):
    """The explicit cache key of a wrapper, without its inputs.

    Parts: the kind (``<module>.<Class>``); the ``_VERSION`` in the body of the
    wrapper's class; the settings ``fields`` (``CACHE_FIELDS``, batch size included; a
    callable counts by its code); the model (weights hash, structure, dtypes,
    library versions, precision, hooks); the input device. ``cache_signature``
    adds the inputs. No source text is part of it, so an edit that keeps values
    (a comment, a docstring, a move) keeps the key.

    ``_VERSION`` is read from the body of the wrapper's own class only. A
    subclass does not inherit it, since its own code can change values; without
    its own ``_VERSION``, ``cache_signature`` does not cache it.

    A torch autocast context around the call changes the values, so when
    autocast is enabled for the input device type, its dtype is part of the
    key (``autocast``). Without autocast the key has no such part.
    """
    config = {'kind': kind(wrapper), 'version': _version(type(wrapper)),
              'settings': {field: getattr(wrapper, field) for field in fields},
              'input_device': str(getattr(wrapper, '_device', None)),
              'model': model_config(wrapper._model)}
    autocast = _autocast(getattr(wrapper, '_device', None))
    if autocast is not None:
        config['autocast'] = autocast
    return config


def _autocast(device):
    """The autocast dtype in force for ``device``'s type at this moment, or None when autocast is off."""
    import torch
    device_type = torch.device(device).type if device is not None else 'cpu'
    try:
        if not torch.is_autocast_enabled(device_type):
            return None
        return {'device_type': device_type, 'dtype': str(torch.get_autocast_dtype(device_type))}
    except RuntimeError:                  # a device type autocast does not know
        return None


def _version(cls):
    """The ``_VERSION`` in the body of ``cls`` itself, or None; never inherited."""
    return vars(cls).get('_VERSION')


_UNVERSIONED = set()        # classes already warned about


def cache_signature(wrapper, stored: str, inputs, stimuli_identifier, *, settings: bool = False):
    """The cache fingerprint for one extraction, or None when the cache does not apply.

    ``stored`` names the wrapper's ``@store_xarray`` method; ``inputs`` is what
    the fingerprint reads for the stimuli (the texts, or the files behind paths).
    With ``settings``, a pair: the fingerprint and the fingerprint of the
    configuration alone (the key without the inputs), from the same traversal.
    """
    if not stimuli_identifier:
        return (None, None) if settings else None
    if _version(type(wrapper)) is None:
        cls = type(wrapper)
        if cls not in _UNVERSIONED:
            _UNVERSIONED.add(cls)
            warnings.warn(
                f'{kind(wrapper)} has no _VERSION of its own, so it extracts every time and does not use '
                f'the activation cache. To cache it, add _VERSION = 1 to its class body and a guard test '
                f'that pins its values (see tests/test_model_helpers/test_wrapper_cache_guard.py).', stacklevel=3)
        return (None, None) if settings else None
    return extraction_fingerprint({'configuration': wrapper.cache_config(), 'inputs': inputs},
                                  cache_identifier=f'{kind(wrapper)}.{stored}',
                                  part='configuration' if settings else None)


class CachingExtractor:
    """The one sequence of the caching wrappers: a stimulus table, its inputs, the cache, its columns.

    ``extractor(stimuli, layers)`` checks ``layers`` first (a non-empty list of
    layer names), before the store and before any compute. It takes a ``StimulusSet`` (its inputs come from
    ``_table_inputs`` and its columns go onto the result with ``_columns``) or a
    list of inputs. Both pass through ``_cached``: with a stimuli identifier and
    the cache on, the wrapper's ``@store_xarray`` method (``_STORED``) gives the
    layers, otherwise ``_from_<inputs>`` computes them; the result names the
    reading wrapper (``labelled``) and has its units in the requested layer
    order (``in_layer_order``). ``_finish`` sets what every result carries.

    A wrapper sets ``_INPUTS`` (``'texts'`` or ``'paths'``), ``CACHE_FIELDS`` and
    its own ``_VERSION``, and defines ``_from_<inputs>(inputs, layers,
    stimuli_identifier)``, its stored method and ``_table_inputs``; it keeps
    its forward pass and its reduce and package functions.
    """

    _INPUTS = 'paths'

    @property
    def identifier(self) -> str:
        return self._identifier

    def __call__(self, stimuli, layers, stimuli_identifier=None, **kwargs):
        if (not isinstance(layers, (list, tuple)) or not layers
                or not all(isinstance(layer, str) for layer in layers)):
            raise ValueError(f'layers is a non-empty list of layer names, for example '
                             f"['encoder.layers.6']; got {layers!r}.")
        if isinstance(stimuli, StimulusSet):
            if stimuli_identifier is None and hasattr(stimuli, 'identifier'):
                stimuli_identifier = stimuli.identifier
            assembly = self._cached(self._table_inputs(stimuli), layers, stimuli_identifier)
            assembly = self._columns(assembly, stimuli)
        else:
            assembly = self._cached(list(stimuli), layers, stimuli_identifier)
        return self._finish(assembly)

    def cache_config(self):
        return cache_config(self, self.CACHE_FIELDS)

    def _cached(self, inputs, layers, stimuli_identifier=None):
        from brainscore._model import parts
        stored = f'_from_{self._INPUTS}_stored'
        log = parts.active()
        if log is None:
            signature = cache_signature(self, stored, self._key_inputs(inputs), stimuli_identifier)
        else:
            signature = self._signature_and_facts(log, stored, inputs, stimuli_identifier)
        if signature is None:
            return in_layer_order(getattr(self, f'_from_{self._INPUTS}')(inputs, layers, stimuli_identifier), layers)
        result = getattr(self, stored)(stimuli_identifier=stimuli_identifier, layers=layers,
                                       extraction_fingerprint=signature, **{self._INPUTS: inputs})
        return in_layer_order(labelled(c_order(result), self._identifier), layers)

    def _signature_and_facts(self, log, stored, inputs, stimuli_identifier):
        """The cache signature, and this call's settings fingerprint in the session's part log.

        The settings fingerprint is the cache key without the inputs, taken from the
        traversal that makes the signature: no tensor is read a second time. With the
        store off nothing is hashed and the fact says so.
        """
        signature, settings = cache_signature(self, stored, self._key_inputs(inputs), stimuli_identifier,
                                              settings=True)
        if signature is None:
            settings = ('not hashed (store off)' if not is_enabled(f'{kind(self)}.{stored}')
                        else 'not hashed (store not used)')
        log.settings.setdefault(id(self), set()).add(settings)
        return signature

    def _key_inputs(self, inputs):
        """What the cache key reads of the inputs: the bytes of the files behind paths."""
        return file_inputs(inputs)

    def _columns(self, assembly, stimuli):
        return attach_stimulus_set(assembly, stimuli)

    def _finish(self, assembly):
        return assembly


def c_order(assembly):
    """``assembly`` with its values in C order, the layout the wrappers compute.

    The store selects the requested layers with an integer index along
    ``neuroid`` (``result_caching``'s ``filter_data``), which returns the same
    values in another memory layout. A reduction rounds by the layout (numpy
    sums a strided axis in another order than a contiguous one), so a score with
    the store on would differ from the score with the store off in the last
    digits. Values, coordinates and attributes are kept; anything that is not
    an assembly with numpy values, and values already in C order, are returned
    as they are (the store-off path does not change).

    Two call sites use it. The caching wrappers apply it at their cache boundary
    (``CachingExtractor._cached``), so a wrapper called directly, by a tool or a
    notebook, returns C order. The model class applies it to every extractor
    result where it receives one (``brainscore._model.dispatch.call_extractor``),
    so any other extractor with a store of its own, such as the legacy vision
    ``PytorchWrapper``, gives the model the same layout with the store on and off.
    """
    values = getattr(assembly, 'values', None)
    if not hasattr(assembly, 'dims') or not isinstance(values, np.ndarray) or values.flags['C_CONTIGUOUS']:
        return assembly
    return assembly.copy(data=np.ascontiguousarray(values))


def labelled(assembly, identifier: str):
    """``assembly`` with its ``model`` and ``neuroid_id`` coordinates naming ``identifier``.

    The cache key leaves the identifier out, so wrappers with the same weights
    and settings share one stored entry, and that entry carries the labels of
    the wrapper that wrote it. Every cache return passes through here. Only the
    two unit labels are replaced; every other coordinate, index, name and
    attribute stays as stored. An assembly that already names ``identifier``, or
    has no ``model`` coordinate, is returned as it is.
    """
    try:
        models = assembly['model'].values
    except KeyError:
        return assembly
    if (models == identifier).all():
        return assembly
    layers, units = assembly['layer'].values, assembly['neuroid_num'].values
    ids = {layer: make_neuroid_ids(identifier, layer, int(units[layers == layer].max()) + 1)
           for layer in set(layers)}
    labels = {'model': [identifier] * len(models),
              'neuroid_id': [ids[layer][unit] for layer, unit in zip(layers, units)]}
    index = assembly.indexes.get('neuroid')
    if index is not None and getattr(index, 'nlevels', 1) > 1:
        import pandas as pd
        arrays = [pd.Index(labels[name], dtype=index.levels[i].dtype) if name in labels
                  else index.get_level_values(i) for i, name in enumerate(index.names)]
        assembly = assembly.assign_coords(neuroid=pd.MultiIndex.from_arrays(arrays, names=index.names))
        labels = {name: values for name, values in labels.items() if name not in index.names}
    return assembly.assign_coords({name: ('neuroid', values) for name, values in labels.items()})


def in_layer_order(assembly, layers):
    """``assembly`` with its units in the order of the requested ``layers``.

    The one rule of the caching wrappers: a result has its units in the order
    the caller requested, whether it was computed now or returned by the store
    (which merges a cached layer and a new one in its own order). Stable within
    a layer; an assembly already in that order is returned as it is.
    """
    position = {layer: i for i, layer in enumerate(layers)}
    order = [position.get(layer, len(position)) for layer in assembly['layer'].values]
    if order == sorted(order):
        return assembly
    return assembly.isel(neuroid=np.argsort(order, kind='stable'))


def catalog_paths(stimuli, channel: str):
    """The values of the first column of ``stimuli`` that the core catalog binds to ``channel``, or None.

    The catalog (``brainscore_core.io_catalog.stimulus_columns``) is the one
    answer to which column feeds which channel, for pre-flight, dispatch and
    the wrappers; columns are tried in its order.
    """
    from brainscore_core.io_catalog import stimulus_columns
    for column, fed in stimulus_columns().items():
        if fed == channel and column in stimuli.columns:
            return list(stimuli[column].values)
    return None


def catalog_columns(channel: str) -> list:
    """The stimulus columns the core catalog binds to ``channel``, in its order (for messages)."""
    from brainscore_core.io_catalog import stimulus_columns
    return [column for column, fed in stimulus_columns().items() if fed == channel]


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
