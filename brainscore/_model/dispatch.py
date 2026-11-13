"""The steps the session parts of BrainScoreModel call: ``record`` and ``act``.

``record`` records the regions that ``start_recording`` set from a stimulus
table; ``act`` asks the policy for one closed-loop step. A session part calls
them directly, and so does ``BrainScoreModel.process``; nothing here selects a
part. The extractor for an input is chosen by ``extractor_for_form``.
"""

import os
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from typing import List, Set

from brainscore_core.io_catalog import canonical_modality


_cache_disable_depth = 0
_cache_disable_prev = None
_cache_disable_lock = threading.Lock()


@contextmanager
def _activation_cache_disabled():
    """Bypass result_caching for the duration. Used when an intervention makes the
    cached (unchanged) activations invalid but the cache key cannot see it.

    Reentrancy- and thread-safe via a lock-guarded depth counter: nested,
    interleaved, or concurrent contexts only restore the original
    ``RESULTCACHING_DISABLE`` when the outermost/last one exits (a plain
    save/restore would clear the flag early; an unlocked counter would race)."""
    global _cache_disable_depth, _cache_disable_prev
    with _cache_disable_lock:
        if _cache_disable_depth == 0:
            _cache_disable_prev = os.environ.get('RESULTCACHING_DISABLE')
            os.environ['RESULTCACHING_DISABLE'] = '1'
        _cache_disable_depth += 1
    try:
        yield
    finally:
        with _cache_disable_lock:
            _cache_disable_depth -= 1
            if _cache_disable_depth == 0:
                if _cache_disable_prev is None:
                    os.environ.pop('RESULTCACHING_DISABLE', None)
                else:
                    os.environ['RESULTCACHING_DISABLE'] = _cache_disable_prev


def _is_clip(stimuli) -> bool:
    """Whether a stimulus table has a column that feeds ``vision:video``."""
    from brainscore_core.io_catalog import column_channel
    return any(column_channel(column) == 'vision:video'
               for column in getattr(stimuli, 'columns', ()))


def fused_for(owner, detected):
    """The fused extractor taking every detected input type, if one is registered."""
    if len(detected) < 2:
        return None
    for types, extractor in owner._fused_extractors:
        if set(detected) <= types:
            return extractor
    return None


def extractor_for(owner, modality: str, stimuli):
    """The extractor for ``modality`` that reads ``stimuli`` (see ``extractor_for_form``)."""
    return extractor_for_form(owner, modality, video=_is_clip(stimuli))


def extractor_for_form(owner, modality: str, video: bool):
    """The extractor for ``modality``; with an image and a video extractor, the one
    for the form: ``vision:video`` when ``video``, else ``vision:image``. None when
    the model has no such extractor."""
    modality = canonical_modality(modality)
    matching = extractor_keys(owner, modality)
    if len(matching) > 1 and modality == 'vision':
        wanted = 'video' if video else 'vision'
        matching = [key for key in matching if key == wanted] or matching
    return owner._extractors[matching[0]] if matching else None


def extractor_keys(owner, modality: str) -> List[str]:
    """The extractor keys that serve ``modality``: every key of its input family
    (``vision`` and ``video`` both serve ``vision``). ``extractor_for_form`` picks
    one of them by the form of the stimuli, so each of their forms is accepted."""
    modality = canonical_modality(modality)
    return [key for key in owner._extractors if canonical_modality(key) == modality]


def normalize_extractors(owner, extractor):
    """The ``extractor`` argument as a modality -> extractor mapping.

    A fused extractor is registered under every input type it takes at once,
    ``{('vision', 'text'): fused}``. It also serves each type alone unless a
    single-type extractor is registered for it.
    """
    from collections.abc import Mapping

    if extractor is None:
        return {}
    if not isinstance(extractor, Mapping):
        raise TypeError(
            'extractor must be a mapping {modality: extractor}; the keys declare '
            'the inputs, for example extractor={"vision": wrapper}.')
    owner._fused_extractors = [
        (frozenset(canonical_modality(m) for m in key), value)
        for key, value in extractor.items() if isinstance(key, tuple)]
    extractors = {}
    # The raw key names stay: ``video`` declares ``vision:video``. Dispatch
    # canonicalizes when it looks an extractor up.
    for key, value in extractor.items():
        if isinstance(key, tuple):
            for modality in key:
                extractors.setdefault(modality, value)
    extractors.update({key: value for key, value in extractor.items()
                       if not isinstance(key, tuple)})
    if any(not callable(value) for value in extractors.values()):
        raise TypeError(
            'Every extractor must be callable as extractor(stimuli, layers=...). '
            'Check the values passed in extractor={modality: extractor}.')
    return extractors


@contextmanager
def one_call_per_extractor(model, stimuli, plan):
    """Inside, the first request to an extractor in ``plan``
    (``[(extractor, layers), ...]``) on ``stimuli`` runs all its listed layers
    once, and each request of those layers gets its part of that output
    (``_SharedCall``). The caller lists only extractors that declare
    ``independent_layers``. Every other request runs as before."""
    previous = model._shared_calls
    model._shared_calls = (stimuli, {id(extractor): _SharedCall(extractor, layers)
                                     for extractor, layers in plan})
    try:
        yield
    finally:
        model._shared_calls = previous


def record(model, stimuli, multi_modality: bool = False):
    """The activity of the regions ``start_recording`` set, for a stimulus table.

    The one step that records from the extractors: the neural session part,
    the readout and ``process`` call it. Without ``multi_modality`` one input
    modality is read (the one the recorded regions name, else the first in
    ``MODALITY_PRIORITY``); with it, every detected one, concatenated along
    ``neuroid`` with a ``modality`` coordinate. Several recording time bins
    repeat a static response across the bins, as legacy vision models do.
    """
    if not model._input_keys:
        raise TypeError(f"Subject '{model.identifier}' declares no input, so it cannot take "
                        f"{type(stimuli).__name__}.")
    output = _record(model, stimuli, multi_modality)
    time_bins = model._recorder.time_bins
    if time_bins is not None and len(time_bins) > 1:
        from brainscore_core.legacy_facade import repeat_over_time_bins
        output = repeat_over_time_bins(output, time_bins)
    return output


def _record(model, stimuli, multi_modality):
    recorder = model._recorder
    detected = detect_modalities(model, stimuli)
    if not detected:
        from brainscore_core.io_catalog import stimulus_columns
        from brainscore_core.streaming import parse_channel
        families = {canonical_modality(key) for key in model._input_keys}
        accepted = {m: sorted(c for c, channel in stimulus_columns().items()
                              if parse_channel(channel)[0] == m)
                    for m in families}
        raise ValueError(
            f"No column in this stimulus set says what kind of input it holds "
            f"(columns present: {list(stimuli.columns)}). Add one column per input "
            f"this subject takes; each row holds the input itself or its file path, "
            f"for example stimuli['image_path'] = [...] for images. Columns this "
            f"subject reads: {accepted}. A stimulus_paths dict alone is not enough: "
            f"the column names the input type."
        )

    layers = recorder.current_layers()
    modality = pick_modality(model, detected)
    requested_modalities = {canonical_modality(model._region_modality_map[r])
                            for r in recorder.recording_regions if r in model._region_modality_map}
    missing = requested_modalities - detected
    if missing:
        raise ValueError(f'Recording targets require missing input modalities: {sorted(missing)}.')
    if len(requested_modalities) == 1:
        modality = next(iter(requested_modalities))
    elif len(requested_modalities) > 1 and not multi_modality:
        raise ValueError('Recording targets span modalities; pass multi_modality=True.')

    # With regions mapped but none recorded, the caller forgot start_recording.
    # Only the modalities that run are checked, so a plain callable on the
    # selected input is not refused because another tower is layer-aware.
    will_run = detected if (multi_modality and len(detected) > 1) else {modality}
    if (not layers and model._region_layer_map_dict
            and not recorder.composite_recording
            and any(supports_layer_extraction(model, m) for m in will_run)):
        raise ValueError(
            f"Subject '{model.identifier}' records no region: request a "
            f"neural channel in the session, for example "
            f"stimulus_session(stimuli, ['neural:IT']). Without one the "
            f"recording has 0 neuroids. Known regions: "
            f"{sorted(model._region_layer_map_dict.keys())}."
        )

    if (multi_modality or fused_for(model, detected) is not None) and len(detected) > 1:
        return extract_modalities(model, stimuli, detected, layers)
    if not multi_modality and len(detected) > 1:
        import warnings
        warnings.warn(
            f"Stimulus set contains multiple supported modalities "
            f"{sorted(detected)} for subject '{model.identifier}'. "
            f"Default process(..., multi_modality=False) will use "
            f"'{modality}' via recording-target routing or MODALITY_PRIORITY. Pass "
            f"multi_modality=True to extract all supported modalities, "
            f"or provide stimulus columns for only the modality you want.",
            UserWarning,
            stacklevel=3,
        )
    if recorder.composite_recording and supports_layer_extraction(model, modality):
        return recorder.process_composite_regions(stimuli, modality)
    layers = recorder.filter_layers_for_modality(layers, modality)
    assembly = extract_for_modality(model, stimuli, modality, layers)
    if recorder.is_multi_region and supports_layer_extraction(model, modality):
        assembly = recorder.tag_neuroids_with_regions(assembly, modality)
    return assembly


def act(model, row):
    """The policy's outputs for one closed-loop step.

    A step is a row: a mapping of channel -> payload (a message from another
    agent is a ``text`` entry, with the sender in the event meta). The policy
    returns a mapping of output channel -> payload, for example
    ``{'motor': action}``; it needs no Brain-Score import. Device payloads are
    checked by their harness (``brainscore.harnesses.robotics``), not here.
    """
    if model._policy is None:
        raise NotImplementedError(
            f"Subject '{model.identifier}' has no policy. Closed-loop evaluation "
            f"requires BrainScoreModel(policy=...), a callable "
            f"policy(row) -> {{output channel: payload}}."
        )
    outputs = model._policy(row)
    if not isinstance(outputs, Mapping):
        raise TypeError(
            f"Subject '{model.identifier}' policy returned {type(outputs).__name__}; "
            f"expected a mapping of output channel -> payload, for example "
            f"{{'motor': action}}."
        )
    return outputs


def _declares(model, modality: str) -> bool:
    """Whether a declared input key is ``modality`` or canonicalizes to it
    (a ``video`` key is temporal vision)."""
    keys = model._input_keys
    return modality in keys or any(canonical_modality(key) == modality for key in keys)


def detect_modalities(model, stimuli) -> Set[str]:
    """Declared input modalities whose columns appear in ``stimuli``."""
    from brainscore_core.io_catalog import column_channel
    from brainscore_core.streaming import parse_channel
    detected: Set[str] = set()
    for col in stimuli.columns:
        channel = column_channel(col)
        if channel is None:
            continue
        # The channel family is the modality: vision:image and vision:video are vision.
        modality = canonical_modality(parse_channel(channel)[0])
        if _declares(model, modality):
            detected.add(modality)
    return detected


def pick_modality(model, detected: Set[str]) -> str:
    for m in model.MODALITY_PRIORITY:
        if m in detected:
            return m
    return sorted(detected)[0]


def supports_layer_extraction(model, modality: str) -> bool:
    return any(canonical_modality(key) == canonical_modality(modality)
               for key in model._extractors)


def extract_for_modality(model, stimuli, modality: str, layers: List[str]):
    """Call the extractor for ``modality`` and the form of ``stimuli``.

    An intervention is a forward hook the activation cache key does not see,
    so the cache is bypassed while any intervention is active.
    """
    if model._interventions.active_interventions:
        with _activation_cache_disabled():
            return _extract(model, stimuli, modality, layers)
    return _extract(model, stimuli, modality, layers)


def _extract(model, stimuli, modality: str, layers: List[str]):
    extractor = extractor_for(model, modality, stimuli)
    if extractor is not None:
        shared = model._shared_calls
        call = (shared[1].get(id(extractor))
                if shared is not None and shared[0] is stimuli else None)
        if call is not None:
            return call(stimuli, layers)
        return extractor(stimuli, layers=layers)
    modality = canonical_modality(modality)
    raise NotImplementedError(
        f"Subject '{model.identifier}' declares {modality!r} input but has no "
        'activation extractor for it. Pass extractor={modality: extractor} to '
        'record internal activity, or use a behavioral task for an output-only model.')


def extract_modalities(model, stimuli, detected: Set[str], layers: List[str]):
    """Invoke every detected-and-supported wrapper and concat assemblies."""
    import numpy as np
    import xarray as xr

    recorder = model._recorder
    fused = fused_for(model, detected)
    if fused is not None:
        # One fused extractor takes every input type: one forward pass.
        assembly = fused(stimuli, layers=layers)
        if recorder.is_multi_region:
            assembly = recorder.tag_neuroids_with_regions(assembly)
        return assembly
    priority = model.MODALITY_PRIORITY
    ordered = sorted(detected, key=lambda m: (
        priority.index(m) if m in priority else len(priority)))

    sub_assemblies = []
    for modality in ordered:
        if not _declares(model, modality):
            continue
        modality_layers = recorder.filter_layers_for_modality(layers, modality)
        if (model._region_modality_map
                and not modality_layers
                and recorder.recording_regions):
            continue
        sub = (recorder.process_composite_regions(stimuli, modality)
               if recorder.composite_recording else
               extract_for_modality(model, stimuli, modality, modality_layers))
        if not hasattr(sub, 'dims') or 'neuroid' not in sub.dims:
            raise TypeError(
                f"multi_modality dispatch requires every extractor "
                f"to return an xarray-shaped assembly with a 'neuroid' "
                f"dim; modality '{modality}' returned "
                f"{type(sub).__name__}. Pass a layer-aware extractor "
                f"(TextWrapper / VLMVisionWrapper) via "
                f"extractor={{modality: extractor}}."
            )
        if recorder.is_multi_region and not recorder.composite_recording:
            sub = recorder.tag_neuroids_with_regions(sub, modality)
        n_neuroid = sub.sizes['neuroid']
        sub = sub.assign_coords(
            modality=('neuroid', np.array([modality] * n_neuroid))
        )
        sub_assemblies.append(sub)

    if not sub_assemblies:
        raise ValueError(
            f"multi_modality=True but no detected modality had a "
            f"matching extractor. detected={detected}, "
            f"available={set(model._input_keys)}."
        )

    return _concat_modalities_along_neuroid(sub_assemblies, xr)


class _SharedCall:
    """One extractor call that answers several requests of the same stimuli.

    For an extractor that declares ``independent_layers = True``: its output
    for some layers, restricted to the neuroids of a subset of them, equals its
    output for that subset alone (values, coordinates, attributes). The first
    request runs every listed layer once; each request then gets its own copy
    of its layers' neuroids. A request is passed to the extractor unchanged
    when it names a layer outside the list, or when the shared output does not
    give its layers, by a per-neuroid ``layer`` coordinate, in the order the
    request names them (an extractor that orders layers otherwise, or that
    omits one, is called as before). When the listed layers have different
    dtypes, one assembly cannot keep each layer's dtype: the extractor refuses
    the shared call (``LayerDtypesDiffer``) and every request is called alone.
    """

    def __init__(self, extractor, layers) -> None:
        self.extractor = extractor
        self.layers = list(layers)
        self.output = None
        self.shared = True

    def __call__(self, stimuli, layers):
        from brainscore.model_helpers._extractors import LayerDtypesDiffer
        if self.shared and set(layers) <= set(self.layers):
            if self.output is None:
                try:
                    self.output = self.extractor(stimuli, layers=self.layers)
                except LayerDtypesDiffer:
                    self.shared = False
                    return self.extractor(stimuli, layers=layers)
            part = _layers_of(self.output, list(layers))
            if part is not None:
                return part
        return self.extractor(stimuli, layers=layers)


def _layers_of(output, layers):
    """The neuroids of ``layers`` in ``output``, or None when ``output`` cannot
    show them in that order. A new assembly with its own attributes."""
    from copy import deepcopy
    import numpy as np
    try:
        labels = output['layer']
    except (KeyError, TypeError):
        return None
    if labels.dims != ('neuroid',):
        return None
    labels = np.asarray(labels.values)
    positions = np.flatnonzero(np.isin(labels, layers))
    if list(dict.fromkeys(labels[positions].tolist())) != layers:
        return None
    part = output.isel(neuroid=positions)
    index = part.indexes.get('neuroid')
    if index is not None and getattr(index, 'nlevels', 1) > 1:
        # Index the part as a call for these layers alone builds its index:
        # from its own neuroids, not the levels of the shared call.
        import pandas as pd
        part = part.assign_coords(neuroid=pd.MultiIndex.from_arrays(
            [index.get_level_values(level) for level in range(index.nlevels)], names=index.names))
    part.attrs = deepcopy(dict(output.attrs))
    return part


def _concat_modalities_along_neuroid(sub_assemblies, xr):
    """Concat modality assemblies without letting presentation indexes collide."""
    from brainscore_core.time_axis import require_one_time_axis
    require_one_time_axis(sub_assemblies)
    flattened = [
        _reset_presentation_index(assembly)
        for assembly in sub_assemblies
    ]
    combined = xr.concat(flattened, dim='neuroid')
    presentation_coords = [
        name for name, coord in combined.coords.items()
        if name != 'presentation' and coord.dims == ('presentation',)
    ]
    if presentation_coords:
        combined = combined.set_index(presentation=presentation_coords)
    return combined


def _reset_presentation_index(assembly):
    """Expose presentation MultiIndex levels as plain coords for concat."""
    if 'presentation' not in getattr(assembly, 'indexes', {}):
        return assembly
    index = assembly.indexes['presentation']
    if index.__class__.__name__ != 'MultiIndex':
        return assembly
    return assembly.reset_index('presentation')


