"""Input-event dispatch and modality extraction for BrainScoreModel."""

import os
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple

from ..capabilities import enabled_capabilities
from brainscore_core.events import OutputEvent
from brainscore_core.streaming import StreamEvent
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
    """The extractor for ``modality``; with a still and a clip extractor, the one
    matching the stimulus's form. None when the model has no such extractor."""
    modality = canonical_modality(modality)
    matching = [key for key in owner._extractors if canonical_modality(key) == modality]
    if len(matching) > 1 and modality == 'vision':
        wanted = 'video' if _is_clip(stimuli) else 'vision'
        matching = [key for key in matching if key == wanted] or matching
    return owner._extractors[matching[0]] if matching else None


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


class InputDispatcher:
    """Routes BrainScoreModel input events to the matching implementation."""

    def __init__(self, owner) -> None:
        self.owner = owner

    def process(self, input_event, multi_modality: bool = False) -> OutputEvent:
        """Answer ``input_event`` with the first capability, in registry order,
        that handles it. A ``StreamEvent`` and its answer are checked against
        the channel catalog."""
        owner = self.owner
        stream = isinstance(input_event, StreamEvent)
        if stream:
            from brainscore_core.io_catalog import validate, INPUT
            issues = validate(input_event.channel, input_event.payload, INPUT)
            if issues:
                raise ValueError('; '.join(issues))
        for capability in enabled_capabilities(owner):
            if capability.handles(owner, input_event, multi_modality=multi_modality):
                output = capability.process(owner, input_event, multi_modality=multi_modality)
                break
        else:
            raise TypeError(
                f"Subject '{owner.identifier}' has no registered capability for "
                f"input event type {type(input_event).__name__}."
            )
        if stream and isinstance(output, StreamEvent):
            from brainscore_core.io_catalog import validate, OUTPUT
            issues = validate(output.channel, output.payload, OUTPUT)
            if issues:
                raise ValueError('; '.join(issues))
        return output

    def _declares(self, modality: str) -> bool:
        """Whether a declared input key is ``modality`` or canonicalizes to it
        (a ``video`` key is temporal vision)."""
        keys = self.owner._input_keys
        return modality in keys or any(canonical_modality(key) == modality for key in keys)

    def detect_modalities(self, stimuli) -> Set[str]:
        owner = self.owner
        detected: Set[str] = set()
        from brainscore_core.io_catalog import column_channel
        from brainscore_core.streaming import parse_channel
        for col in stimuli.columns:
            channel = column_channel(col)
            if channel is None:
                continue
            # The channel family is the modality: vision:image and vision:video are vision.
            modality = canonical_modality(parse_channel(channel)[0])
            if self._declares(modality):
                detected.add(modality)
        return detected

    def pick_modality(self, detected: Set[str]) -> str:
        owner = self.owner
        for m in owner.MODALITY_PRIORITY:
            if m in detected:
                return m
        return sorted(detected)[0]

    def supports_layer_extraction(self, modality: str) -> bool:
        return any(canonical_modality(key) == canonical_modality(modality)
                   for key in self.owner._extractors)

    def extract_for_modality(self, stimuli, modality: str, layers: List[str]):
        # An intervention (lesion/stimulation) is a mutable forward-hook that the
        # activation cache key does not fingerprint, so a cached unchanged
        # extraction would be wrongly returned for a changed run. Bypass the
        # cache while any intervention is active.
        if self.owner._interventions.active_interventions:
            with _activation_cache_disabled():
                return self._extract_for_modality_inner(stimuli, modality, layers)
        return self._extract_for_modality_inner(stimuli, modality, layers)

    def _extract_for_modality_inner(self, stimuli, modality: str, layers: List[str]):
        owner = self.owner
        extractor = extractor_for(owner, modality, stimuli)
        if extractor is not None:
            return extractor(stimuli, layers=layers)
        modality = canonical_modality(modality)
        raise NotImplementedError(
            f"Subject '{owner.identifier}' declares {modality!r} input but has no "
            'activation extractor for it. Pass extractor={modality: extractor} to '
            'record internal activity, or use a behavioral task for an output-only model.')

    def process_multi_modality(self, stimuli, detected: Set[str],
                               layers: List[str]):
        """Invoke every detected-and-supported wrapper and concat assemblies."""
        import numpy as np
        import xarray as xr

        owner = self.owner
        fused = fused_for(owner, detected)
        if fused is not None:
            # One fused extractor takes every input type: one forward pass.
            assembly = fused(stimuli, layers=layers)
            if owner._recorder.is_multi_region:
                assembly = owner._tag_neuroids_with_regions(assembly)
            return assembly
        ordered = sorted(
            detected,
            key=lambda m: (
                owner.MODALITY_PRIORITY.index(m)
                if m in owner.MODALITY_PRIORITY
                else len(owner.MODALITY_PRIORITY)
            ),
        )

        sub_assemblies = []
        for modality in ordered:
            if not self._declares(modality):
                continue
            modality_layers = owner._filter_layers_for_modality(
                layers, modality)
            if (owner._region_modality_map
                    and not modality_layers
                    and owner._recorder.recording_regions):
                continue
            sub = (owner._process_composite_regions(stimuli, modality)
                   if owner._recorder.composite_recording else
                   owner._extract_for_modality(stimuli, modality, modality_layers))
            if not hasattr(sub, 'dims') or 'neuroid' not in sub.dims:
                raise TypeError(
                    f"multi_modality dispatch requires every extractor "
                    f"to return an xarray-shaped assembly with a 'neuroid' "
                    f"dim; modality '{modality}' returned "
                    f"{type(sub).__name__}. Pass a layer-aware extractor "
                    f"(TextWrapper / VLMVisionWrapper) via "
                    f"extractor={{modality: extractor}}."
                )
            if owner._recorder.is_multi_region and not owner._recorder.composite_recording:
                sub = owner._tag_neuroids_with_regions(sub, modality)
            n_neuroid = sub.sizes['neuroid']
            sub = sub.assign_coords(
                modality=('neuroid', np.array([modality] * n_neuroid))
            )
            sub_assemblies.append(sub)

        if not sub_assemblies:
            raise ValueError(
                f"multi_modality=True but no detected modality had a "
                f"matching extractor. detected={detected}, "
                f"available={set(owner._input_keys)}."
            )

        return _concat_modalities_along_neuroid(sub_assemblies, xr)


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


