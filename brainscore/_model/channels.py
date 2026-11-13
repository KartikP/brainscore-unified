"""Record requested neural channels, each exactly as it would be recorded alone.

A session may request regions that different extractors record (a video
region and an audio region), or several regions of one extractor. Outputs of
different extractors have their own time axes and coordinates, so they are
never merged here: every channel runs the standalone path, and its payload
equals the payload of that channel requested alone. Channels run in the
model's usual source order.

Regions that one extractor records share its call when the extractor declares
``independent_layers = True``: the extractor runs once on the session's
stimuli for the layers of all those regions, and each channel's own request
gets its layers' part of that output (``InputDispatcher.one_call_per_extractor``).
Every other step of the standalone path still runs per channel. Regions of an
extractor without the declaration run one call each, as before.
"""

from typing import Dict, List, Sequence, Tuple

from brainscore_core.io_catalog import canonical_modality


def run_neural_channels(model, stimuli, channel_regions: Sequence[Tuple[str, str]],
                        time_bins=None, multi_modality: bool = False) -> Dict[str, object]:
    """Return ``{channel: payload}`` for the requested ``neural:<region>`` channels."""
    regions = [region for _, region in channel_regions]
    _refuse_duplicates([channel for channel, _ in channel_regions])
    _start(model, regions[0] if len(regions) == 1 else regions, time_bins)
    state = model._recorder.snapshot()   # the combined recording, as before
    try:
        payloads = {}
        order = _plan(model, stimuli, regions)
        with model._dispatcher.one_call_per_extractor(stimuli, _shared_layers(model, stimuli, order)):
            for region in order:
                _start(model, region, time_bins)
                output = _process(model, stimuli, multi_modality)
                check_layer_provenance(model, output, region)
                payloads[region] = output
        return {channel: payloads[region] for channel, region in channel_regions}
    finally:
        model._recorder.restore(state)


def _refuse_duplicates(channels: List[str]) -> None:
    repeated = sorted({channel for channel in channels if channels.count(channel) > 1})
    if repeated:
        raise ValueError(
            f'Duplicate output channels requested: {repeated}. Request each '
            'neural:<region> channel once.')


def _plan(model, stimuli, regions: List[str]) -> List[str]:
    """The order in which ``regions`` run. Nothing runs here.

    Every input is checked first. Regions run in the model's source order
    (``MODALITY_PRIORITY`` of the input each reads, then request order), the
    order in which a combined recording would run its extractors.
    """
    detected = model.detect_modalities(stimuli)
    routing = model._region_modality_map
    modality_of = {region: canonical_modality(routing[region])
                   for region in regions if region in routing}
    missing = set(modality_of.values()) - set(detected)
    if missing:
        raise ValueError(
            f'Recording targets require missing input modalities: {sorted(missing)}.')

    priority = list(model.MODALITY_PRIORITY)
    rank = lambda modality: (priority.index(modality) if modality in priority
                             else len(priority))

    def source_rank(region):
        modality = modality_of.get(region)
        return rank(modality) if modality is not None else min(
            (rank(m) for m in detected), default=len(priority))

    return sorted(regions, key=lambda r: (source_rank(r), regions.index(r)))


def _shared_layers(model, stimuli, regions: List[str]) -> List[Tuple[object, List[str]]]:
    """``(extractor, layers)`` for each extractor that records two or more of ``regions``
    and declares ``independent_layers``.

    The extractor of a region is the one dispatch selects for the input the
    region reads (``region_modality``, then ``extractor_for``), so two
    registration keys that hold different extractors never share. A region
    whose input is not named, when several inputs could serve it, shares nothing.
    """
    from .dispatch import extractor_for
    from .recording import region_modality, region_selector
    groups: Dict[int, Tuple[object, List[str], int]] = {}
    for region in regions:
        modality = region_modality(model, region)
        extractor = None if modality is None else extractor_for(model, modality, stimuli)
        if extractor is None or not _declares_independent_layers(extractor):
            continue
        _, layers, count = groups.get(id(extractor), (extractor, [], 0))
        layers.extend(path for path in region_selector(model, region).layer_paths
                      if path not in layers)
        groups[id(extractor)] = (extractor, layers, count + 1)
    return [(extractor, layers) for extractor, layers, count in groups.values() if count > 1]


def _declares_independent_layers(extractor) -> bool:
    try:
        return extractor.independent_layers is True
    except AttributeError:
        return False


def check_layer_provenance(model, output, region: str) -> None:
    """Every neuroid names a layer the region records, and every required layer has one."""
    import numpy as np

    allowed, required = _region_layers(model, region)
    if not hasattr(output, 'dims') or 'neuroid' not in output.dims:
        raise ValueError(f"Neural output for region {region!r} has no 'neuroid' dimension.")
    for dim, index in output.indexes.items():
        if dim != 'neuroid' and 'layer' in (getattr(index, 'names', None) or ()):
            raise ValueError(
                f"Neural output for region {region!r} has a 'layer' coordinate on "
                f"{dim!r}; expected it on 'neuroid'.")
    if not output.sizes['neuroid']:
        raise ValueError(f'Neural output for region {region!r} contains no neuroids.')
    try:
        layer = output['layer']
    except KeyError:
        if len(allowed) == 1:
            return      # one layer: attribution is unambiguous
        raise ValueError(
            f"Neural output for region {region!r} has no per-neuroid 'layer' "
            'coordinate; attribution by concatenation order is not allowed.')
    if layer.dims != ('neuroid',):
        raise ValueError(
            f"Neural output for region {region!r} has a 'layer' coordinate with dims "
            f"{layer.dims}; expected ('neuroid',).")
    labels = set(np.asarray(layer.values).astype(str))
    unknown = labels - allowed
    if unknown:
        raise ValueError(
            f'Neural output for region {region!r} contains neuroids from layers '
            f'{sorted(unknown)}, which it does not record (expected {sorted(allowed)}).')
    uncovered = required - labels
    if uncovered:
        raise ValueError(
            f'Neural output for region {region!r} contains no neuroids for the '
            f'requested layers {sorted(uncovered)}.')


def _region_layers(model, region: str):
    """``(allowed, required)`` layers of a requested address.

    The selector's ``required_layers`` is the one rule for which layers must contribute.
    """
    from .recording import region_selector
    selector = region_selector(model, region)
    return set(selector.layer_paths), set(selector.required_layers)


def _start(model, target, time_bins) -> None:
    if time_bins is None:
        model.start_recording(target)
    else:
        model.start_recording(target, time_bins=time_bins)


def _process(model, stimuli, multi_modality: bool):
    if multi_modality:
        return model.process(stimuli, multi_modality=True)
    return model.process(stimuli)
