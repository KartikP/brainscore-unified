"""How a BrainScoreModel records a region.

Region and layer resolution, the ``RecordingRequest`` state that ``record`` reads,
and the live recordings taken during the pass that answers a task or acts
in a closed loop (``open_recordings``, ``per_stimulus``, ``trial_regions``).
"""

from typing import Dict, List, Optional, Tuple, Union

from brainscore_core.io_catalog import canonical_modality
from ..selection import LayerSelector, UnitSelector

INPUT_SITE = ':input'


def region_selector(owner, region: str) -> UnitSelector:
    """The selector that records ``region``: its region map entry, or else the
    raw layer of that name. The selector answers which layers and units it uses."""
    return owner._region_layer_selectors.get(region, LayerSelector(region))


def region_modality(owner, region: str) -> Optional[str]:
    """The input ``region`` reads: the modality its map entry names, or the only
    one the extractors take. None when the region does not say and several
    inputs could serve it."""
    modality = owner._region_modality_map.get(region)
    if modality is not None:
        return canonical_modality(modality)
    modalities = {canonical_modality(key) for key in owner._extractors}
    if len({id(e) for e in owner._extractors.values()}) == 1:
        return sorted(modalities)[0]        # one (fused) extractor records every region
    return next(iter(modalities)) if len(modalities) == 1 else None


def record_part(extractor):
    """The extractor's ``record(layers, stimuli)``, or None for a callable without it.

    Every wrapper in ``brainscore.model_helpers`` has it (the ``Extractor``
    protocol names it); a plain function or the legacy ``PytorchWrapper`` does
    not, and records during a task only through hooks on ``model=``.
    """
    try:
        return extractor.record
    except AttributeError:
        return None


def records_while_answering(owner, region: str, stimuli=None) -> bool:
    """Whether ``region`` can be recorded during the pass that answers or acts.

    The run passes its stimulus table (None for a policy step); pre-flight
    passes nothing and takes each input the region is answered from
    (``output_inputs``). For each, the extractor is the one the run selects
    (``dispatch.extractor_keys``). It records when it has
    ``record(layers, stimuli)``, or when every layer of the region is a module
    of the network passed as ``model=`` (forward hooks). An extractor for
    another input does not count.
    """
    from .dispatch import extractor_for, extractor_keys
    from brainscore.model_helpers._extractors import get_layer
    modality = region_modality(owner, region)
    if modality is None:
        return False
    layers = region_selector(owner, region).layer_paths
    hooks = owner._model is not None
    if hooks:
        try:
            for layer in layers:
                get_layer(owner._model, layer)
        except ValueError:
            hooks = False
    if stimuli is not None:
        extractors = [extractor_for(owner, modality, stimuli, layers)]
    else:
        listed = owner.output_inputs.get(f'neural:{region}') or (None,)
        extractors = [owner._extractors[keys[0]] if keys else None
                      for keys in (extractor_keys(owner, modality, layers,
                                                  None if c is None else {c})
                                   for c in listed)]
    return any(e is not None and (record_part(e) is not None or hooks) for e in extractors)


def plan_recordings(owner, stimuli, layers):
    """The recording groups of ``layers`` during an answering pass or a policy step.

    Returns ``((extractor, group layers, regions), ...)``. Each region is
    recorded by the extractor the run selects for it (``dispatch.extractor_keys``
    for its input and its own layers, as for a passive recording), so regions on
    an image and a video extractor are recorded by each; the regions on one
    extractor share one group. A region that no extractor can record during the
    pass is refused before any compute (``ExecutionNotFeasibleError``). A
    session decides this once, in its plan, for its table (``stimuli``; None for
    a policy step).
    """
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    from .dispatch import extractor_for
    cannot = [region for region in layers
              if not records_while_answering(owner, region, stimuli)]
    if cannot:
        raise ExecutionNotFeasibleError(
            f'Subject {owner.identifier!r} cannot record {cannot} while it answers or acts: '
            'the extractor of each region needs record(layers, stimuli), or every layer of the region '
            'must be a module of the network passed as BrainScoreModel(model=...).',
            channels={f'neural:{region}' for region in cannot})
    groups: Dict[tuple, Tuple[object, List[str], List[str]]] = {}
    for region, region_layers in layers.items():
        modality = region_modality(owner, region)
        # The key keeps the modality: a fused extractor under two modalities stays two groups.
        extractor = extractor_for(owner, modality, stimuli, region_layers)
        _, group_layers, regions = groups.setdefault((modality, id(extractor)), (extractor, [], []))
        group_layers.extend(layer for layer in region_layers if layer not in group_layers)
        regions.append(region)
    return tuple((extractor, tuple(group_layers), tuple(regions))
                 for extractor, group_layers, regions in groups.values())


def planned_recordings(owner, layers):
    """The recording groups the running session's plan made for ``layers``, or None."""
    plan = owner._session_plan
    if plan is None or plan.recordings is None or tuple(layers) != plan.regions:
        return None
    return plan.recordings


def open_recordings(owner, stimuli, layers, groups=None):
    """``{regions: live recording}``, opened before any compute.

    ``groups`` are the session plan's (``plan_recordings``); without them they
    are made here. Each group is recorded by its extractor's own
    ``record(layers, stimuli)`` when it has one (``stimuli`` is None for a policy
    step), otherwise by forward hooks on the model passed as
    ``BrainScoreModel(model=...)``, with layer paths read from that model. The
    regions on one extractor share one recording, keyed by the tuple of their names.
    """
    groups = plan_recordings(owner, stimuli, layers) if groups is None else groups
    recordings = {}
    from . import parts
    for extractor, group_layers, regions in groups:
        record = record_part(extractor)
        recordings[tuple(regions)] = (record(list(group_layers), stimuli) if record is not None else
                                      _hook_recording(owner._model, list(group_layers), owner.identifier))
        # The recording runs during the answering pass: listed for the regions it records.
        parts.note('extractor' if record is not None else 'model hooks',
                   extractor if record is not None else owner._model, group_layers,
                   serves=[f'neural:{region}' for region in regions])
    return recordings


def _needs_path(cell) -> bool:
    """A string that is a local file name which does not open as it is.

    A string with a URI scheme of two or more characters (``http:``, ``https:``, ``data:``,
    ``file:``) is not a local name and is never replaced. One letter before a colon is read
    as a drive letter of a local path, not as a scheme.
    """
    import os
    import re
    return (isinstance(cell, str) and not re.match(r'^[A-Za-z][A-Za-z0-9+.\-]+:', cell)
            and not os.path.exists(cell))


def openable_paths(stimuli, task=None):
    """``stimuli`` as the parts of a model see them: a picture, clip or sound cell is a path that opens.

    An old stimulus set keeps a bare file name in its column and the file's place in its
    path map (``stimulus_paths``, read by ``get_stimulus``), as the extractors read it. Only
    a cell used as media is changed: a column the task presents as text (``present`` entry
    ``'text:<column>'``) keeps its value; a cell that already opens, and a string with a URI
    scheme, stay. The map holds one path for a stimulus, so it is given to a media cell only
    when the row has exactly one media cell that needs a path; with two or more nothing is
    resolved (no guess). The table that is stored with the answer is not this one: it keeps
    the stimulus set's own values. The same table when nothing changes.
    """
    from brainscore_core.io_catalog import stimulus_columns
    from brainscore_core.responses import PART_CHANNELS
    if getattr(stimuli, 'stimulus_paths', None) is None or 'stimulus_id' not in stimuli.columns:
        return stimuli
    path_channels = {PART_CHANNELS[kind] for kind in ('image', 'video', 'audio')}
    as_text = {entry.split(':', 1)[1] for entry in (getattr(task, 'present', None) or ())
               if ':' in entry and entry.split(':', 1)[0] not in ('image', 'video', 'audio')}
    columns = [column for column, channel in stimulus_columns().items()
               if channel in path_channels and column in stimuli.columns and column not in as_text]
    shown = None
    for position, stimulus_id in enumerate(stimuli['stimulus_id'].values):
        needing = [column for column in columns if _needs_path(stimuli[column].values[position])]
        if len(needing) != 1:
            continue
        try:
            path = str(stimuli.get_stimulus(stimulus_id))
        except KeyError:
            continue
        shown = stimuli.copy() if shown is None else shown
        shown.iloc[position, shown.columns.get_loc(needing[0])] = path
    return stimuli if shown is None else shown


def per_stimulus(owner, stimuli, layers, call, prompts=None, shown=None):
    """Call ``call(row, prompt)`` once per stimulus, recording ``layers`` on each first pass.

    Each trial's recording is compared with the first trial's as soon as it is
    taken; a difference in coordinates, which only the extractor's result
    shows, is refused after the pass that shows it, before the next stimulus.
    """
    from contextlib import ExitStack
    recordings = open_recordings(owner, stimuli, layers, planned_recordings(owner, layers))
    replies, trials = [], []
    with ExitStack() as stack:
        live = {regions: stack.enter_context(r) for regions, r in recordings.items()}
        shown_rows = [row for _, row in (stimuli if shown is None else shown).iterrows()]
        for index, (_, row) in enumerate(stimuli.iterrows()):
            replies.append(call(shown_rows[index], None if prompts is None else prompts[index]))
            trials.append({regions: recording.take(row) for regions, recording in live.items()})
            for regions, taken in trials[-1].items():
                for layer, trial in taken.items():
                    _same_axes(trials[0][regions].get(layer), trial)
    return replies, trial_regions(owner, trials, stimuli, layers)


def trial_regions(owner, trials, stimuli, layers):
    """Per-trial recordings -> one NeuroidAssembly per region (composite regions gathered)."""
    neural = {}
    for region, region_layers in layers.items():
        recorded = [next(taken for regions, taken in trial.items() if region in regions)
                    for trial in trials]
        needed = region_selector(owner, region).required_layers
        # Only required layers: a layer whose unit subset is empty contributes no
        # units (UnitSelector.required_layers). Each layer is packaged on its own,
        # so each keeps its dtype.
        per_layer = [_trials_to_assembly(recorded, stimuli, layer, region)
                     for layer in region_layers if layer in needed]
        neural[region] = gather_region(owner, region, per_layer)
    return neural


def gather_region(owner, region, per_layer):
    """A region's units: one layer as is; a composite region's units gathered from its layers.

    Every layer the region requires must have contributed neuroids, as in
    the neural session path, whichever path (readout or generation) recorded.
    """
    import numpy as np
    import xarray as xr
    selector = region_selector(owner, region)
    needed = selector.required_layers
    present = {str(layer) for block in per_layer if block.sizes['neuroid']
               for layer in np.asarray(block['layer'].values)}
    missing = [layer for layer in needed if layer not in present]
    if missing or not present:
        raise ValueError(
            f'Region {region!r} needs neuroids from every selected layer '
            f'{list(needed)}; the extractor returned no neuroids for '
            f'{missing or "any of its layers"}.')
    if not selector.gathers_units:
        return per_layer[0]
    from brainscore_core.time_axis import require_one_time_axis
    require_one_time_axis(per_layer, sources=[f'{region}:layer{i}' for i in range(len(per_layer))])
    combined = xr.concat([RecordingRequest.ensure_unit_indices(a) for a in per_layer], dim='neuroid')
    selected = selector.gather_units(combined)
    if not selected.sizes['neuroid']:
        raise ValueError(f'Region {region!r} selects no neuroids from its layers.')
    return selected


def _hook_recording(model, layers, identifier):
    """Forward hooks on ``model`` for extractors without ``record()``.

    ``take()`` returns ``{layer: assembly}`` for the first pass since the
    previous take, each layer's output flattened per presentation, as a
    still-image extractor packages it.
    """
    from brainscore.model_helpers._extractors import LiveRecording, as_float32, layer_assembly

    return LiveRecording(model, model, layers, as_float32, lambda activity, _read, _row: {
        layer: layer_assembly(identifier, layer, values.reshape(1, -1)) for layer, values in activity.items()})


def _same_axes(first, trial) -> None:
    """Refuse a trial whose recording differs from the first trial's in sizes, in
    coordinate names or their dimensions, or in coordinate values off the
    presentation axis: one assembly keeps one axis, as the passive recording does."""
    import numpy as np
    import pandas as pd
    if first is None or trial is None:
        return                                              # _trials_to_assembly names the layer
    if trial.sizes != first.sizes:
        raise ValueError('Trials gave different numbers of steps or parts; present trials of one '
                         'length (clips of one duration, sequences of one length) together.')

    def coordinates(assembly):                              # MultiIndex levels too
        named = {name: coord.dims for name, coord in assembly.coords.items()}
        for dim, index in assembly.indexes.items():
            named.update({level: (dim,) for level in index.names if isinstance(index, pd.MultiIndex)})
        return named
    named, other = coordinates(first), coordinates(trial)
    for name, dims in {**other, **named}.items():
        if other.get(name) != named.get(name) or 'presentation' not in dims and not pd.Index(
                np.asarray(trial[name].values).ravel()).equals(pd.Index(np.asarray(first[name].values).ravel())):
            raise ValueError(f'Trials gave different {name!r} coordinates on {dims!r}; one assembly '
                             'keeps one axis, as the passive recording does.')


def _trials_to_assembly(trials, stimuli, layer: str, region: str):
    """Stack the one-presentation recordings of ``layer`` into a NeuroidAssembly for ``region``.

    The neuroid and extra-axis coordinates come from the first trial, and every
    presentation coordinate the extractor gave (a clip's path) is kept; the
    extractor gives every trial of a session the same extra axis (a clip's clock).
    """
    import numpy as np
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    trials = [trial.get(layer) for trial in trials]
    for index, trial in enumerate(trials):
        if trial is None or not trial.sizes['neuroid']:
            raise ValueError(
                f'No activity reached layer {layer!r} while answering stimulus '
                f'{stimuli["stimulus_id"].iloc[index]!r}; that input may not use it.')
    first = trials[0]
    # A clip keeps its time steps and a multi-image trial its parts: one extra dimension.
    extra = [dim for dim in first.dims if dim not in ('presentation', 'neuroid')]
    if len(extra) > 1:
        raise ValueError(f'A recording during a task may have one extra dimension; got {extra}.')
    for trial in trials[1:]:
        _same_axes(first, trial)
    dims = ['presentation', *extra, 'neuroid']

    def names(dim):         # MultiIndex levels and ordinary coordinates, the dimension's own (time_bin) too
        index = first.indexes.get(dim)
        listed = list(index.names) if isinstance(index, pd.MultiIndex) else []
        return listed + [name for name, coord in first.coords.items() if coord.dims == (dim,) and name not in listed
                         and (name != dim or not listed and dim != 'presentation')]
    coords = {name: ('neuroid', np.asarray(first[name].values)) for name in names('neuroid')}
    coords['region'] = ('neuroid', [region] * first.sizes['neuroid'])
    for column in stimuli.columns:
        coords[column] = ('presentation', list(stimuli[column].values))
    coords.update({name: ('presentation', [trial[name].values[0] for trial in trials])
                   for name in names('presentation') if name not in stimuli.columns})
    coords.update({name: (dim, np.asarray(first[name].values)) for dim in extra for name in names(dim)})
    rows = [np.asarray(trial.transpose(*dims).values)[0] for trial in trials]
    return NeuroidAssembly(np.stack(rows), coords=coords, dims=dims)


def declared_layers(owner, modality: Optional[str] = None):
    """Layer paths the extractors for ``modality`` (all, when None) declare.

    An extractor declares its layer namespace through an optional public
    ``layer_names()`` (paths relative to the module it wraps). Returns None when
    there is no extractor or any candidate declares nothing (e.g. a plain
    callable), so validation never rejects a layer it cannot see.
    """
    wanted = canonical_modality(modality) if modality else None
    candidates = {id(e): e for key, e in owner._extractors.items()
                  if wanted is None or canonical_modality(key) == wanted}
    if not candidates:
        return None
    names = set()
    for extractor in candidates.values():
        try:
            layer_names = extractor.layer_names
        except AttributeError:
            return None
        names.update(layer_names())
    return names


def unknown_layer_message(owner, path: str, modality: Optional[str] = None) -> Optional[str]:
    """Why ``path`` is not a layer of the routed extractor, or None if it is
    (or cannot be checked). ``':input'`` sites check their module path."""
    names = declared_layers(owner, modality)
    base = path[:-len(INPUT_SITE)] if path.endswith(INPUT_SITE) else path
    if names is None or base in names:
        return None
    import difflib
    # A full path given where the extractor wraps a sub-module ends with a valid one.
    suffixes = sorted((n for n in names if base.endswith('.' + n)), key=len, reverse=True)
    close = list(dict.fromkeys(suffixes + difflib.get_close_matches(base, names, n=5, cutoff=0.6)))[:5]
    tower = f'the {modality!r} extractor' if modality else 'the extractor'
    hint = (f'Close matches: {close}.' if close else
            f'Some valid layers: {sorted(names)[:10]} ({len(names)} in all; '
            'see extractor.layer_names()).')
    return f'{path!r} is not a layer of {tower} of {owner.identifier!r}. {hint}'


def validate_region_layers(owner, selectors) -> None:
    """Fail at registration when a region maps to a layer its extractor lacks."""
    for region, selector in selectors.items():
        for path in selector.layer_paths:
            message = unknown_layer_message(owner, path, selector.modality)
            if message:
                raise ValueError(f'region_layer_map[{region!r}]: {message}')
        if owner._extractors and not region_extractor_keys(owner, selector):
            raise ValueError(f'region_layer_map[{region!r}]: no one extractor has all of '
                             f'the layers {list(selector.layer_paths)}.')


def region_extractor_keys(owner, selector) -> List[str]:
    """The extractor keys a region is recorded from (``dispatch.extractor_keys``
    for its modality and layers). ``output_inputs`` lists the input channel of
    each, so declarations, pre-flight and the run agree on the forms of a region,
    except when the run takes the fused path (``dispatch.fused_for``)."""
    from .dispatch import extractor_keys
    return extractor_keys(owner, selector.modality, selector.layer_paths)


class RecordingRequest:
    """What a BrainScoreModel is asked to record: the regions, their layers and the time bins."""

    def __init__(self, owner) -> None:
        self.owner = owner
        self.recording_layer: Optional[str] = None
        self.recording_regions: List[str] = []
        self.recording_layers: List[str] = []
        self.is_multi_region: bool = False
        self.time_bins: Optional[List[Tuple[int, int]]] = None
        self.composite_recording: bool = False

    def start_recording(self,
                        recording_target: Union[str, List[str]],
                        time_bins: Optional[List[Tuple[int, int]]] = None,
                        recording_type: Optional[str] = None) -> None:
        """Configure the layers extracted by subsequent ``process`` calls.

        A string names one region, and a list records several targets in one
        extraction. A name that is not a region is read as a layer path: when
        the extractor lists its layers, a name that is not one of them raises
        ``ValueError`` (with close matches); when it lists none, the name is
        used as given, with a warning. Strings and lists follow the same rule.
        ``recording_type`` is accepted for legacy adapter compatibility and is
        otherwise ignored by ``BrainScoreModel``.
        """
        del recording_type
        owner = self.owner

        targets = ([recording_target] if isinstance(recording_target, str)
                   else list(recording_target))
        if not targets or any(not isinstance(target, str) or not target for target in targets):
            raise ValueError('Recording targets must be nonempty region or layer names.')
        import warnings
        for target in targets:
            if target in owner._region_layer_selectors:
                continue
            message = unknown_layer_message(owner, target)
            if message:
                raise ValueError(
                    f"Recording target {target!r} is not a region in region_layer_map "
                    f"(known regions: {sorted(owner._region_layer_selectors)}). As a layer "
                    f"path: {message}")
            if declared_layers(owner) is None:
                warnings.warn(
                    f"Recording target {target!r} is not a region in region_layer_map; "
                    "treating it as a raw layer path. If you meant a region, check the spelling.",
                    UserWarning, stacklevel=2)
        selectors = [region_selector(owner, target) for target in targets]
        self.recording_regions = targets
        self.recording_layers = list(dict.fromkeys(
            layer for selector in selectors for layer in selector.layer_paths))
        self.recording_layer = self.recording_layers[0] if len(self.recording_layers) == 1 else None
        self.composite_recording = any(selector.gathers_units for selector in selectors)
        self.is_multi_region = len(targets) > 1
        self.time_bins = time_bins

    def reset(self) -> None:
        self.recording_layer = None
        self.recording_layers = []
        self.recording_regions = []
        self.is_multi_region = False
        self.composite_recording = False
        self.time_bins = None

    def snapshot(self):
        """Copy all recording configuration, including raw-layer and idle states."""
        from copy import deepcopy
        return {key: deepcopy(value) for key, value in vars(self).items()
                if key != 'owner'}

    def restore(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    def regions_for_modality(self, modality):
        if modality is None:
            return list(self.recording_regions)
        routing = self.owner._region_modality_map
        return [region for region in self.recording_regions
                if region not in routing
                or canonical_modality(routing[region]) == canonical_modality(modality)]

    def current_layers(self) -> List[str]:
        if self.recording_layers:
            return list(self.recording_layers)
        if self.recording_layer:
            return [self.recording_layer]
        return []

    def filter_layers_for_modality(self, layers: List[str],
                                   modality: str) -> List[str]:
        """Return the subset of ``layers`` that belong to ``modality``."""
        owner = self.owner
        if not owner._region_modality_map or not self.recording_regions:
            return list(layers)
        allowed = {layer for region in self.regions_for_modality(modality)
                   for layer in region_selector(owner, region).layer_paths}
        return [layer for layer in layers if layer in allowed]

    def tag_neuroids_with_regions(self, assembly, modality=None):
        """Map recorded layer provenance to a per-neuroid ``region`` coord."""
        import numpy as np

        assembly = self.ensure_layer_provenance(
            assembly,
            self.filter_layers_for_modality(self.current_layers(), modality)
            if modality is not None else self.current_layers(),
            description="Multi-region neural output",
        )
        layer_to_regions: Dict[str, List[str]] = {}
        for region in self.regions_for_modality(modality):
            for layer in region_selector(self.owner, region).layer_paths:
                layer_to_regions.setdefault(layer, []).append(region)

        neuroid_layers = assembly['layer'].values
        neuroid_regions = np.array([
            '|'.join(layer_to_regions.get(layer, []))
            for layer in neuroid_layers
        ])
        return assembly.assign_coords(region=('neuroid', neuroid_regions))

    @staticmethod
    def ensure_layer_provenance(assembly, layer_paths, description):
        """Require or safely add per-neuroid layer attribution.

        A missing coordinate can only be filled when every neuroid necessarily
        came from one layer. With several requested layers, assigning by position
        would recreate the unstable-order attribution bug, so fail explicitly.
        """
        import numpy as np

        if not hasattr(assembly, 'dims') or 'neuroid' not in assembly.dims:
            raise ValueError(
                f"{description} has no 'neuroid' dimension and cannot carry "
                "per-neuroid layer provenance."
            )
        try:
            layer_coord = assembly['layer']
        except KeyError:
            unique_layers = list(dict.fromkeys(layer_paths))
            if len(unique_layers) != 1:
                raise ValueError(
                    f"{description} has no per-neuroid 'layer' coordinate for "
                    f"requested layers {unique_layers}; attribution by "
                    "concatenation order is not allowed."
                )
            assembly = assembly.assign_coords(
                layer=('neuroid', np.array(
                    [unique_layers[0]] * assembly.sizes['neuroid']))
            )
            layer_coord = assembly['layer']
        if layer_coord.dims != ('neuroid',):
            raise ValueError(
                f"{description} has a 'layer' coordinate with dims "
                f"{layer_coord.dims}; expected ('neuroid',)."
            )
        return assembly

    def process_composite_regions(self, stimuli, modality, check_only=False):
        """Extract and concatenate blocks for composite recording regions.

        ``check_only``: refuse a region with no extractor, and call none.
        """
        import numpy as np
        import xarray as xr

        from .dispatch import extract_for_modality, extractor_or_refuse
        owner = self.owner
        blocks = []
        for region in self.regions_for_modality(modality):
            selector = region_selector(owner, region)
            layer_paths = list(selector.layer_paths)
            if check_only:
                extractor_or_refuse(owner, stimuli, modality, layer_paths, in_session=True)
                continue
            assembly = extract_for_modality(owner, stimuli, modality, layer_paths)
            assembly = self.ensure_layer_provenance(
                assembly,
                layer_paths,
                description=f"Composite region {region!r} output",
            )
            labels = set(np.asarray(assembly['layer'].values).astype(str))
            needed = selector.required_layers
            missing = [layer for layer in needed if layer not in labels]
            if missing or labels - set(layer_paths):
                raise ValueError(
                    f"Composite region {region!r} needs neuroids from every selected "
                    f"layer {list(needed)} (and none outside {layer_paths}); the "
                    f"extractor returned layers {sorted(labels)}.")
            assembly = self.ensure_unit_indices(assembly)
            block = selector.gather_units(assembly)
            n = block.sizes['neuroid']
            block = block.assign_coords(
                region=('neuroid', np.array([region] * n)))
            blocks.append(block)
        if check_only:
            return None
        from brainscore_core.time_axis import require_one_time_axis
        require_one_time_axis(blocks, sources=[str(b['region'].values[0]) if b.sizes['neuroid'] else '?'
                                               for b in blocks])
        return xr.concat(blocks, dim='neuroid')

    @staticmethod
    def ensure_unit_indices(assembly):
        """Attach original per-layer positions BEFORE any composite subsetting."""
        import numpy as np
        try:
            assembly['unit_index']
            return assembly
        except KeyError:
            pass
        for name in ('neuron_number_in_layer', 'neuroid_num'):
            try:
                coord = assembly[name]
            except KeyError:
                continue
            if coord.dims == ('neuroid',):
                return assembly.assign_coords(unit_index=('neuroid', coord.values))
        counts = {}
        indices = []
        for layer in assembly['layer'].values:
            indices.append(counts.get(layer, 0))
            counts[layer] = indices[-1] + 1
        return assembly.assign_coords(unit_index=('neuroid', np.asarray(indices)))
