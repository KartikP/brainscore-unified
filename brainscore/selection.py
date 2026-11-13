"""Unit-selection helpers for recording and interventions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

from brainscore_core.events import Selection


class UnitSelector(ABC):
    """Base for selectors used as ``region_layer_map`` values.

    Use ``LayerSelector`` for one layer or ``CompositeSelector`` for units
    across layers. A string layer path is promoted to ``LayerSelector``.
    ``Selection`` instead describes a concrete recording/intervention target.

    A selector answers every question about its region itself: which layers it
    records (``layer_paths``), which of them must contribute units
    (``required_layers``), whether it keeps chosen units of its layers
    (``gathers_units``, ``keeps_unit_subset``) and which units those are
    (``gather_units``). The input it reads is ``modality``. Recording code asks
    the selector and never checks its class. A subclass needs only
    ``layer_path``; the defaults describe one whole layer.
    """

    modality: Optional[str] = None
    # True when the region keeps chosen units of its layers. Recording then
    # numbers each layer's units (``unit_index``) and calls ``gather_units``.
    gathers_units: bool = False

    @property
    @abstractmethod
    def layer_path(self) -> str:
        ...

    @property
    def layer_paths(self) -> Tuple[str, ...]:
        """Every layer this selector records, in order."""
        return (self.layer_path,)

    @property
    def required_layers(self) -> Tuple[str, ...]:
        """The layers that must contribute units to the region."""
        return self.layer_paths

    @property
    def keeps_unit_subset(self) -> bool:
        """True when some layer keeps only some of its units."""
        return False

    def gather_units(self, assembly):
        """The region's units of ``assembly``, a recording of ``layer_paths``."""
        return assembly


@dataclass(frozen=True)
class LayerSelector(UnitSelector):
    """Select one layer for a region mapping; ``name`` is its layer path."""
    name: str
    modality: Optional[str] = None

    @property
    def layer_path(self) -> str:
        return self.name


@dataclass(frozen=True)
class CompositeSelector(UnitSelector):
    """Select units across multiple layers as one region.

    A composite region's neuroids are gathered from several layers, each
    optionally restricted to a subset of unit positions. Used when a brain
    region is best modeled by units drawn from more than one model layer
    (for example a whole-brain parcel that pools early and late features).

    :param layers: ordered tuple of ``(layer_path, indices)`` pairs, where
        ``indices`` is a tuple of unit positions to take from that layer
        (counted within that layer's own neuroids), or ``None`` to take every
        unit of the layer.
    """
    layers: Tuple[Tuple[str, Optional[Tuple[int, ...]]], ...]
    modality: Optional[str] = None
    gathers_units = True

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("A composite selector needs at least one layer.")

    @property
    def layer_path(self) -> str:
        # Canonical (first) layer; the full set is in `layers`.
        return self.layers[0][0]

    @property
    def layer_paths(self) -> Tuple[str, ...]:
        """Every layer this selector draws from, in order."""
        return tuple(lp for lp, _ in self.layers)

    @property
    def required_layers(self) -> Tuple[str, ...]:
        """The layers that must contribute units to the region.

        A layer whose selected unit subset is empty is intentionally unused: it
        may be absent from the extractor output and is never required. This one
        rule serves the composite check before unit selection and the channel
        check after it.
        """
        return tuple(layer for layer, indices in self.layers
                     if indices is None or len(indices))

    @property
    def keeps_unit_subset(self) -> bool:
        return any(indices is not None for _, indices in self.layers)

    def gather_units(self, assembly):
        """Keep the selected units of a possibly multi-layer assembly, layer by layer."""
        import numpy as np

        layer_coord = assembly['layer'].values
        keep_positions: List[int] = []
        for layer_path, indices in self.layers:
            layer_positions = np.where(layer_coord == layer_path)[0]
            if indices is None:
                keep_positions.extend(layer_positions.tolist())
            else:
                keep_positions.extend(layer_positions[list(indices)].tolist())
        return assembly.isel(neuroid=keep_positions)


def _promote_to_selector(value: Union[str, UnitSelector]) -> UnitSelector:
    """Normalize a ``region_layer_map`` value to a ``UnitSelector``."""
    if isinstance(value, UnitSelector):
        return value
    if isinstance(value, str):
        return LayerSelector(name=value)
    raise TypeError(
        f"region_layer_map values must be str or UnitSelector, "
        f"got {type(value).__name__}"
    )


class UnitSelection(ABC):
    """Resolves to a concrete :class:`Selection` for a subject.

    :class:`FunctionalSelection` records the subject through a session, so it
    works for any subject that records the region. Resolve a selection before
    the intervention: the ``target`` of a :class:`StateChange` is a resolved
    ``Selection``::

        units = FunctionalSelection('IT', localizer, contrast=(['face'], ['object'])).resolve(subject)
        apply_state_change(subject, StateChange('ablation', target=units, perturbation=Perturbation('zero')))

    The localizer is its own session, and a subject takes part in one session
    at a time. A resolved target is also the exact set of units, so a run
    record of the intervention replays without the localizer.
    """

    @abstractmethod
    def resolve(self, subject) -> 'Selection':
        ...


def require_resolved(state_change) -> None:
    """Refuse a ``StateChange`` whose ``target`` is an unresolved ``UnitSelection``.

    The one check for every intervention entry (``apply_state_change``, the
    ``Intervene`` tool and a session that carries the change).
    """
    target = state_change.target
    if isinstance(target, UnitSelection):
        raise TypeError(
            f"The intervention target is an unresolved {type(target).__name__}. Resolve "
            f"it first, then send the change: "
            f"change = StateChange(..., target=selection.resolve(subject)). The "
            f"selection runs its own session, and a subject takes part in one session "
            f"at a time.")


@dataclass
class IndexSelection(UnitSelection):
    """Explicit unit indices — the trivial selection. ``resolve`` ignores the subject."""
    layer: str
    indices: List[int]

    def resolve(self, subject) -> 'Selection':
        return Selection(layer=self.layer, indices=list(self.indices),
                         metadata={'selector': 'index'})


@dataclass
class RandomSelection(UnitSelection):
    """``n_units`` units drawn uniformly at random from ``[0, n_total)`` — the
    causal control for any functional selection.

    Deterministic given ``seed``; needs no forward pass (``resolve`` ignores the
    subject). Match ``layer`` / ``n_units`` / ``n_total`` to a
    :class:`FunctionalSelection` to get a same-size random lesion at the same
    layer — the standard "is the effect specific, or just damage?" control.
    ``n_total`` is the layer's unit count, available as
    ``FunctionalSelection`` result ``metadata['n_recorded']``.

    For subset recordings, pass ``population`` from the functional selection's
    ``metadata['unit_population']``. Its entries are original layer indices;
    ``n_total`` is the population size, not the largest original index.
    """
    layer: str
    n_units: int
    n_total: int
    seed: int = 0
    population: Optional[List[int]] = None
    axis: Optional[int] = -1
    expected_shape: Optional[Tuple[int, ...]] = None

    @classmethod
    def matched_to(cls, selection: 'Selection', seed: int = 0) -> 'RandomSelection':
        """The random control for a resolved selection: the same layer, layout and
        number of units, drawn from the same population.

        ``RandomSelection.matched_to(units, seed=0)`` is shorthand for passing
        ``layer``, ``n_units``, ``n_total``, ``population``, ``axis`` and
        ``expected_shape`` by hand.
        """
        metadata = selection.metadata or {}
        population = metadata.get('unit_population')
        if population is not None:
            n_total = len(population)
        elif 'n_recorded' in metadata:
            n_total = int(metadata['n_recorded'])
        else:
            raise ValueError('matched_to needs a selection that records its unit population '
                             "(metadata['unit_population'] or metadata['n_recorded']), "
                             'such as a resolved FunctionalSelection.')
        return cls(layer=selection.layer, n_units=len(selection.indices), n_total=n_total,
                   seed=seed, population=None if population is None else list(population),
                   axis=selection.axis, expected_shape=selection.expected_shape)

    def resolve(self, subject) -> 'Selection':
        import numpy as np
        k = min(self.n_units, self.n_total)
        rng = np.random.RandomState(self.seed)
        population = (np.arange(self.n_total) if self.population is None
                      else _validate_unit_indices(self.population))
        if len(population) != self.n_total:
            raise ValueError('n_total must match the declared unit population.')
        idx = sorted(int(i) for i in rng.choice(population, size=k, replace=False))
        return Selection(layer=self.layer, indices=idx, axis=self.axis,
                         expected_shape=self.expected_shape,
                         metadata={'selector': 'random', 'seed': self.seed,
                                   'n_total': self.n_total,
                                   'unit_population': population.tolist()})


def _validate_unit_indices(values):
    import operator
    import numpy as np
    try:
        indices = np.asarray([operator.index(value) for value in values], dtype=int)
    except TypeError as error:
        raise ValueError('Original unit indices must be integers.') from error
    if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise ValueError('Original unit indices must be non-negative and unique within a layer.')
    return indices


def selection_unit_indices(assembly, subject=None, region=None):
    """Map recorded positions to original units in a single layer.

    Full-layer output retains the historical position convention. Subset
    recordings must carry addresses from before subsetting, never renumbered
    positions in the reduced assembly.
    """
    import numpy as np
    for name in ('unit_index', 'neuron_number_in_layer', 'neuroid_num'):
        try:
            coord = assembly[name]
        except KeyError:
            continue
        if coord.dims != ('neuroid',):
            raise ValueError(f'{name} must have dims (neuroid,) for unit selection.')
        return _validate_unit_indices(coord.values)
    selector = getattr(subject, 'region_layer_selectors', {}).get(region)
    if selector is not None and selector.keeps_unit_subset:
        raise ValueError('Subset recording has no original unit_index coordinates.')
    return np.arange(assembly.sizes['neuroid'])


@dataclass
class FunctionalSelection(UnitSelection):
    """Select units by a functional contrast — the fMRI-localizer analogue.

    Records the subject's responses to ``localizer_stimuli`` at ``region``
    through a session, splits the stimuli into a positive and a negative group
    by their ``contrast_column`` label, scores each unit by the Cohen's d of
    positive vs negative, and keeps the top ``n_units`` (or every unit whose
    score clears ``threshold`` when ``n_units`` is ``None``). This is the
    word-vs-non-word VWFA localizer from the induced-dyslexia experiment, and
    the one Cohen's d localizer in the library.

    Units are addressed by what the subject reports: each selection's layer is
    a value of the recording's per-neuroid ``layer`` coordinate, and a
    recording without that coordinate is refused. Labels come from
    ``localizer_stimuli``, matched by ``stimulus_id``, so a subject need not
    send stimulus columns back.

    :param region: the region to record (the session requests
        ``neural:<region>``).
    :param contrast: ``(positive_labels, negative_labels)`` matched against
        ``contrast_column`` of ``localizer_stimuli``.
    :param sign: ``'positive'`` keeps units that respond MORE to the positive
        group (d > 0), ``'negative'`` the opposite, ``'abs'`` the most
        discriminating either way.
    """
    region: str
    localizer_stimuli: Any
    contrast: Tuple[List[str], List[str]]
    contrast_column: str = 'label'
    n_units: Optional[int] = None
    threshold: float = 2.0
    sign: str = 'positive'

    def resolve(self, subject) -> 'Selection':
        """The selected units of a region that the subject records from one layer."""
        selections = self.resolve_layers(subject)
        if len(selections) > 1:
            raise ValueError(
                f"recording {self.region!r} spans {len(selections)} layers "
                f"{[s.layer for s in selections]}; one Selection addresses one layer. "
                f"Use resolve_layers() for one selection per layer.")
        return selections[0]

    def resolve_layers(self, subject) -> List['Selection']:
        """One selection per layer the recording reports, in recording order.

        Each layer's units are ranked on their own, so ``n_units`` applies per
        layer.
        """
        import numpy as np
        from brainscore_core.streaming_helpers import neural_response
        if self.sign not in ('positive', 'negative', 'abs'):
            raise ValueError(
                f"sign must be 'positive'|'negative'|'abs', got {self.sign!r}")
        asm = neural_response(subject, self.localizer_stimuli, [f'neural:{self.region}'])
        # collapse any extra dims (e.g. time_bin) to per-(presentation, neuroid)
        extra = [d for d in asm.dims if d not in ('presentation', 'neuroid')]
        if extra:
            asm = asm.mean(extra)
        asm = asm.transpose('presentation', 'neuroid')
        try:
            layer_coord = np.asarray(asm['layer'].values).ravel()
        except KeyError:
            raise ValueError(
                f"a functional selection addresses units by the layer the recording "
                f"reports; the {self.region!r} recording has no 'layer' coord on its "
                f"neuroids.") from None
        label_of = dict(zip(self.localizer_stimuli['stimulus_id'].values,
                            self.localizer_stimuli[self.contrast_column].values))
        labels = np.asarray([label_of[sid] for sid in asm['stimulus_id'].values])
        pos_mask = np.isin(labels, list(self.contrast[0]))
        neg_mask = np.isin(labels, list(self.contrast[1]))
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            raise ValueError(
                f"contrast {self.contrast} matched {int(pos_mask.sum())} positive "
                f"/ {int(neg_mask.sum())} negative presentations in column "
                f"{self.contrast_column!r}; both groups must be non-empty.")
        vals = asm.values
        selections = []
        for layer in dict.fromkeys(str(value) for value in layer_coord):
            columns = np.where(layer_coord == layer)[0]
            recorded = asm.isel(neuroid=columns)
            units = vals[:, columns]
            pos, neg = units[pos_mask], units[neg_mask]
            mp, mn = pos.mean(0), neg.mean(0)
            sp, sn = pos.std(0, ddof=1), neg.std(0, ddof=1)
            pooled = np.sqrt((sp ** 2 + sn ** 2) / 2.0)
            d = np.divide(mp - mn, pooled, out=np.zeros_like(mp, dtype=float),
                          where=pooled > 0)
            score = {'positive': d, 'negative': -d, 'abs': np.abs(d)}[self.sign]
            if self.n_units is not None:
                order = np.argsort(score)[::-1][:self.n_units]
            else:
                order = np.where(score >= self.threshold)[0]
            population = selection_unit_indices(recorded, subject, self.region)
            axis, expected_shape = selection_coordinates(recorded)
            indices = sorted(int(i) for i in population[order])
            selections.append(Selection(
                layer=layer, indices=indices, axis=axis, expected_shape=expected_shape,
                metadata={'selector': 'functional', 'sign': self.sign,
                          'contrast': self.contrast,
                          'n_recorded': int(units.shape[1]),
                          'unit_population': population.tolist(),
                          'n_selected': len(indices)}))
        return selections


def selection_coordinates(assembly: Any) -> Tuple[Optional[int], Optional[Tuple[int, ...]]]:
    """Recover tensor coordinates from unpooled extraction, without guessing axes.

    Vision extraction publishes the original non-batch coordinates. Text and
    feature-vector extraction use the last feature axis. Custom extractors can
    declare unit_axis and unit_shape attributes when their layout differs.
    """
    import numpy as np
    if 'unit_axis' in assembly.attrs:
        axis = assembly.attrs['unit_axis']
        return (None if axis == 'flattened' else int(axis),
                assembly.attrs.get('unit_shape'))
    coordinate_sets = (
        ('channel_temporal', 'channel_x', 'channel_y', 'channel'),
        ('channel', 'channel_x', 'channel_y'),
        ('channel', 'embedding'),
    )
    for names in coordinate_sets:
        try:
            coordinates = [assembly[name].values for name in names]
        except KeyError:
            continue
        if coordinates:
            shape = tuple(int(np.nanmax(values)) + 1
                          for values in coordinates if np.isfinite(values).any())
            # Subset coordinates cannot establish the original shape. Fail closed
            # rather than reinterpret these positions as last-axis indices.
            count = int(np.prod(shape))
            if count != assembly.sizes['neuroid']:
                raise ValueError('Subset spatial recordings require unit_axis and unit_shape metadata.')
            return None, shape
    return -1, None
