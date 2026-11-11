"""Topographic-organization benchmark.

Scores whether a model's spatial unit layout matches cortical topography — the
correlation-vs-distance profile r(d), beyond predictivity — with a matched
shuffle-coordinate NULL. The score reported is ``raw - null`` (the spatial signal above
the permutation floor), with both stored in ``attrs``. Predictivity is permutation-invariant
over units and so is blind to this; the shuffle null makes that explicit.

A *real* registered instance needs (a) a topographic model whose ``neural:<region>`` response carries
per-unit ``tissue_x``/``tissue_y`` coords (Topo-Omni's sheet, a TDANN's tissue map, or a conv
feature-map grid) and (b) a brain assembly with voxel coordinates (e.g. an fsaverage-surface
fMRI assembly). This module ships the reusable benchmark + factory; register a concrete
instance once such a surface assembly is staged (EC2). The wiring is validated offline on
synthetic assemblies in ``tests/test_topographic_benchmark.py``.
"""
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.streaming import Session, parse_channel
from brainscore_core.streaming_helpers import neural_response, stimulus_session
from brainscore import load_metric
from brainscore.topographic_support import ensure_tissue_coords, shuffle_tissue_coords

# The order in which BrainScoreModel.process picks one input modality from a
# table that holds several (BrainScoreModel.MODALITY_PRIORITY).
_INPUT_PRIORITY = ('vision', 'text', 'audio')


def _table_events(stimulus_set, region):
    """The input events of ``stimulus_set``, without the control events."""
    full = stimulus_session(stimulus_set, record=region)
    events = []
    while (event := full.next_input()) is not None:
        if event.channel != 'control':
            events.append(event)
    return events


def _record(candidate, stimulus_set, region):
    """The candidate's ``neural:<region>`` response, read through ``participate``.

    The benchmark used to call ``process(stimuli)``, which reads one input
    modality. A session whose inputs span several channel families lets a
    subject that declares them combine them, so for such a table only the
    events of the first family in ``_INPUT_PRIORITY`` that the subject
    declares are sent. The whole table stays the session's ``stimulus_set``,
    so the subject reads the same table as before. A table with one input
    family goes through ``neural_response`` unchanged.
    """
    events = _table_events(stimulus_set, region)
    present = {parse_channel(event.channel)[0] for event in events}
    declared = {parse_channel(channel)[0] for channel in candidate.in_channels}
    chosen = next((family for family in _INPUT_PRIORITY
                   if family in present and family in declared), None)
    if len(present) < 2 or chosen is None:
        return neural_response(candidate, stimulus_set, record=region)
    channel = f'neural:{region}'
    session = Session(
        [event for event in events if parse_channel(event.channel)[0] == chosen],
        (channel,), stimulus_set=stimulus_set)
    candidate.participate(session)
    return session.collect(channel)


class TopographicBenchmark(BenchmarkBase):
    """Reusable topographic-alignment benchmark.

    :param brain_assembly: target measurements with voxel coordinates
        (``tissue_x``/``tissue_y`` or ``x``/``y``/``z`` neuroid coords).
    :param stimulus_set: stimuli to drive the candidate (same stimuli the brain saw).
    :param region: the region requested from the candidate as ``neural:<region>``.
    :param metric_id: registered metric (default the topographic-alignment metric).
    :param null_seed: seed for the shuffle-coordinate null.

    It sends the input channels of the columns of ``stimulus_set`` and asks for
    ``neural:<region>``. Each stimulus is an independent trial.
    """

    trial_protocol = 'independent'

    def __init__(self, identifier, brain_assembly, stimulus_set, region, *,
                 metric_id='topographic-alignment', null_seed=0,
                 parent='neural', bibtex=None, ceiling=None):
        super().__init__(identifier=identifier, ceiling=ceiling or Score(1.0),
                         version=1, parent=parent, bibtex=bibtex)
        self._brain = brain_assembly
        self._stimulus_set = stimulus_set
        self._region = region
        self.required_input_channels = frozenset(
            event.channel for event in _table_events(stimulus_set, region))
        self.requested_output_channels = frozenset({f'neural:{region}'})
        self._metric = load_metric(metric_id)
        self._null_seed = null_seed

    def __call__(self, candidate) -> Score:
        # ponytail: the tissue_x/tissue_y neuroid coords reach this line only
        # because a neural:<region> payload passes through the session as the
        # subject built it. The neural catalog entry declares no meta keys or
        # coords for unit positions; if it ever validates or strips coords,
        # declare tissue_x/tissue_y there. Guarded by the tissue-coord tests in
        # tests/test_topographic_benchmark.py.
        predictions = ensure_tissue_coords(
            _record(candidate, self._stimulus_set, self._region))

        raw = float(self._metric(predictions, self._brain).values)
        null = float(self._metric(
            shuffle_tissue_coords(predictions, self._null_seed), self._brain).values)

        signal = raw - null
        score = Score(signal)
        score.attrs['raw'] = raw          # alignment of the model's r(d) profile to the brain's
        score.attrs['null'] = null        # same, with unit positions shuffled (the floor)
        score.attrs['ceiling'] = self.ceiling
        return score


def make_topographic_benchmark(identifier, brain_assembly, stimulus_set, region, **kwargs):
    """Factory — build a :class:`TopographicBenchmark`. Register the returned instance in
    ``brainscore.benchmark_registry`` once you have a real surface fMRI target + stimuli."""
    return TopographicBenchmark(identifier, brain_assembly, stimulus_set, region, **kwargs)
