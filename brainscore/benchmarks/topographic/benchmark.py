"""Topographic-organization benchmark.

Scores whether a model's spatial unit layout matches cortical topography — the
correlation-vs-distance profile r(d), beyond predictivity — with a matched
shuffle-coordinate NULL. The score reported is ``raw - null`` (the spatial signal above
the permutation floor), with both stored in ``attrs``. Predictivity is permutation-invariant
over units and so is blind to this; the shuffle null makes that explicit.

A *real* registered instance needs (a) a topographic model whose ``process()`` output carries
per-unit ``tissue_x``/``tissue_y`` coords (Topo-Omni's sheet, a TDANN's tissue map, or a conv
feature-map grid) and (b) a brain assembly with voxel coordinates (e.g. an fsaverage-surface
fMRI assembly). This module ships the reusable benchmark + factory; register a concrete
instance once such a surface assembly is staged (EC2). The wiring is validated offline on
synthetic assemblies in ``tests/test_topographic_benchmark.py``.
"""
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore import load_metric
from brainscore.topographic_support import ensure_tissue_coords, shuffle_tissue_coords


class TopographicBenchmark(BenchmarkBase):
    """Reusable topographic-alignment benchmark.

    :param brain_assembly: target measurements with voxel coordinates
        (``tissue_x``/``tissue_y`` or ``x``/``y``/``z`` neuroid coords).
    :param stimulus_set: stimuli to drive the candidate (same stimuli the brain saw).
    :param region: recording target passed to ``candidate.start_recording``.
    :param metric_id: registered metric (default the topographic-alignment metric).
    :param null_seed: seed for the shuffle-coordinate null.
    """

    def __init__(self, identifier, brain_assembly, stimulus_set, region, *,
                 metric_id='topographic-alignment', null_seed=0,
                 parent='neural', bibtex=None, ceiling=None):
        super().__init__(identifier=identifier, ceiling=ceiling or Score(1.0),
                         version=1, parent=parent, bibtex=bibtex)
        self._brain = brain_assembly
        self._stimulus_set = stimulus_set
        self._region = region
        self._metric = load_metric(metric_id)
        self._null_seed = null_seed

    def __call__(self, candidate) -> Score:
        candidate.start_recording(self._region)
        predictions = ensure_tissue_coords(candidate.process(self._stimulus_set))

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
