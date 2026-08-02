"""TODO: one-line description — what measurements, which model capability, which metric.

A benchmark drives the candidate model and returns a ceiled Score. It calls the model only
through the unified interface (start_recording / start_task / process) — never legacy shims.

**This file runs as-is, against a real model.** `_load_assembly()` builds a small synthetic
target *with real image files on disk*, so `templates/new_model/` can be scored against it
out of the box:

    import new_model, new_benchmark          # register both
    bench = brainscore.load_benchmark('your-benchmark')
    score = bench(brainscore.load_model('your-model'))

Copy the folder, run `pytest`, watch it pass, then swap in your real data and metric a
piece at a time.
"""
import os
import tempfile

import numpy as np
from PIL import Image

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore import benchmark_registry
import brainscore_vision

BIBTEX = """TODO: @article{...} citation for the data."""

# 32 is not arbitrary: the cross-validated metrics below stratify their splits by
# `object_name`, and a test fold must hold at least one stimulus per category. With
# 4 categories, fewer than ~32 stimuli gives "test_size = 3 should be greater or equal
# to the number of classes = 4". Real datasets are far larger; this is the floor.
N_STIMULI, N_NEUROIDS = 32, 16
CATEGORIES = ['alpha', 'beta', 'gamma', 'delta']


def _load_assembly():
    """TODO: load your real target measurements as a DataAssembly.

    Replace this whole function. For naturalistic/temporal data, build the stimulus_set
    + per-(subject, run) assembly and reuse `brainscore_core/temporal.py`
    (temporal_bin, hrf_convolve, contiguous_block_cv).

    The synthetic stand-in below exists so the template runs before you have data. Note
    what it establishes, because your real assembly needs all of it:
      - dims (presentation, neuroid)
      - a `stimulus_id` coord on presentation, matching the stimulus set
      - an `object_name` coord — the cross-validated metrics stratify their splits on it
      - at least TWO coords per axis (a single coord is not promoted to a MultiIndex,
        and metrics then fail with "no stimulus_id on the presentation axis")
      - a `stimulus_set` whose `stimulus_paths` point at files that REALLY EXIST; a real
        candidate opens them, unlike the dummy in the tests
    """
    rng = np.random.RandomState(0)
    stimulus_ids = [f'stim{i:03d}' for i in range(N_STIMULI)]
    categories = [CATEGORIES[i % len(CATEGORIES)] for i in range(N_STIMULI)]

    directory = tempfile.mkdtemp(prefix='your_benchmark_stimuli_')
    paths = {}
    for i, sid in enumerate(stimulus_ids):
        path = os.path.join(directory, f'{sid}.png')
        image_rng = np.random.RandomState(i)
        Image.fromarray(
            image_rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(path)
        paths[sid] = path

    stimulus_set = StimulusSet([
        {'stimulus_id': sid, 'image_file_name': paths[sid], 'object_name': obj}
        for sid, obj in zip(stimulus_ids, categories)])
    stimulus_set.stimulus_paths = paths
    stimulus_set.identifier = 'your-benchmark-stimuli'

    assembly = NeuroidAssembly(
        rng.randn(N_STIMULI, N_NEUROIDS),
        coords={'stimulus_id': ('presentation', stimulus_ids),
                'object_name': ('presentation', categories),
                'neuroid_id': ('neuroid', list(range(N_NEUROIDS))),
                'region': ('neuroid', ['IT'] * N_NEUROIDS)},
        dims=['presentation', 'neuroid'])
    assembly.attrs['stimulus_set'] = stimulus_set
    return assembly


def _ceiling() -> Score:
    # TODO: estimate the data ceiling (e.g. split-half reliability). Score(1.0) means
    # "unceiled" — fine to start, but say so wherever you report the number, because an
    # unceiled score is not comparable to a ceiled one.
    return Score(1.0)


class YourBenchmark(BenchmarkBase):
    def __init__(self):
        super().__init__(identifier='your-benchmark',
                         ceiling=_ceiling(), version=1,
                         parent='neural',            # 'neural' | 'behavioral' | 'engineering'
                         bibtex=BIBTEX)
        self._assembly = _load_assembly()
        self._stimulus_set = self._assembly.attrs['stimulus_set']
        # TODO: your metric. A predictivity metric REGRESSES model units onto target
        # units, so the two need not have matching unit counts — that is what lets any
        # model be scored against fixed measurements. An element-wise metric such as
        # 'direct-comparison' requires identical shapes, so it only works when the model
        # happens to output exactly as many units as the data has.
        #
        # `linear_predictivity` is used here rather than the more common `pls` because
        # PLS defaults to 25 components and raises if a model has fewer units than that
        # ("`n_components` upper bound is 16. Got 25"). Production neural benchmarks
        # generally do use `pls` — switch once your models are wide enough.
        self._metric = brainscore_vision.load_metric('linear_predictivity')

    def __call__(self, candidate) -> Score:
        # 1) tell the model what to record / which task to run
        candidate.start_recording('IT', time_bins=[(70, 170)])   # TODO: your region/time-bins
        # 2) run the model through the ONE evaluation method
        predictions = candidate.process(self._stimulus_set)       # -> assembly
        # 3) score against the target, then normalize by the ceiling
        raw = self._metric(predictions, self._assembly)
        score = Score(float(raw) / float(self.ceiling))
        score.attrs['raw'] = raw
        score.attrs['ceiling'] = self.ceiling
        return score


benchmark_registry['your-benchmark'] = lambda: YourBenchmark()
