"""TODO: one-line description — what measurements, which model capability, which metric.

A benchmark drives the subject and returns a Score: raw divided by a measured ceiling,
or raw when the ceiling is not estimated. It runs the subject
through typed helpers such as `neural_response`, which hand it a session via
`subject.participate(session)`. Any Subject can be scored, including one that implements
only `participate` (no `start_recording` / `process`).

**This file runs as-is, against a real model.** `_load_assembly()` builds a small synthetic
target *with real image files on disk*, so `templates/new_model/` can be scored against it
out of the box:

    import new_model, new_benchmark          # register both
    bench = brainscore.load_benchmark('your-benchmark')
    score = bench(brainscore.load_model('your-model'))

Copy the folder, run `pytest`, watch it pass, then swap in your real data and metric a
piece at a time.
"""
import io
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from brainscore_core.assembly_builder import replace_presentation_coord
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.streaming_helpers import neural_response
from brainscore import benchmark_registry, load_metric
from brainscore.benchmarks import benchmark_score, not_estimated

BIBTEX = """TODO: @article{...} citation for the data."""

# 32 is not arbitrary. The metric below correlates prediction and data within each test
# fold, and by default a test fold is 10 % of the stimuli: 4 of 32. A correlation on two
# stimuli is always +1 or -1, so a score from folds that small means nothing, and the
# metric does not warn. The folds are also stratified by `object_name`, so a test fold
# needs one stimulus per category: with 4 categories, fewer than ~32 stimuli gives
# "test_size = 3 should be greater or equal to the number of classes = 4". With fewer
# stimuli, set a larger test fold (see the metric below).
N_STIMULI, N_NEUROIDS = 32, 16
CATEGORIES = ['alpha', 'beta', 'gamma', 'delta']


def _write_once(path, data: bytes) -> None:
    """Write ``data`` to ``path`` unless the same bytes are already there.

    The bytes go to an exclusive staging file with a unique name in the same folder, which
    then replaces the final name in one step, so a reader never sees half a file and two
    writers do not collide. A link at the final name is refused: the write would land in
    another file.
    """
    if os.path.islink(path):
        raise ValueError(f'{path} is a link; remove it and load again')
    if os.path.isfile(path):
        with open(path, 'rb') as existing:
            if existing.read() == data:
                return
    descriptor, staging = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.partial')
    try:
        with os.fdopen(descriptor, 'wb') as file:
            file.write(data)
        os.replace(staging, path)
    except BaseException:
        os.unlink(staging)
        raise


def _load_assembly():
    """TODO: load your real target measurements as a DataAssembly.

    Replace this whole function. For naturalistic/temporal data, build the stimulus_set
    + per-(subject, run) assembly and reuse `brainscore/temporal.py`
    (temporal_bin, hrf_convolve, contiguous_block_cv).

    The synthetic stand-in below exists so the template runs before you have data. Note
    what it establishes, because your real assembly needs all of it:
      - dims (presentation, neuroid)
      - a `stimulus_id` coord on presentation, matching the stimulus set
      - an `object_name` coord — the cross-validated metrics stratify their splits on it
      - at least TWO coords per axis (a single coord is not promoted to a MultiIndex,
        and metrics then fail with "no stimulus_id on the presentation axis")
      - a `stimulus_set` whose `stimulus_paths` point at files that REALLY EXIST; a real
        subject opens them, unlike the dummy in the tests
    """
    rng = np.random.RandomState(0)
    stimulus_ids = [f'stim{i:03d}' for i in range(N_STIMULI)]
    categories = [CATEGORIES[i % len(CATEGORIES)] for i in range(N_STIMULI)]

    # The images are the same on every load, so they are written once, to one folder in the
    # Brain-Score home, and reused.
    directory = Path(os.getenv('BRAINSCORE_HOME', '~/.brain-score')).expanduser() / 'your_benchmark_stimuli'
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for i, sid in enumerate(stimulus_ids):
        path = directory / f'{sid}.png'
        image = io.BytesIO()
        Image.fromarray(
            np.random.RandomState(i).randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(image, format='PNG')
        _write_once(str(path), image.getvalue())
        paths[sid] = str(path)

    stimulus_set = StimulusSet([
        {'stimulus_id': sid, 'image_path': paths[sid], 'object_name': obj}
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
    # TODO: estimate the data ceiling (e.g. split-half reliability) and return it as
    # Score(estimate). Until then the ceiling is not estimated and the score is the raw
    # value (normalized=False), which is not comparable to a ceiled score.
    return not_estimated()


class YourBenchmark(BenchmarkBase):
    # What the benchmark shows and what it asks for, checked before any compute.
    # Inputs: pre-flight reads them from `self.stimulus_set` (set in __init__): each
    # stimulus column feeds one channel ('image_path' feeds 'vision:image', 'sentence'
    # feeds 'text'). A benchmark with a task keeps it as `self.task`; its instruction is
    # one more input.
    # Outputs: the exact channel you request, e.g. 'neural:<region>' or 'behavior'.
    requested_output_channels = frozenset({'neural:IT'})   # TODO: your region
    # 'none': every trial starts empty. 'history' needs memory:trial_history.
    trial_memory = 'none'

    def __init__(self):
        super().__init__(identifier='your-benchmark',
                         ceiling=_ceiling(), version=1,
                         parent='neural',            # 'neural' | 'behavioral' | 'engineering'
                         bibtex=BIBTEX)
        self._assembly = _load_assembly()
        self.stimulus_set = self._assembly.attrs['stimulus_set']
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
        #
        # The split: 10 folds, 90 % of the stimuli to fit and 10 % to test. Keep at least
        # 3 stimuli in each test fold; more give a steadier score. With few stimuli,
        # change the split, for example
        #   load_metric('linear_predictivity',
        #               crossvalidation_kwargs=dict(splits=10, train_size=0.5))
        # (12 stimuli: 6 to fit, 6 to test). docs/guides/write_a_benchmark.md lists the
        # metrics and their arguments.
        self._metric = load_metric('linear_predictivity')

    def __call__(self, subject) -> Score:
        # 1) show the stimuli and record the subject's IT response, 70-170 ms after onset
        predictions = neural_response(
            subject=subject,
            stimuli=self.stimulus_set,
            outputs=['neural:IT'],    # TODO: your region
            time_bins=[(70, 170)],    # TODO: your time window, in ms
        )
        # 2) label each response with its stimulus category, matched by stimulus_id;
        #    the metric's cross-validation keeps categories balanced across folds
        categories = dict(zip(self.stimulus_set['stimulus_id'], self.stimulus_set['object_name']))
        predictions = replace_presentation_coord(
            predictions,
            coord_name='object_name',
            values=[categories[stimulus_id] for stimulus_id in predictions['stimulus_id'].values],
        )
        # 3) score against the target
        raw = self._metric(predictions, self._assembly)
        # benchmark_score divides by a measured ceiling (the score is raw otherwise) and
        # writes raw, ceiling, normalized, metric and benchmark_version.
        return benchmark_score(self, raw=raw, metric='linear_predictivity')


benchmark_registry['your-benchmark'] = lambda: YourBenchmark()
