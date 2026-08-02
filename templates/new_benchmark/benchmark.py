"""TODO: one-line description — what measurements, which model capability, which metric.

A benchmark drives the candidate model and returns a ceiled Score. It calls the model only
through the unified interface (start_recording / start_task / process) — never legacy shims.

**This file runs as-is.** `_load_assembly()` builds a small synthetic target so the
benchmark constructs, registers, and scores a candidate offline. Copy the folder, run
`pytest`, watch it pass, then swap in your real data and metric a piece at a time.
"""
import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore import benchmark_registry, load_metric

BIBTEX = """TODO: @article{...} citation for the data."""

N_STIMULI, N_NEUROIDS = 12, 16


def _load_assembly():
    """TODO: load your real target measurements as a DataAssembly.

    Replace this whole function. For naturalistic/temporal data, build the
    stimulus_set + per-(subject, run) assembly and reuse
    `brainscore_core/temporal.py` (temporal_bin, hrf_convolve, contiguous_block_cv).

    The synthetic stand-in below exists so the template runs before you have data.
    Note what it establishes, because your real assembly needs the same things:
      - dims (presentation, neuroid)
      - a `stimulus_id` coord on presentation that matches the stimulus set
      - a `stimulus_set` attribute the benchmark can hand to the model
    """
    rng = np.random.RandomState(0)
    stimulus_ids = [f'stim{i:03d}' for i in range(N_STIMULI)]

    stimulus_set = StimulusSet([{'stimulus_id': sid, 'image_file_name': f'{sid}.png'}
                                for sid in stimulus_ids])
    # A real stimulus set maps each id to a file on disk. Ours points nowhere, which
    # is fine only because the dummy candidate in the tests never opens the images.
    stimulus_set.stimulus_paths = {sid: f'/nonexistent/{sid}.png' for sid in stimulus_ids}
    stimulus_set.identifier = 'your-benchmark-stimuli'

    # NOTE — two coords per dimension, not one. brainio promotes coords to a pandas
    # MultiIndex, and with only a single coord on an axis that promotion does not
    # happen, so `assembly.stimulus_id` is then missing and metrics fail with
    # "no stimulus_id on the presentation axis". Carry at least two coords per dim.
    assembly = NeuroidAssembly(
        rng.randn(N_STIMULI, N_NEUROIDS),
        coords={'stimulus_id': ('presentation', stimulus_ids),
                'presentation_index': ('presentation', list(range(N_STIMULI))),
                'neuroid_id': ('neuroid', list(range(N_NEUROIDS))),
                'region': ('neuroid', ['IT'] * N_NEUROIDS)},
        dims=['presentation', 'neuroid'])
    assembly.attrs['stimulus_set'] = stimulus_set
    return assembly


def _ceiling() -> Score:
    # TODO: estimate the data ceiling (e.g. split-half reliability). Score(1.0) means
    # "unceiled" — fine to start, but say so wherever you report the number, since an
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
        # TODO: your metric id. `direct-comparison` needs no fitting, which keeps this
        # template offline; a predictivity benchmark usually wants a regression metric.
        self._metric = load_metric('direct-comparison')

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
