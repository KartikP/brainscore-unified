"""TODO: one-line description — what measurements, which model capability, which metric.

A benchmark drives the candidate model and returns a ceiled Score. It calls the model only
through the unified interface (start_recording / start_task / process) — never legacy shims.
"""
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore import benchmark_registry, load_metric

BIBTEX = """TODO: @article{...} citation for the data."""


def _load_assembly():
    # TODO: load your target measurements as a DataAssembly (neural or behavioral).
    # For naturalistic/temporal data, build the stimulus_set + per-(subject,run) assembly
    # and reuse core/brainscore_core/temporal.py (temporal_bin, hrf_convolve, contiguous_block_cv).
    ...


def _ceiling() -> Score:
    # TODO: estimate the data ceiling (e.g. split-half reliability). Score(value) is fine to start.
    return Score(1.0)


class YourBenchmark(BenchmarkBase):
    def __init__(self):
        super().__init__(identifier='your-benchmark',
                         ceiling=_ceiling(), version=1,
                         parent='neural',            # 'neural' | 'behavioral' | 'engineering'
                         bibtex=BIBTEX)
        self._assembly = _load_assembly()
        self._stimulus_set = self._assembly.stimulus_set  # TODO: or build separately
        self._metric = load_metric('topographic-alignment')  # TODO: your metric id

    def __call__(self, candidate) -> Score:
        # 1) tell the model what to record / which task to run
        candidate.start_recording('IT', time_bins=[(70, 170)])   # TODO: your region/time-bins
        # 2) run the model through the ONE evaluation method
        predictions = candidate.process(self._stimulus_set)       # -> assembly
        # 3) score against the target, then normalize by the ceiling
        raw = self._metric(predictions, self._assembly)
        score = Score(raw.values / self.ceiling.values)
        score.attrs['raw'] = raw
        score.attrs['ceiling'] = self.ceiling
        return score


benchmark_registry['your-benchmark'] = lambda: YourBenchmark()
