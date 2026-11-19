"""TODO: one-line description — what measurements, which model capability, which metric.

A benchmark drives the subject and returns a Score: raw divided by a measured ceiling,
or raw when the ceiling is not estimated. It runs the subject
through typed helpers such as `neural_response`, which hand it a session via
`subject.participate(session)`. Any Subject can be scored, including one that implements
only `participate` (no `start_recording` / `process`).

**This file runs as-is, against a real model.** `standin_data.py` builds a small synthetic
target *with real image files on disk*, so `templates/new_model/` can be scored against it
out of the box:

    import new_model, new_benchmark          # register both
    bench = brainscore.load_benchmark('your-benchmark')
    score = bench(brainscore.load_model('your-model'))

Copy the folder, run `pytest`, watch it pass, then swap in your real data and metric a
piece at a time.
"""
from brainscore_core.assembly_builder import replace_presentation_coord
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.streaming_helpers import neural_response
from brainscore import load_metric
from brainscore.benchmarks import benchmark_score, not_estimated

from .standin_data import BIBTEX, load_assembly   # TODO: your data package


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
        # TODO: estimate the data ceiling (e.g. split-half reliability) and pass it as
        # Score(estimate). Until then the ceiling is not estimated and the score is the raw
        # value (normalized=False), which is not comparable to a ceiled score.
        super().__init__(identifier='your-benchmark',
                         ceiling=not_estimated(), version=1,
                         parent='neural',            # 'neural' | 'behavioral' | 'engineering'
                         bibtex=BIBTEX)
        self._assembly = load_assembly()
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

