# Write a benchmark

A benchmark presents stimuli, requests outputs, and compares them with measured data. It is a `Benchmark` subclass with declarations and a `__call__(subject)` that returns a `Score`.

Already have a `TaskProtocol`? Follow the [experiment-to-benchmark walkthrough](../../notebooks/lessons/10_run_an_experiment.ipynb) (its last section, "Make it a benchmark"): keep the protocol and recorders, run them inside `Benchmark.__call__`, and compare the saved assembly with reference data. `Experiment` continues to own session creation and resets. The manual-session example below is for authors who need that lower-level control.

```python
from brainscore_core import Benchmark
from brainscore_core.streaming_helpers import stimulus_session
from brainscore.benchmarks import benchmark_score

class MyITBenchmark(Benchmark):
    identifier = 'MyLab2026.IT-pls'
    uses_session = True
    required_input_channels = {'vision:image'}    # what the benchmark provides
    requested_output_channels = {'neural:IT'}     # what it needs back

    version = 1

    def __init__(self, stimuli, measured, metric, ceiling):
        self._stimuli, self._measured = stimuli, measured
        self._metric, self.ceiling = metric, ceiling    # ceiling: a Score

    def __call__(self, subject):
        session = stimulus_session(self._stimuli, ['neural:IT'], time_bins=[(70, 170)])
        subject.participate(session)
        raw = self._metric(session.collect(), self._measured)
        return benchmark_score(self, float(raw) / float(self.ceiling), raw=raw,
                               metric='pls', normalized=True)
```

`benchmark_score` writes the attributes every unified score carries: `raw` (the metric value before normalization), `ceiling` (a `Score`), `normalized` (whether the score is `raw` divided by a measured ceiling), `metric` and `benchmark_version`. Keyword arguments after these are your own attributes. With no ceiling, pass `ceiling=not_estimated()` (from `brainscore.benchmarks`) to the benchmark and `normalized=False`: the score is then the raw value. `score()` adds `benchmark_identifier`, `model_identifier`, `compatibility` and `trial_memory`.

## The parts

| Part | What to provide |
| --- | --- |
| Stimuli | A `StimulusSet` (one row per stimulus, a `stimulus_id` column, and a column per input type such as `image_path` or `sentence`), or your own list of `StreamEvent`s. |
| Declarations | `required_input_channels`, `requested_output_channels`, and any `required_traits` such as `memory:trial_history`. Channels match by exact name. Compatibility compares these with the subject's declarations. |
| Session | Usually `stimulus_session(stimuli, outputs)`. Add a `TaskContext(...)` as the third argument for behavior; `TaskContext(trial_memory='history')` keeps the subject's history across trials. With the default `trial_memory`, a second answer to one trial is an error; an answer belongs to the trial named by the row identity in its metadata. For closed-loop tasks, use `run_environment` ([Closed loop](closed_loop.md)). |
| Metric | Any function of model output and measured data. Brain-Score ships metrics such as PLS regression, ridge regression, RSA and accuracy (`brainscore.load_metric`). |
| Ceiling | An estimate of how well the data can be predicted, for example split-half reliability. With none, use `not_estimated()` and report the raw value (`normalized=False`). |
| Reference data | Measured responses in an assembly whose `presentation` dimension carries `stimulus_id` and at least one more coordinate (a condition, a category, a subject). A dimension with only one coordinate keeps its values as the dimension's index and loses the coordinate's name, so `reference['stimulus_id']` would fail. |

## Checklist

- Score `chance-baseline` and `random-vit-b-32` first. A model result is only meaningful above these floors.
- Fit any readout or regression on different stimuli from the ones you test on.
- Build the score with `benchmark_score`, so it says whether the value is raw or ceiled (`normalized`) and carries its `ceiling`; record how the ceiling was computed.
- How a model answers (its answer function or its readout) is a property of the model. A behavioral benchmark that accepts both fills in the instruction and `fitting_stimuli`; to compare the paths, score one model built for each path ([Behavior](behavior.md)).

[Lesson 2](../../notebooks/lessons/02_your_first_benchmark.ipynb) builds a complete toy benchmark. Registered examples live in `brainscore/benchmarks/`.
