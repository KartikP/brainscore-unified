# Write a benchmark

A benchmark presents stimuli, requests outputs, and compares them with measured data. It is a `Benchmark` subclass with declarations and a `__call__(subject)` that returns a `Score`.

Already have a `TaskProtocol`? Follow the [experiment-to-benchmark walkthrough](experiments.md#7-make-it-a-benchmark): keep the protocol and recorders, run them inside `Benchmark.__call__`, and compare the saved assembly with reference data. `Experiment` continues to own session creation and resets. The manual-session example below is for authors who need that lower-level control.

```python
from brainscore_core import Benchmark, Score
from brainscore_core.streaming import BufferedSession

class MyITBenchmark(Benchmark):
    identifier = 'MyLab2026.IT-pls'
    uses_session = True
    required_input_channels = {'vision'}          # what the benchmark provides
    requested_output_channels = {'neural:IT'}     # what it needs back

    def __init__(self, stimuli, measured, metric, ceiling):
        self._stimuli, self._measured = stimuli, measured
        self._metric, self._ceiling = metric, ceiling

    def __call__(self, subject):
        session = BufferedSession(stimulus_set=self._stimuli,
                                  requested_output_channels=['neural:IT'],
                                  time_bins=[(70, 170)])
        subject.participate(session)
        raw = self._metric(session.collect(), self._measured)
        score = Score(float(raw) / self._ceiling)
        score.attrs['raw'] = raw
        score.attrs['ceiling'] = self._ceiling
        return score
```

## The parts

| Part | What to provide |
| --- | --- |
| Stimuli | A `StimulusSet` (one row per stimulus, a `stimulus_id` column, and a column per input type such as `image_path` or `sentence`), or your own list of `StreamEvent`s. |
| Declarations | `required_input_channels`, `requested_output_channels`, and any `required_traits` such as `memory:session`. Compatibility compares these with the subject's declarations. |
| Session | Usually `BufferedSession`. Add `task_context=TaskContext(...)` for behavior, or use `EnvironmentSession` for closed-loop tasks. |
| Metric | Any function of model output and measured data. Brain-Score ships metrics such as PLS regression, ridge regression, RSA and accuracy (`brainscore.load_metric`). |
| Ceiling | An estimate of how well the data can be predicted, for example split-half reliability. Leave it out and report the raw value if you have none. |
| Reference data | Measured responses in an assembly whose `presentation` dimension carries `stimulus_id` and at least one more coordinate (a condition, a category, a subject). A dimension with only one coordinate keeps its values as the dimension's index and loses the coordinate's name, so `reference['stimulus_id']` would fail. |

## Checklist

- Score `chance-baseline` and `random-vit-b-32` first. A model result is only meaningful above these floors.
- Fit any readout or regression on different stimuli from the ones you test on.
- Record in `score.attrs` whether the value is raw or ceiled, and how the ceiling was computed.
- Behavioral benchmarks that depend on how an answer is produced should set `TaskContext(prefer_path='readout')` or `'generation'` explicitly ([Behavior](behavior.md)).

[Lesson 2](../../notebooks/lessons/02_your_first_benchmark.ipynb) builds a complete toy benchmark. Registered examples live in `brainscore/benchmarks/`.
