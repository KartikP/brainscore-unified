# Write a benchmark

A benchmark presents stimuli, requests outputs, and compares them with measured data. It is a `Benchmark` subclass with declarations and a `__call__(subject)` that returns a `Score`.

Already have a `TaskProtocol`? Follow the [experiment-to-benchmark walkthrough](../../notebooks/lessons/10_run_an_experiment.ipynb) (its last section, "Make it a benchmark"): keep the protocol and recorders, run them inside `Benchmark.__call__`, and compare the saved assembly with reference data. `Experiment` continues to own session creation and resets. The manual-session example below is for authors who need that lower-level control.

```python
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.streaming_helpers import stimulus_session
from brainscore.benchmarks import benchmark_score


class MyITBenchmark(BenchmarkBase):
    uses_session = True
    required_input_channels = {'vision:image'}    # what the benchmark provides
    requested_output_channels = {'neural:IT'}     # what it needs back

    def __init__(self, stimuli, measured, metric, ceiling):
        super().__init__(identifier='MyLab2026.IT-pls', version=1, ceiling=ceiling,  # ceiling: a Score
                         parent='neural', bibtex='')
        self._stimuli, self._measured, self._metric = stimuli, measured, metric

    def __call__(self, subject):
        session = stimulus_session(self._stimuli, ['neural:IT'], time_bins=[(70, 170)])
        subject.participate(session)
        raw = self._metric(session.collect('neural:IT'), self._measured)
        return benchmark_score(self, float(raw) / float(self.ceiling), raw=raw,
                               metric='pls', normalized=True)
```

`BenchmarkBase` holds the identifier, version and ceiling. To try the class, give it six
stimuli, made-up measurements, a correlation as the metric and a subject that answers
each image with its position in the table. In your benchmark, these are your stimuli,
your recorded data, a metric such as `brainscore.load_metric('pls')` and the model to test:

```python
import numpy as np
import pandas as pd
from brainscore_core import Score, Subject
from brainscore_core.streaming import StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

ids = [f'image{i}' for i in range(6)]
stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ids, 'image_path': [f'{i}.png' for i in ids]}))
measured = NeuroidAssembly(np.arange(6.0)[:, None] * [1, 2], dims=['presentation', 'neuroid'],
                           coords={'stimulus_id': ('presentation', ids),
                                   'condition': ('presentation', ['a'] * 6),
                                   'neuroid_id': ('neuroid', [0, 1])})


def correlation(predicted, measured):
    return Score(np.corrcoef(predicted.values.mean(1), measured.values.mean(1))[0, 1])


class Counter(Subject):
    """Answers each image with its position in the table."""
    identifier = 'counter'
    in_channels = {'vision:image'}
    out_channels = {'neural:IT'}

    def participate(self, session):
        while (event := session.next_input()) is not None:
            position = event.meta['stimulus_index']
            response = NeuroidAssembly([[float(position)]], dims=['presentation', 'neuroid'],
                                       coords={'stimulus_id': ('presentation', [event.meta['stimulus_id']]),
                                               'stimulus_index': ('presentation', [position]),
                                               'neuroid_id': ('neuroid', ['IT.0'])})
            session.respond(StreamEvent('neural:IT', response, t_ms=event.t_ms, meta=dict(event.meta)))


import brainscore

benchmark = MyITBenchmark(stimuli, measured, correlation, ceiling=Score(0.8))
score = brainscore.score(Counter(), benchmark)
print(round(float(score), 3), round(float(score.attrs['raw']), 3), score.attrs['normalized'])
# 1.25 1.0 True
```

The counter matches the made-up data exactly (raw 1.0), and dividing by a ceiling of 0.8
gives 1.25: a ceiled score can exceed 1.

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

- Score the null model that fits your benchmark first. A model result is only meaningful above that floor, and a null on which the benchmark is N/A tells you nothing.
  - `random-vit-b-32` (CLIP ViT-B/32 with random weights; it downloads the CLIP configuration and processor, not the trained weights) takes images and text and gives the regions of CLIP's map and a readout. Use it for neural benchmarks with images or text and for behavioral tasks answered by a readout.
  - `chance-baseline` gives only the plain `behavior` channel (uniform probabilities over the labels). Use it for a task on that channel, such as `Yeatman2021-lexical_decision-image`. It is N/A on neural benchmarks, on tasks that request `behavior:choice` or `behavior:text`, and on closed-loop benchmarks.
  - For a closed-loop benchmark, score a random policy: `BrainScoreModel(identifier='random-moves', policy=random_action_policy(seed=0))`, with `random_action_policy` from `brainscore.harnesses.grid_game` ([Score a model](score_a_model.md)).
- Fit any readout or regression on different stimuli from the ones you test on.
- Build the score with `benchmark_score`, so it says whether the value is raw or ceiled (`normalized`) and carries its `ceiling`; record how the ceiling was computed.
- How a model answers (its answer function or its readout) is a property of the model. A behavioral benchmark that accepts both fills in the instruction and `fitting_stimuli`; to compare the paths, score one model built for each path ([Behavior](behavior.md)).

[Lesson 2](../../notebooks/lessons/02_your_first_benchmark.ipynb) builds a complete toy benchmark. Registered examples live in `brainscore/benchmarks/`.
