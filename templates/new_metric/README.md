# Template: a new metric

A metric compares two assemblies and returns a `Score`. See `EXTENDING.md` (Seam 3) for the
full contract.

## Use it

1. Copy this folder to `brainscore/metrics/<your_name>/`.
2. Edit `metric.py`. Implement `__call__(assembly1, assembly2) -> Score`. Document the coords
   each assembly must carry, and fail loudly on a mismatch.
3. Edit `__init__.py`. Pick your identifier in `metric_registry['<your-id>']`.
4. Nothing else to edit: the folder registers by its presence.
5. Move `test_metric.py` to `tests/test_metrics/test_<your_name>.py`, adapt it and run it (`pytest tests/test_metrics/test_<your_name>.py`).

From the `templates/` folder, with the data template for an assembly:

```python
import brainscore
import new_data, new_metric          # registers 'your-measurements' and 'your-metric'

measured = brainscore.load_dataset('your-measurements')
metric = brainscore.load_metric('your-metric')
print(round(float(metric(measured, measured)), 6))   # identical assemblies: 1.0
```

In a benchmark, the first argument is the model's response and the second the measured data.

## A metric of choices

A behavior metric reads the model's answer as the session stores it: one row per trial,
a `choice` axis with the options, and `response_state` (`valid`, `invalid` or `missing`)
for each row. It states what it does with an invalid and a missing trial. The people's
choices are a long table, one row per participant and trial. Here an invalid or missing
answer counts as wrong:

```python
from collections import Counter

from brainscore import BrainScoreModel
from brainscore_core.contract import TaskContext
from brainscore_core.metrics import Score
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


# Metric: agreement with the people's majority choice
def majority_agreement(answer, people):
    """Share of trials where the model chose the people's most common choice."""
    majority = {stimulus: Counter(people.values[people['stimulus_id'].values == stimulus]).most_common(1)[0][0]
                for stimulus in set(people['stimulus_id'].values)}
    options = list(answer['choice'].values)
    agree = [state == 'valid' and options[int(row.argmax())] == majority[stimulus]
             for stimulus, state, row in zip(answer['stimulus_id'].values,
                                             answer['response_state'].values, answer.values)]
    score = Score(sum(agree) / len(agree))
    score.attrs['answer_policy'] = 'invalid: wrong, counted; missing: wrong, counted'
    return score


# Three words; two participants' choices for each
stimuli = StimulusSet({'stimulus_id': ['s1', 's2', 's3'], 'sentence': ['cat', 'dog', 'cat or dog']})
stimuli.identifier = 'metric-example-choices'
people = BehavioralAssembly(['cat', 'cat', 'dog', 'dog', 'cat', 'dog'], dims=['presentation'], coords={
    'stimulus_id': ('presentation', ['s1', 's1', 's2', 's2', 's3', 's3']),
    'participant': ('presentation', ['p1', 'p2', 'p1', 'p2', 'p1', 'p2'])})


def answer(*, stimulus_row, instruction, label_set):      # a stand-in model
    return stimulus_row['sentence'] if stimulus_row['sentence'] in label_set else 'unsure'


# Run the stand-in on the task, then score
model = BrainScoreModel(identifier='stand-in', in_channels={'text'}, answer=answer)
session = stimulus_session(stimuli, ['behavior:choice'],
                           TaskContext(instruction='Cat or dog?', response=Choice(['cat', 'dog'])))
model.participate(session)
print(float(majority_agreement(session.collect('behavior:choice'), people)))   # 0.6666666666666666
```

The third answer is invalid ('unsure' is not an option), so it counts as wrong. In a
benchmark, pass the answer to `benchmark_score(..., answers=answer, answer_policy=...)`, which
also writes `n_valid`, `n_invalid` and `n_missing`; see "Trial states on a score" in
[Behavior](../../docs/guides/behavior.md) and the worked benchmark in
[External libraries](../../docs/guides/external_libraries.md).

## Worked reference

`brainscore/metrics/topographic/` (`load_metric('topographic-alignment')`) is a complete,
non-trivial example: it scores a model's spatial unit layout against cortical topography via the
correlation-vs-distance profile. Read it alongside this template.

## Keep axes separate

Don't overload one metric to mean several things. Predictivity, RSA, topographic alignment, and
SCA-style representational alignment are *distinct* metrics that answer distinct questions; register
them separately.
