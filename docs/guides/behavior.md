# Behavior

Behavioral benchmarks compare choices or answers instead of activity. The task is described by a `TaskContext`:

```python
from brainscore_core import TaskContext

task = TaskContext(
    task_type='probabilities',            # 'probabilities', 'label' or 'classification'
    label_set=['real', 'pseudo'],         # answers the benchmark accepts
    fitting_stimuli=train_stimuli,        # labeled examples, for a readout
    instruction='Is this a real word?',   # for models that follow instructions
    prefer_path='readout',                # 'readout', 'generation' or 'auto'
)
```

Run it through a session on the `behavior` channel:

```python
from brainscore_core.streaming import BufferedSession

session = BufferedSession(task_context=task, requested_output_channels=['behavior'])
model.participate(session)
choices = session.collect()      # BehavioralAssembly: presentation x choice
model.reset()
```

## Two ways to answer

| Path | How | Model needs |
| --- | --- | --- |
| **Readout** | Fit a logistic classifier on one layer's activity using `fitting_stimuli`, then predict. | `behavioral_readout_layer` and an extractor |
| **Generation** | Call `answer(stimulus_row, instruction, label_set)` for each stimulus. | `answer` |

These are different measurements: a readout asks what information a layer contains; generation asks what the model does when asked. A model can support both. With `prefer_path='auto'` generation is used when available; set the path explicitly whenever your conclusion depends on it.

If `answer` returns a label outside `label_set`, the response is recorded as invalid: its probabilities are NaN, `answer_valid` is `False`, and the reply is kept in `raw_answer`. A warning is issued. No label is credited.

## Hosted models

`build_api_generation_fn` (in `brainscore.model_helpers.api_behavioral`) builds an `answer` for a hosted model API. Such a model has no layers, so it can take part in behavioral benchmarks only.

See [Lesson 6](../../notebooks/lessons/06_behavior.ipynb) and [Experiment 1](../../notebooks/experiments/01_does_the_model_beat_random_features.ipynb).
