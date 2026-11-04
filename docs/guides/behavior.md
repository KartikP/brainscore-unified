# Behavior

Behavioral benchmarks compare answers instead of activity. A `TaskContext` says what the subject is asked to do; its `response` says what form the answer takes:

```python
from brainscore_core import TaskContext
from brainscore_core.responses import Choice, Scale, Text

task = TaskContext(
    instruction='Is this a real word?',
    response=Choice(['real', 'pseudo']),   # the form of the answer
    fitting_stimuli=train_stimuli,         # labeled examples, for a readout
)
```

| Response | Answer | Channel |
| --- | --- | --- |
| none: `TaskContext(task_type='passive')` | nothing; plain viewing | (neural only) |
| `Choice(labels)` | one label | `behavior:choice` |
| `Choice(labels, probabilities=True)` | also the implied probability of each label | `+ behavior:probabilities` |
| `Choice(labels, confidence=Scale(1, 5, 1))` | also a reported confidence, in the same reply | `+ behavior:confidence` |
| `Scale(1, 7, 1)` | a rating | `behavior:rating` |
| `Text()` | a written answer, kept as written | `behavior:text` |

Each response phrases its own directive after the instruction ("Answer with exactly one of: real, pseudo.") and parses the reply. Prompt wording and parsing are the benchmark's science: subclass a response to change them.

**The order of a trial.** A model that replies receives a `Prompt`: the instruction, the stimulus, then the response's directive. When a stimulus has several parts (a sentence and a clip), say the order people saw them in with `present`; a stimulus with several parts and no order fails before any compute:

```python
TaskContext(instruction='Which way does the ball move?', present=('text', 'video'),
            response=Choice(['left', 'right', 'up', 'down']))
```

Run the task through a session, or attach `RecordBehavior` in an [experiment](experiments.md):

```python
from brainscore_core.streaming import BufferedSession

session = BufferedSession(stimulus_set=test_stimuli, task_context=task,
                          requested_output_channels=['behavior:choice'])
model.participate(session)
choices = session.collect('behavior:choice')   # BehavioralAssembly: presentation x choice
model.reset()
```

## Reported and implied

A **reported** answer is the model's reply, parsed. An **implied** measure is read from the model's internals by a stated link, as Futrell2018 reads reading times from surprisal. Every result says which it is in `attrs['measure']`.

| Path | How | Gives | Model needs |
| --- | --- | --- | --- |
| **Generation** | the model replies to the instruction; the response parses it | choice, rating, confidence, text (reported) | `reply` (or `answer` for a plain choice) |
| **Likelihood** | the model scores each label as its reply, without generating | choice, probabilities (implied) | `likelihood` |
| **Readout** | a classifier or ridge regression on one layer, fitted on `fitting_stimuli` | choice, probabilities, rating (implied) | `behavioral_readout_layer` and an extractor |

`prefer_path` (`'generation'`, `'likelihood'`, `'readout'` or `'auto'`) picks one; `'auto'` tries them in that order. A readout asks what information a layer holds; generation asks what the model does when asked. Set the path explicitly whenever your conclusion depends on it.

One benchmark can take models of both kinds, as long as it fills in both the instruction and `fitting_stimuli`. Choices, ratings and implied probabilities come from either kind. Reported confidence and written answers need a model that replies; for a readout-only model those benchmarks are N/A before any compute.

A reply that cannot be parsed is recorded as invalid: NaN values, `answer_valid` is `False`, the reply is kept in `raw_answer` and the reason in `parse_error`. Nothing is credited. Written answers are stored as written; scoring them (a parser, a judge, a comparison with people) is the benchmark's job.

**Parts.** `reply(prompt)` returns the model's raw reply to one trial; `likelihood(prompt, candidates)` returns the log probability of each candidate answer. `chat_parts(model, processor)` in `brainscore.model_helpers.replies` builds both for a Hugging Face chat model; it shows the prompt's parts in order, images as images and clips as video:

```python
reply, likelihood = chat_parts(hf_model, processor, chat_template_kwargs={'enable_thinking': False})
BrainScoreModel(..., reply=reply, likelihood=likelihood)
```

**Older task types.** `task_type='label'`, `'classification'` and `'probabilities'` still work on the plain `behavior` channel with `answer`, `label_set` and a readout, so existing benchmarks and scores are unchanged.

## Hosted models

`build_api_generation_fn` (in `brainscore.model_helpers.api_behavioral`) builds an `answer` for a hosted model API. Such a model has no layers, so it can take part in behavioral benchmarks only.

See [Lesson 6](../../notebooks/lessons/06_behavior.ipynb) and [Lesson 5](../../notebooks/lessons/05_layers_and_regions.ipynb).
