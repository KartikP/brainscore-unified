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
| `Text(max_tokens=512, temperature=1.0)` | a written answer under the task's generation settings | `behavior:text` |

Each response phrases its own directive after the instruction ("Answer with exactly one of: real, pseudo.") and parses the reply. Prompt wording is the benchmark's science: the benchmark writes the instruction and the stimulus text. A task accepts only `Choice`, `Scale` and `Text` themselves, not a subclass, so a benchmark cannot change the directive by subclassing. To add no sentence at all, use `Choice(..., raw_text=True)`.

Three fields of `Choice` serve a library that compares options by their likelihood (EWoK):

| Field | Meaning |
| --- | --- |
| `labels_column='options'` | the options differ for each row; the column holds each row's options. The `choice` axis is then the position of each option in that cell. |
| `raw_scores=True` | `behavior:probabilities` carries each option's score exactly as the `likelihood` part gave it (summed log probability), not normalized; `attrs['measure']['scale']` is `'log'`. Only a `likelihood` part gives it; any other model is N/A. |
| `raw_text=True` | the stimulus text is shown exactly as given: no directive, and no instruction. A `likelihood` part scores `f"{text} {option}"` as plain text, with no chat template and no beginning-of-sequence token. A `reply` part gets no added sentence but still answers through its chat template. |

**The order of a trial.** A model that replies receives a `Prompt`: the instruction, the stimulus, then the response's directive. When a stimulus has several parts (a sentence and a clip), say the order people saw them in with `present`; a stimulus with several parts and no order fails before any compute:

```python
TaskContext(instruction='Which way does the ball move?', present=('text', 'video'),
            response=Choice(['left', 'right', 'up', 'down']))
```

Run the task through a session, or attach `RecordBehavior` in an [experiment](experiments.md):

```python
from brainscore_core.streaming_helpers import stimulus_session

session = stimulus_session(test_stimuli, ['behavior:choice'], task)
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

**Generation settings.** The settings of a written answer belong to the task, so the benchmark sets them on `Text`: `max_tokens`, `temperature` (0 or `None` is greedy), `top_p` (needs a temperature above 0) and `seed`. A `reply` part reads them from `prompt.response`. The reply that `chat_parts` builds obeys them. With no settings it is greedy and at most `max_new_tokens` (64) tokens long, as before.

**Older task types.** `task_type='label'`, `'classification'` and `'probabilities'` still work on the plain `behavior` channel with `answer`, `label_set` and a readout, so existing benchmarks and scores are unchanged.

## Hosted models

`build_api_generation_fn` (in `brainscore.model_helpers.api_behavioral`) builds an `answer` for a hosted model API. Such a model has no layers, so it can take part in behavioral benchmarks only.

`build_api_reply` builds a `reply` part from the same clients. It shows the trial's text and images in order, sends the task's instruction as the system text, and obeys the task's generation settings. A provider that refuses a setting the task gives fails before any request: the Anthropic API has no seed and refuses `top_p` and any temperature other than its default 1.0 (which is then not sent). The registered closed models come in two forms: `<name>` (an `answer` only, for choice tasks) and `<name>-reply` (a `reply`, which also writes answers, for example for CogGym).

See [Lesson 6](../../notebooks/lessons/06_behavior.ipynb) and [Lesson 5](../../notebooks/lessons/05_layers_and_regions.ipynb).
