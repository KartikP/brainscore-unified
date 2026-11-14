# Behavior

Behavioral benchmarks compare answers instead of activity. A `TaskContext` says what the subject is asked to do; its `response` says what form the answer takes.

The code on this page runs as written with these stand-ins: a few words as stimuli, and a model whose `reply` part always answers "real". Replace them with your stimuli and your model:

```python
import pandas as pd
from brainscore import BrainScoreModel
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def word_table(words, labels):
    return StimulusSet(pd.DataFrame({'stimulus_id': words, 'sentence': words, 'label': labels}))


train_stimuli = word_table(['house', 'blick', 'table', 'florp'], ['real', 'pseudo', 'real', 'pseudo'])
test_stimuli = word_table(['river', 'snarp'], ['real', 'pseudo'])
model = BrainScoreModel(identifier='always-real', in_channels={'text'},
                        reply=lambda prompt: 'real')   # a reply part: always answers 'real'
```

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
| `labels='options'` | a column name in place of the options: the options differ for each row, and the column holds each row's options. The `choice` axis is then the position of each option in that cell. |
| `raw_scores=True` | `behavior:probabilities` carries each option's score exactly as the `likelihood` part gave it (summed log probability), not normalized; the response record in `attrs['response']` has `category` `'model-derived'` and `parameters['scale']` `'log'`. Only a `likelihood` part gives it; any other model refuses the session (`ExecutionNotFeasibleError`). |
| `raw_text=True` | the stimulus text is shown exactly as given: no directive, and no instruction. A `likelihood` part scores `f"{text} {option}"` as plain text, with no chat template and no beginning-of-sequence token. A `reply` part gets no added sentence but still answers through its chat template. |

**The order of a trial.** A model that replies receives a `Prompt`: the instruction, the stimulus, then the response's directive. When a stimulus has several parts (a sentence and a clip), say the order people saw them in with `present`; a stimulus with several parts and no order fails before any compute. A model that does not take in a part (a text-only model on a task with a clip) gets the task without that entry, so its trial shows only the parts it takes in. An entry that names a column no stimulus has is an error for every model:

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
print(choices.values.tolist())   # [[1.0, 0.0], [1.0, 0.0]]: 'real' for both words
```

## How an answer was given

Every answer carries a response record in `attrs['response']`, and every score copies the records of its answers. The record says the format (the channel), the category (`behavioral` or `model-derived`), the origin (`native`, or `derived` by a conversion or a link), the method (`generation`, `likelihood`, `readout`, `link`), the conversion (`argmax`, `softmax`, `softmax_argmax` or none), the source format of a derived answer, the conversion's parameters (for example the tie rule `first`) and who parsed generated text (`parsed_by`). The conversions are core functions in `brainscore_core.response_formats`. A choice from a readout is the `argmax` of its probabilities; a choice from a likelihood part is `softmax_argmax` of the options' log probabilities.

| Path | How | Gives | Model needs |
| --- | --- | --- | --- |
| **Generation** | the model replies to the instruction; the response parses it | choice, rating, confidence, text | `reply` (or `answer` for a plain choice) |
| **Likelihood** | the model scores each label as its reply, without generating | choice (`softmax_argmax`), probabilities (`softmax`) | `likelihood` |
| **Readout** | a classifier or ridge regression on one layer, fitted on `fitting_stimuli` | choice (`argmax`), probabilities, rating | `behavioral_readout_layer` and an extractor |

Which path answers is a property of the model, not of the task. When only one of its paths can give an answer, that path gives it. When several can give the same answer (for example a reply and a likelihood both give a choice), the registration states which one owns it, per answer channel:

```python
def answer(stimulus_row, instruction, label_set):   # answers a legacy task
    return 'real'


def likelihood(prompt, candidates):                 # the log probability of each option
    return {candidate: 0.0 if candidate == 'real' else -2.0 for candidate in candidates}


owned = BrainScoreModel(
    identifier='owned-parts', in_channels={'text'},
    answer=answer, reply=lambda prompt: 'real', likelihood=likelihood,
    capability_config={'behavior-sessions': {
        'behavior': 'generation',                # bare behavior: the answer to a legacy task
        'behavior:choice': 'generation',         # the choice of a Choice response: another channel
        'behavior:probabilities': 'likelihood'}})

legacy = stimulus_session(test_stimuli, ['behavior'], TaskContext(
    task_type='probabilities', instruction='Is this a real word?', label_set=['real', 'pseudo']))
owned.participate(legacy)
choice = stimulus_session(test_stimuli, ['behavior:choice', 'behavior:probabilities'], TaskContext(
    instruction='Is this a real word?', response=Choice(['real', 'pseudo'], probabilities=True)))
owned.participate(choice)
print(legacy.collect('behavior').attrs['response']['method'],          # generation (answer)
      choice.collect('behavior:choice').attrs['response']['method'],   # generation (reply)
      choice.collect('behavior:probabilities').attrs['response']['method'])   # likelihood
```

Bare `behavior` is the answer channel of a legacy task; `behavior:choice` is the choice of a `Choice` response. They are different channels, and neither is shorthand for the other: a model whose answer and readout parts can both answer a legacy task states an owner for `behavior` as well. A model with several paths for one answer and no stated owner is refused when it is built. The owner answers its channel. When a task form excludes the owner (for example raw scores, which only the likelihood gives, or a legacy task without an instruction, which only the readout can answer) and exactly one other path can serve it, that path answers; when two or more paths that do not own the channel could serve it, the session is refused before any compute. A choice from the likelihood with probabilities from the readout (or the reverse) cannot come from one session and is refused; the choice alone still runs. To compare paths, build one model for each with only the parts of that path (for example the same network without `answer` and `reply`, so it answers through its readout). A readout asks what information a layer holds; generation asks what the model does when asked.

One benchmark can take models of both kinds, as long as it fills in both the instruction and `fitting_stimuli`. Choices, ratings and implied probabilities come from either kind. Reported confidence and written answers need a model that replies; a readout-only model refuses those sessions before any compute (`ExecutionNotFeasibleError`, an error of `score`).

A reply that cannot be parsed is recorded as invalid: NaN values, `answer_valid` is `False`, the reply is kept in `raw_answer` and the reason in `parse_error`. Nothing is credited. When a session accepts an answer on an addressed `behavior:<format>` channel, core checks it against the task and the trials it answers: the type, the dimensions, the dtype of the format, the `stimulus_id` of each row, the choice axis (the task's options, or positions when the options differ by row) and the response record. A response that fails a check raises and nothing is kept. Core then stores each trial with a state in `response_state` (`valid`, `invalid`, or `missing`) and a reason in `response_reason`. A trial is `missing` when the reply was `None`, or, in a session built from a table, when no answer named its row. A subject may also answer one trial with one cell: a string on `behavior:text` or `behavior:choice`, one number on `behavior:rating`, `behavior:confidence` or `behavior:rt` (with `meta['units']`), or a 1-D array on `behavior:probabilities`; core stores it with a native record whose method is `external`. Written answers are stored as written; scoring them (a parser, a judge, a comparison with people) is the benchmark's job.

**Parts.** `reply(prompt)` returns the model's raw reply to one trial. It may also return extra data the way a policy does: a `StreamEvent` with the text as payload and the extra data (for example `reasoning` or `token_usage`) as meta. The meta goes into the answer event, where `RecordReasoning` reads `meta['reasoning']`. An answer event that holds several trials keeps, under each key, a list with one value per trial. `likelihood(prompt, candidates)` returns the log probability of each candidate answer. `chat_parts(model, processor)` in `brainscore.model_helpers.replies` builds both for a Hugging Face chat model; it shows the prompt's parts in order, images as images and clips as video. This example needs a chat model and its processor (`hf_model`, `processor`), so it downloads weights:

```python
reply, likelihood = chat_parts(hf_model, processor, chat_template_kwargs={'enable_thinking': False})
BrainScoreModel(..., reply=reply, likelihood=likelihood,
                capability_config={'behavior-sessions': {'behavior:choice': 'generation'}})
```

**Generation settings.** The settings of a written answer belong to the task, so the benchmark sets them on `Text`: `max_tokens`, `temperature` (0 or `None` is greedy), `top_p` (needs a temperature above 0) and `seed`. A `reply` part reads them from `prompt.response`. The reply that `chat_parts` builds obeys them. With no settings it is greedy and at most `max_new_tokens` (64) tokens long, as before.

**A part obeys the task or stops.** A `reply` or `likelihood` part reads the task from `prompt.response`. It obeys each setting the task gives (the generation settings of `Text`, `raw_text`, `raw_scores` and the options) or stops with an error; it never ignores a setting. For example, the Claude reply part raises `ValueError` for a temperature other than 1.0, for `top_p` and for `seed`, because the Anthropic API cannot obey them.

**Older task types.** `task_type='label'`, `'classification'` and `'probabilities'` still work on the plain `behavior` channel with `answer`, `label_set` and a readout, so existing benchmarks and scores are unchanged. The same chooser picks the path for them: `answer` (with `instruction` and `label_set`), then the readout (with `fitting_stimuli`). The answer keeps its older form: one-hot rows over `label_set`, where only an exact label counts (`attrs['response']['method']` is `'generation'`), or the readout's probabilities. A behavioral answer comes only from a session (`behavioral_response`, `stimulus_session` and `participate`); `process()` does not answer a task.

## Hosted models

`build_api_generation_fn` (in `brainscore.model_helpers.api_behavioral`; the name is older than the part's) builds an `answer` for a hosted model API. Such a model has no layers, so it can take part in behavioral benchmarks only.

`build_api_reply` builds a `reply` part from the same clients: one table, `PROVIDERS`, with one client per provider. A custom client is a callable `(model, system, messages, settings)` that returns the text, or a `StreamEvent` with the text as payload. It shows the trial's text and images in order, sends the task's instruction as the system text, and obeys the task's generation settings. It returns the reasoning (when the provider gives it) and the token counts (`token_usage`) as extra data. A provider that refuses a setting the task gives fails before any request: the Anthropic API has no seed and refuses `top_p` and any temperature other than its default 1.0 (which is then not sent). Each registered closed model has both parts under one identifier: a task with a response (a choice, a rating, a written answer such as CogGym's) goes to the `reply` part, as for every model that has one; an older `task_type`/`label_set` task goes to the `answer` part.

See [Lesson 6](../../notebooks/lessons/06_behavior.ipynb) and [Lesson 5](../../notebooks/lessons/05_layers_and_regions.ipynb).
