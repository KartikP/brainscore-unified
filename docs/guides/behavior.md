# Behavior

Behavioral benchmarks compare answers instead of activity. A `TaskContext` says what the subject is asked to do; its `response` says what form the answer takes.

The code on this page runs as written with these stand-ins: a few words as stimuli, and a model whose `reply` part always answers "real". Replace them with your stimuli and your model:

```python
import pandas as pd
from brainscore import BrainScoreModel
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


# Words shown as text, with their correct labels
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
| `Choice(labels)` | one label, the probability of each label, or the log probability of each option; the session's requested outputs say which | `behavior:choice`, `behavior:probabilities`, `behavior:log_probabilities` |
| `Choice(labels, reaction_time='surprisal')` (or `'evidence_margin'`) | also an implied reaction time, from the probabilities | `behavior:rt` |
| `Scale(1, 7, 1)` | a rating | `behavior:rating` |
| `Text()` | a written answer, kept as written | `behavior:text` |
| `Text(max_tokens=512, temperature=1.0)` | a written answer under the task's generation settings | `behavior:text` |

These six channels are the behavior formats; core refuses any other `behavior` address. A benchmark asks for a format by the outputs it requests (`requested_output_channels`, and the outputs of its session); the task's response says which formats it allows. A requested format that the response does not name is refused.

Each response defines its directive ("Answer with exactly one of: real, pseudo.") and its parse once, in core. They are the model side's tools: a model part that generates text phrases its answer line with `directive` (the API and Qwen `answer` parts do; BLIP-2 OPT has no chat template and keeps its own wording, "Question: ... Answer:": its `answer` part names the options in that question, and its `reply` part puts the instruction and the directive there), and `BrainScoreModel` reads every reply, from a `reply` or an `answer` part, with `parse`. The benchmark gives the specification only: the instruction, the stimulus text and the options. A task accepts only `Choice`, `Scale` and `Text` themselves, not a subclass, so a benchmark cannot change the directive by subclassing. To add no sentence at all, use `Choice(..., verbatim=True)`.

Two fields of `Choice` and one format serve a library that compares options by their likelihood (EWoK):

| Field | Meaning |
| --- | --- |
| `labels='options'` | a column name in place of the options: the options differ for each row, and the column holds each row's options. The `choice` axis is then the position of each option in that cell. |
| `behavior:log_probabilities` (a requested output, not a field) | each option's summed log probability exactly as the `likelihood` part gave it (float64), not normalized; the record in `attrs['response']` has `category` `'model-derived'` and states the scoring in `parameters`: `layout` `'continuation'`, `context` `'shared'`, `tokenization` (plain text joined by a space, no special tokens, no template, no truncation), `aggregation` `'sum'`, `units` `'nats'`. Only a `likelihood` part gives it, and only for `verbatim=True`; any other request is refused before any compute. |
| `verbatim=True` | the stimulus text is shown exactly as given: no directive, and no instruction. A `likelihood` part scores `f"{text} {option}"` as plain text, with no chat template and no beginning-of-sequence token. A `reply` part gets no added sentence and answers through its own chat template if it has one (BLIP-2 OPT has none: its reply sees the text alone); the record notes that: `parameters['template']`. |

**The order of a trial.** A model that replies receives a `Prompt`: the instruction, the stimulus, then the response's directive. When a stimulus has several parts (a sentence and a clip), say the order people saw them in with `present`; a stimulus with several parts and no order fails before any compute. A model that does not take in a part (a text-only model on a task with a clip) gets the task without that entry, so its trial shows only the parts it takes in. An entry that names a column no stimulus has is an error for every model:

```python
TaskContext(instruction='Which way does the ball move?', present=('text', 'video'),
            response=Choice(['left', 'right', 'up', 'down']))
```

Run the task through a session, or attach `RecordBehavior` in an [experiment](experiments.md):

```python
from brainscore_core.streaming_helpers import stimulus_session

# Session: stimuli, requested outputs, task
session = stimulus_session(test_stimuli, ['behavior:choice'], task)
model.participate(session)
choices = session.collect('behavior:choice')   # BehavioralAssembly: presentation x choice
model.reset()
print(choices.values.tolist())   # [[1.0, 0.0], [1.0, 0.0]]: 'real' for both words
```

## How an answer was given

`session.collect(channel)` gives the answers of a format as one assembly. During the session, an answer given for one trial as a cell is kept on its event in `session.emitted` as its checked row: a dict with the trial's `stimulus_id`, `values`, the `choice` axis (for a choice or probabilities answer), `raw_answer`, `answer_valid`, `parse_error`, `response_state`, `response_reason` and the `response` record. An answer given as an assembly stays an assembly on its event.

Every collected answer carries a response record in `attrs['response']`, and every score copies the records of its answers (a stored answer whose trials were all missing has none: nothing was answered). The record says the format (the channel), the category (`behavioral` or `model-derived`), the origin (`native`, or `derived` by a conversion or a link), the method (`generation`, `likelihood`, `readout`, `link`), the conversion (`argmax`, `softmax`, `softmax_argmax` or none), the source format of a derived answer, the conversion's parameters (for example the tie rule `first`; the readout keeps its own column order for ties in this stage) and who parsed generated text (`parsed_by`). The conversions are core functions in `brainscore_core.response_formats`. A choice from a readout is the `argmax` of its probabilities; a choice from a likelihood part is `softmax_argmax` of the options' log probabilities. A source row with a non-finite value gives an invalid result; in this stage a finite positive log value passes through `softmax` as valid.

| Path | How | Gives | Model needs |
| --- | --- | --- | --- |
| **Generation** | the model replies to the instruction; `BrainScoreModel` parses the reply with the response's `parse` | choice, rating, text | `reply`, or `answer` (it answers a plain choice when it is wired) |
| **Likelihood** | the model scores each label as its reply, without generating | log probabilities (native, verbatim tasks only), choice (`softmax_argmax`), probabilities (`softmax`) | `likelihood` |
| **Readout** | a classifier or ridge regression on one layer, fitted on `fitting_stimuli` | choice (`argmax`), probabilities, rating | `behavioral_readout_layer` and an extractor |

Each kind of answer (choice, rating, probabilities, reaction time, text, log probabilities) comes from one named part of the model: writing (`reply`), a fixed-choice answer (`answer`), the likelihood of the options (`likelihood`) or a trained readout (`behavioral_readout_layer`). A part gives a kind directly or converts another kind (the table above). When two parts could give it, the registration says which one, in `capability_config={'behavior-sessions': {...}}` (the code calls this line a statement, and the chosen part the owner); a kind that a part only converts is given only when that line names it, for example `('likelihood', 'softmax')` for the probabilities. A session that asks for a kind the model has no part for stops before any compute:

```python
def answer(stimulus_row, instruction, label_set):   # answers a plain Choice
    return 'real'


def likelihood(prompt, candidates):                 # the log probability of each option
    return {candidate: 0.0 if candidate == 'real' else -2.0 for candidate in candidates}


owned = BrainScoreModel(
    identifier='owned-parts', in_channels={'text'},
    answer=answer, reply=lambda prompt: 'real', likelihood=likelihood,
    capability_config={'behavior-sessions': {
        'behavior:probabilities': ('likelihood', 'softmax')}})   # derived from the log scores

choice = stimulus_session(test_stimuli, ['behavior:choice', 'behavior:probabilities'], TaskContext(
    instruction='Is this a real word?', response=Choice(['real', 'pseudo'])))
owned.participate(choice)
print(choice.collect('behavior:choice').attrs['response']['method'],   # generation (answer)
      choice.collect('behavior:probabilities').attrs['response']['method'])   # likelihood
```

When one part gives a kind directly, the registration names nothing; a converted kind it does not name is not given:

```python
# Reply and likelihood parts, nothing named
plain = BrainScoreModel(identifier='plain-parts', in_channels={'text'},
                        reply=lambda prompt: 'real', likelihood=likelihood)
choice = stimulus_session(test_stimuli, ['behavior:choice'], TaskContext(
    instruction='Is this a real word?', response=Choice(['real', 'pseudo'])))
plain.participate(choice)
print(choice.collect('behavior:choice').attrs['response']['method'],   # generation (reply)
      'behavior:probabilities' in plain.out_channels)                  # False: no conversion named
```

Bare `behavior` carries the answer of a response task under its first format (the choice of a `Choice`, the rating of a `Scale`, the text of a `Text`); `behavior:choice` is the choice of a `Choice` response. Bare `behavior` is declared when `answer` or a readout (with an extractor) is wired. The registration names no part for it; a line that names it is refused when the model is built, with a message that says to delete the line. The part that gives the response's kind of answer answers, and bare `behavior` carries that answer. A line of the registration names a part for a kind the part gives directly, or a part and a conversion for a kind it converts: the readout gives `behavior:choice` by `('readout', 'argmax')`, the likelihood gives `behavior:probabilities` by `('likelihood', 'softmax')` and `behavior:choice` by `('likelihood', 'softmax_argmax')`. A model with two parts that give one kind directly (`behavior:rating` from `reply` and the readout) and no line for it is refused when it is built, and so is a wrong line; the message says what to write. `behavior:rt` comes from the part that gives `behavior:probabilities` and is never named. A session that asks for a kind the model only converts, with no line for it, is refused, and the message names the line to write, for example `capability_config={'behavior-sessions': {'behavior:probabilities': ('likelihood', 'softmax')}}`. A choice from the likelihood with probabilities from the readout (or the reverse) cannot come from one session and is refused; the choice alone still runs. To compare paths, build one model for each with only the parts of that path (for example the same network without `answer` and `reply`, so it answers through its readout). A readout asks what information a layer holds; generation asks what the model does when asked. A session that requests only `behavior:probabilities` or `behavior:rt`, records no region and keeps no trial history runs only the part that gives the probabilities; the score's `provenance` names only that part.

One benchmark can take models of both kinds, as long as it fills in both the instruction and `fitting_stimuli`. Choices, ratings and implied probabilities come from either kind; a readout gives choices when the registration names `'behavior:choice': ('readout', 'argmax')`, as the registered readout-only models (CLIP, random ViT, GPT-2) do. Written answers need a model that replies; a readout-only model refuses those sessions before any compute (`ExecutionNotFeasibleError`, an error of `score`).

A reply that cannot be parsed is recorded as invalid: NaN values, `answer_valid` is `False`, the reply is kept in `raw_answer` exactly as the model generated it, and the reason in `parse_error`. Nothing is credited. When a session accepts an answer on an addressed `behavior:<format>` channel, core checks it against the task and the trials it names: the type, the dimensions, the dtype of the format, the `stimulus_id` of each row (also a list in `meta['stimulus_id']`), the choice axis of each row (the task's options, or positions when the options differ by row), how often each table row is answered and the response record. A response that fails a check raises and nothing is kept. Core then stores the answer with a state for each trial in `response_state` (`valid`, `invalid`, or `missing`) and a reason in `response_reason`, rows in the order they arrived. An answer is expected for each row of the session's table (the k-th answer with stimulus id X answers the k-th table row with id X) and, in a session without a table, for each `respond` signal. A trial is `missing` when the reply was `None`; once the session has delivered every input, a table row with no answer is stored as a missing row after the answered rows, and a `respond` signal with no answer is stored as a missing row when its episode ends. A session with no table and no `respond` signal requests no trial. A subject may also answer one trial with one cell, naming its row in `meta['stimulus_id']`: a string on `behavior:text` or `behavior:choice`, one number on `behavior:rating` or `behavior:rt` (with `meta['units']`), or a 1-D array on `behavior:probabilities`; core stores it with a native record whose method is `external`. Probabilities stay float64 when the subject gives float64, and are stored as float32 otherwise. A derived answer is an assembly with its record and its source evidence in `raw_answer`:

```python
import numpy as np
from brainscore_core.response_formats import response_record
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

# One trial, 'pseudo', derived by argmax
choice = BehavioralAssembly(
    np.array([[0.0, 1.0]], dtype=np.float32), dims=['presentation', 'choice'],
    coords={'stimulus_id': ('presentation', ['w1']), 'choice': ['real', 'pseudo'],
            'raw_answer': ('presentation', ['[0.2, 0.8]'])},     # the probabilities it came from
    attrs={'response': response_record(format='behavior:choice', method='external',
                                       conversion='argmax',
                                       source_format='behavior:probabilities')})
```

Written answers are stored as written; scoring them (a parser, a judge, a comparison with people) is the benchmark's job.

**Trial states on a score.** A benchmark's metric reads `response_state` and states what it does with an invalid and with a missing trial (for example: an invalid or missing choice counts as wrong). `benchmark_score(..., answers=answer, answer_policy='invalid: wrong, counted; missing: wrong, counted')` writes `n_trials`, `n_valid`, `n_invalid`, `n_missing` and `answer_policy` on the score, and `result_validity`: `'no limit stated'` when the benchmark passes no `invalid_limit` (the default; no check was made); with a limit, `'invalid'` when the share of invalid and missing trials is above it, with `result_validity_reason`, and `'valid'` otherwise. Such a score is returned, not raised. In an [experiment](experiments.md), `ExperimentResult.trials()` has one row per presented trial with `<answer>_state` and `<answer>_reason`; its column `rows_from` says whether the rows come from the answers saved after the session or, for a session that ended with an error, from the answers seen before the error.

**Parts.** `answer(stimulus_row, instruction, label_set)` returns the text the model generated for one choice trial, with at most the echo of its own prompt removed; `BrainScoreModel` reads it with `Choice.parse`, as it reads a reply. A function that returns a label still works: a label reads as itself. The registered `answer` parts show what the row has: its picture (the first of the catalog's picture columns, `image_file_name`, `image_path`, ..., that the row has) and its text (`sentence` or `text`), the text before the instruction; a row with neither stops with an error that names the accepted columns. `reply(prompt)` returns the model's raw reply to one trial. It (and an `answer`) may also return extra data the way a policy does: a `StreamEvent` with the text as payload and the extra data (for example `reasoning` or `token_usage`) as meta. The meta goes into the answer event, where `RecordReasoning` reads `meta['reasoning']`. An answer event that holds several trials keeps, under each key, a list with one value per trial. `likelihood(prompt, candidates)` returns the log probability of each candidate answer. `chat_parts(model, processor)` in `brainscore.model_helpers.replies` builds both for a Hugging Face chat model; it shows the prompt's parts in order, images as images and clips as video, and sends the task's instruction (the text part named `instruction`) as the system turn. A screen in a stimulus column named `instruction` (`present=('image:instruction', ...)`) is an image part and is shown like any other. This example needs a chat model and its processor (`hf_model`, `processor`), so it downloads weights:

```python
reply, likelihood = chat_parts(hf_model, processor, chat_template_kwargs={'enable_thinking': False})
BrainScoreModel(..., reply=reply, likelihood=likelihood,   # '...': your model's other arguments
                capability_config={'behavior-sessions': {
                    'behavior:probabilities': ('likelihood', 'softmax')}})
```

**Generation settings.** The settings of a written answer belong to the task, so the benchmark sets them on `Text`: `max_tokens`, `temperature` (0 or `None` is greedy), `top_p` (needs a temperature above 0) and `seed`. A `reply` part reads them from `prompt.response`. The reply that `chat_parts` builds obeys them. With no settings it is greedy and at most `max_new_tokens` (64) tokens long, as before.

**A part obeys the task or stops.** A `reply` or `likelihood` part reads the task from `prompt.response`. It obeys each setting the task gives (the generation settings of `Text`, `verbatim` and the options) or stops with an error; it never ignores a setting. For example, the Claude reply part raises `ValueError` for a temperature other than 1.0, for `top_p` and for `seed`, because the Anthropic API cannot obey them.

**Older task types.** A `BrainScoreModel` answers a task with a response (`Choice`, `Scale`, `Text`) or `task_type='passive'`; a task such as `task_type='probabilities'` without a response is refused with a `ValueError` that names the response to give. Old vision benchmarks reach a model through the legacy facade: `start_task('label', [labels])` is asked as a `Choice` on `behavior:choice` and turned back into one label per stimulus; `start_task('probabilities', fitting_stimuli)` is asked as a `Choice` among the labels of the fitting stimuli on `behavior:probabilities` (fitted on those stimuli) and handed back as the old probabilities answer, so a model answers it with the part its registration states for `behavior:probabilities` (`qwen2.5-vl-3b`, `qwen2.5-vl-7b`, `qwen2.5-vl-3b-bf16` and `qwen3.6-27b` state the readout and need the fitting stimuli; the `-vwfa` variants state the likelihood); the label task with `'imagenet'`, `odd_one_out` and the language tasks (`next_word`, `reading_times`) are asked in the old form on bare `behavior`. A behavioral answer comes only from a session (`behavioral_response`, `stimulus_session` and `participate`); `process()` does not answer a task.

## Hosted models

`api_answer` (in `brainscore.model_helpers.api_behavioral`) builds an `answer` for a hosted model API; it returns the provider's text exactly. Such a model has no layers, so it can take part in behavioral benchmarks only.

`build_api_reply` builds a `reply` part from the same clients: one table, `PROVIDERS`, with one client per provider. A custom client is a callable `(model, system, messages, settings)` that returns the text, or a `StreamEvent` with the text as payload. It shows the trial's text and images in order, sends the task's instruction as the system text, and obeys the task's generation settings. It returns the reasoning (when the provider gives it) and the token counts (`token_usage`) as extra data. A provider that refuses a setting the task gives fails before any request: the Anthropic API has no seed and refuses `top_p` and any temperature other than its default 1.0 (which is then not sent). Each registered closed model has both parts under one identifier: a plain choice goes to the `answer` part, as for every model with an answer part; a choice with a task setting (`verbatim`, `present`, options per row), a rating or a written answer such as CogGym's goes to the `reply` part.

See [Lesson 6](../../notebooks/lessons/06_behavior.ipynb) and [Lesson 5](../../notebooks/lessons/05_layers_and_regions.ipynb).
