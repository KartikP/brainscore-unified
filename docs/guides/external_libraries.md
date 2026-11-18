# External experiment libraries

This guide is for the maintainer of an experiment library: a collection of experiments with people, each with stimuli, a procedure and the people's responses. You map each experiment to a Brain-Score benchmark once. The experiment then runs on every model that fits, with Brain-Score's tools (recording, lesions, replay) and with your data as the reference. You get a column: results for every model. A modeler who adapts a model once gets a row: results on every experiment that fits, yours included.

Two libraries are mapped today: CogGym (`brainscore/benchmarks/coggym/`, `brainscore/harnesses/coggym.py`) and EWoK (`brainscore/benchmarks/ewok/`, `brainscore/data/ewok/`). Their code and tests (`tests/test_coggym.py`, `tests/test_coggym_benchmark.py`, `tests/test_ewok.py`) are the full examples.

## What your library gives

**Stimuli as a table.** One row per trial, a `stimulus_id` column, and one column per input, named by the channel it feeds: `text` or `sentence` for text, `image_path` for a picture, `video_path` for a clip, `audio_path` for sound. The benchmark keeps the table as `self.stimulus_set`; pre-flight reads the inputs from its columns, so the benchmark does not declare them again. A column that names no input (an item number, the options of a row, a condition) stays in the table for your metric.

**A task with its response.** `TaskContext(response=...)` says what a participant was asked to give:

- `Choice(labels)`: one option. `Choice(labels='options')` names a column whose cell holds each row's own options. The session's requested outputs say whether the answer is the choice, the probability of each option or the log probability of each option.
- `Scale(low, high, step)`: a rating.
- `Text(max_tokens=..., temperature=...)`: a written answer, with the generation settings of your protocol.

Where your protocol fixes the exact text, use `Choice(..., verbatim=True)`: the model sees the trial text exactly as given, with no added sentence and no instruction. Otherwise give the instruction people saw in `TaskContext(instruction=...)`.

**The outputs it requests.** `requested_output_channels` names the formats your metric reads: `behavior:choice`, `behavior:probabilities`, `behavior:log_probabilities`, `behavior:rating`, `behavior:rt` or `behavior:text`. A model that cannot give one of them is not applicable.

**Its own scoring.** Your metric compares the stored answers with your human data and states what it does with an invalid and with a missing trial. A library that parses free text itself requests `behavior:text` and parses the text (CogGym does). A score that can differ from the number of your own evaluator says why in `score.attrs['input_adaptation']`.

## What Brain-Score gives back

- **The stored answer of every trial.** `session.collect(channel)` returns one row per presented trial: the value, the state core gave the trial (`response_state`: `valid`, `invalid` or `missing`), its reason (`response_reason`), the raw response (`raw_answer`) and a record of how the model gave the answer (`attrs['response']`). A trial the model did not answer is a `missing` row.
- **The counts on the score.** `benchmark_score(..., answers=answer, answer_policy=...)` writes `n_trials`, `n_valid`, `n_invalid`, `n_missing`, your policy and `result_validity` on the score (`'no limit stated'` unless you pass `invalid_limit=`; then `'valid'` or `'invalid'`).
- **N/A before any compute** for a model that does not fit: one that takes in none of your inputs, or cannot give a format you request.
- **The same experiment on every registered model that fits**, and the experiment tools of [Experiments](experiments.md) on each run.

## Three shapes

1. **Your metric on answers that Brain-Score collects** (EWoK). The benchmark builds a session from the table and the task, the subject takes part, and the metric reads the stored answers. Use this shape when your evaluator needs only the answers.
2. **Your own runner, driven through a harness** (CogGym). The library keeps its trial loop and its scoring. Each request the runner sends to its model becomes one session (`brainscore.harnesses.adapter.run_request`), and the subject's written answer goes back to the runner. The score is your evaluator's number, unchanged (`brainscore.harnesses.adapter.adapter_score`).
3. **A closed-loop environment.** The subject's action decides the next observation (a game, a simulator). See [Closed loop](closed_loop.md).

## What your library must not do

- **Add a `behavior` format.** The behavior formats above are a closed list; core refuses any other `behavior` address. An action or a measurement that is not an answer is another kind of output: see [A new kind of output](new_output.md).
- **Send a private format.** An answer is one of the formats above, in its schema; core refuses a payload that does not match.
- **Add its own switch to `Choice`.** A task takes `Choice`, `Scale` and `Text` themselves, not a subclass. The settings of a format are its fields (`labels`, `verbatim`, the generation settings of `Text`).

## Where the data lives

The stimuli and the human responses are a data plugin in `brainscore/data/<name>/`, with the scripts that package them ([Data](data.md)). Data you may not redistribute stays local and is declared as a local asset: EWoK's prepared items (`$BRAINSCORE_DATA_HOME/ewok-core-1.0`). A library with packages of its own runs in its own environment: CogGym's evaluator runs in a separate process, from a folder Brain-Score makes at the first load (the pinned checkout and a virtual environment with CogGym's pinned packages; `BS_INSTALL_DEPENDENCIES=no` forbids making it), and each model call comes back to your process. Nothing is installed into the scoring environment: git and pip get only allowlisted variables, and pip refuses to run outside the folder's environment. Linux and macOS only ([the CogGym page](../../brainscore/data/coggym/README.md)). A registered benchmark has brain or behavior data, or reads a model-derived measure under a published evaluation protocol and states its reference (EWoK's log-probability form compares likelihoods by the criterion of its paper).

## A complete example

A made-up library of four items: a context, two continuations, and the continuation most people chose. The benchmark asks for one continuation per item, verbatim, and scores agreement with the people's choices:

```python
import brainscore
from brainscore import BrainScoreModel
from brainscore.benchmarks import benchmark_score, not_estimated
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

# The library's items: a context, two continuations, and the one most people chose.
ITEMS = [
    ('i1', 'The glass fell off the table and', ('shattered', 'sang'), 'shattered'),
    ('i2', 'She put the milk back in the', ('fridge', 'oven'), 'fridge'),
    ('i3', 'He opened his umbrella because it was', ('raining', 'Tuesday'), 'raining'),
    ('i4', 'The baby was tired, so she', ('slept', 'calculated'), 'slept'),
]


class MadeUpLibraryChoice(BenchmarkBase):
    """Agreement of the subject's continuation with the continuation people chose."""
    requested_output_channels = frozenset({'behavior:choice'})

    def __init__(self):
        self.stimulus_set = StimulusSet({
            'stimulus_id': [item for item, _, _, _ in ITEMS],
            'text': [context for _, context, _, _ in ITEMS],         # the catalog's text column
            'options': [options for _, _, options, _ in ITEMS]})     # each row's own options
        self.stimulus_set.identifier = 'made-up-library-items'
        self.people = {item: chosen for item, _, _, chosen in ITEMS}
        # The library's protocol fixes the text: verbatim adds no sentence.
        self.task = TaskContext(response=Choice(labels='options', verbatim=True))
        super().__init__(identifier='MadeUpLibrary-choice', version=1, parent='behavioral',
                         ceiling=not_estimated())

    def __call__(self, subject):
        # Run the items, then score agreement with people
        session = stimulus_session(self.stimulus_set, ['behavior:choice'], self.task)
        subject.participate(session)
        answer = session.collect('behavior:choice')       # one stored row per item
        options = dict(zip(self.stimulus_set['stimulus_id'], self.stimulus_set['options']))
        agree = []
        for item, state, row in zip(answer['stimulus_id'].values, answer['response_state'].values,
                                    answer.values):
            chosen = options[item][int(row.argmax())] if state == 'valid' else None
            agree.append(chosen == self.people[item])   # an invalid or missing answer counts as wrong
        return benchmark_score(self, raw=sum(agree) / len(agree), metric='agreement with people',
                               answers=answer,
                               answer_policy='invalid: wrong, counted; missing: wrong, counted')
```

Two stand-in models take part: one with a `likelihood` part, which scores each continuation, and one with a `reply` part, which writes an answer. A third reply model cannot answer one item. A model with a likelihood part but no registration line for the choice does not declare `behavior:choice`, so it is not applicable:

```python
def likelihood(prompt, candidates):
    """A stand-in language model: a shorter continuation is more likely."""
    return {candidate: -float(len(candidate)) for candidate in candidates}


# Three stand-in models: likelihood, reply, unsure reply
scorer = BrainScoreModel(identifier='likelihood-stand-in', in_channels={'text'}, likelihood=likelihood,
                         capability_config={'behavior-sessions': {
                             'behavior:choice': ('likelihood', 'softmax_argmax')}})
replier = BrainScoreModel(identifier='reply-stand-in', in_channels={'text'},
                          reply=lambda prompt: prompt.response.labels[0])   # always the first option
unsure = BrainScoreModel(identifier='unsure-stand-in', in_channels={'text'},
                         reply=lambda prompt: 'I am not sure' if 'milk' in prompt.text()
                         else prompt.response.labels[0])

for subject in (scorer, replier, unsure):
    score = brainscore.score(subject, MadeUpLibraryChoice(), check_mem=False)
    print(subject.identifier, float(score), score.attrs['n_valid'], score.attrs['n_invalid'])
# likelihood-stand-in 0.5 4 0
# reply-stand-in 1.0 4 0
# unsure-stand-in 0.75 3 1

# No line for the choice: the pair is N/A
viewer = BrainScoreModel(identifier='no-choice', in_channels={'text'}, likelihood=likelihood)
print(brainscore.score(viewer, MadeUpLibraryChoice(), check_mem=False).attrs['status'])   # N/A
```

To publish the experiment, put the items and the human choices in a data plugin, load them in the benchmark's `__init__`, and add one registry line: `benchmark_registry['MadeUpLibrary-choice'] = MadeUpLibraryChoice` ([Write a benchmark](write_a_benchmark.md)).
