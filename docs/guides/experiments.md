# Experiments

An **experiment** runs one subject through a protocol under named conditions and
repetitions, with recorders that say what to save. A **benchmark** is the same protocol
plus reference data and a score.

```
benchmark  = protocol + reference data + score
protocol   : for each condition, for each repetition:
                 reset -> one session -> reset
session    : trial, trial, trial, ...        (or one continuous stream)
```

[Lesson 10](../../notebooks/lessons/10_run_an_experiment.ipynb) builds one study three
ways (a session run by hand, an experiment, a benchmark), and
[Lesson 11](../../notebooks/lessons/11_interventions_in_an_experiment.ipynb) adds
interventions. This page is the reference for what they use.

## The pieces

```python
from brainscore.experiments import Experiment, Intervene, RecordBehavior, RecordNeural, TaskProtocol

protocol = TaskProtocol(stimuli, task, identifier='animal-or-vehicle',
                        conditions=['intact', 'lesioned'], repetitions=[0])
experiment = Experiment(subject=subject, protocol=protocol,
                        tools=[RecordNeural(['IT']), RecordBehavior(),
                               Intervene(lesion, conditions=['lesioned'])])
experiment.validate()          # checks subject, protocol and tools against each other; runs nothing
result = experiment.run()
choices = result.behavior(condition='lesioned', repetition=0, measure='choice')
it = result.neural('IT', condition='lesioned', repetition=0)
```

- `TaskProtocol` builds each session from the stimuli and the task, and resets the subject
  before and after every repetition.
- `validate()` raises before any compute when a recorder asks for a region the subject does
  not declare, a passive task is asked for answers, a required trait is missing, or the
  stimuli carry no input the subject takes.
- A run lives in a temporary folder until the result is no longer used. Pass
  `output_dir='runs/my-study'`, or call `result.save('runs/my-study')`, to keep it. A named
  folder must be new, so a run never overwrites another.

## Conditions and stimulus factors

Conditions name states of the subject: intact, lesioned, a scaled layer. A factor of the
stimulus (image noise, contrast, word length) belongs in the stimulus table as a column.
Every trial then carries its level, and the analysis groups by that column. A design that
crosses both, such as a psychometric curve under a lesion, has the stimulus factor in the
table and the lesion in the conditions.

## Writing it as a benchmark

A small `Benchmark` subclass runs the same experiment in `__call__` and compares the saved
assembly with reference data; the end of Lesson 10 has a complete one, and
[Write a benchmark](write_a_benchmark.md) covers benchmark authoring.

## When to use manual sessions

`TaskProtocol` is the standard path for a fixed stimulus table. Use `SessionProtocol`
with a session factory only for custom event ordering, continuous streams or environments
that need their own session construction. The factory receives
`Repetition(condition, identifier)`, and the protocol still owns resets.
`CallableProtocol` instead wraps an existing evaluator that owns its loop and resets.

For interventions, attach `Intervene(change, conditions=[...])` after the recorders.
It sends the intervention with `apply_state_change` at the start of each matching
repetition and undoes it by its handle at the end. This is the one way to change a
subject in an experiment ([Perturb and select units](perturb_and_select_units.md)). A readout is fitted on the network as it is during the run (`readout_fit='current'`,
the default). `TaskContext(..., readout_fit='intact')` fits it with every change switched
off and tests the changed network, as an animal trained before a lesion is tested after it.
Each change stays in place during the fit, so the order of the hooks on the network does
not change. Only a change made by `build_pytorch_ablation_fn` can be switched off; with
another intervention part active, the fit stops with an error.

For models that generate answers, `RecordNeural(..., during='prompt')` keeps the prompt
forward pass. The extractor must support recording during answers. Declared channels
and traits are pre-flight checks; they do not guarantee every task setting, data file
or extractor implementation is valid. Such errors may be raised when a session starts.

`RecordInputsOutputs` can additionally save input events and lifecycle events for
`result.record` and replay. Neural and behavioral recorders save their own folders;
they do not by themselves create a complete input replay log.

See [Lesson 10](../../notebooks/lessons/10_run_an_experiment.ipynb) for the executable
version and [Write a benchmark](write_a_benchmark.md) for benchmark authoring details.
The bundled [asset notes](../../notebooks/assets/lessons/README.md) record data and weight sources.

# Reference

## Lower-level tools

When the recorders do not cover what you need, build sessions yourself and attach tools that reach into the network. `Ablate`, `ScaleActivity` and `PatchActivity` change the network outside the subject: use them for a model that is not a subject (an OpenPI policy server, a bare network). A subject is changed with `Intervene`.

```python
from contextlib import contextmanager
from brainscore_core.streaming_helpers import stimulus_session
from brainscore.experiments import (Experiment, SessionProtocol, RecordInputsOutputs,
                                    RecordActivity, Ablate, TorchInstrumentation)

@contextmanager
def make_session(repetition):                # repetition.condition, repetition.identifier
    yield stimulus_session(stimuli, ['behavior'])

protocol = SessionProtocol(identifier='silence-layer3', session_factory=make_session,
                           conditions=['baseline', 'silenced'], repetitions=[0, 1, 2],
                           required_input_channels=['vision:image'], requested_output_channels=['behavior'])

experiment = Experiment(
    subject=subject,
    protocol=protocol,
    tools=[RecordInputsOutputs(),                        # save inputs, outputs, events
           Ablate(['layer3'], conditions=['silenced']),   # zero layer3 in one condition
           RecordActivity(['layer3'])],                   # every forward pass, unaggregated
    instrumentation=TorchInstrumentation(net),
    output_dir='runs/silence-layer3',
    metadata={'checkpoint': 'v1.2', 'seed': 7},
)
experiment.validate()          # checks channels and tools without running the model
result = experiment.run()
```

List tools in this order: recorders, activity recorded *before* an intervention, interventions, activity recorded *after*.

## Tools

| Tool | Does |
| --- | --- |
| `RecordNeural(regions)` | Requests `neural:<region>` from the subject and saves it. Read with `result.neural(region, condition=...)`. |
| `RecordBehavior()` | Requests every kind of answer the task's response asks for and saves them. Read with `result.behavior(condition=..., measure=...)`. |
| `Intervene(change, conditions=...)` | Sends an intervention (a `StateChange`) to the subject with `apply_state_change`, undone after each repetition. |
| `RecordInputsOutputs` | Saves every input, output and lifecycle event with its condition and repetition. Needs no access to model internals. |
| `RecordActivity(targets)` | Records layer outputs through instrumentation. |
| `Ablate(targets, conditions=...)` | Zeros layer outputs or a `Selection` of units of a network that is not a subject, in chosen conditions. |
| `ScaleActivity(targets, factor=0.5)` | Multiplies outputs of a network that is not a subject by a factor. |
| `PatchActivity` | Replaces activity with activity recorded on another input. |

A tool watches a run in two places only: the session (`on_event`, `RecordInputsOutputs`,
and `observed(session, sink)` from `brainscore.instrumentation` outside an experiment) for
inputs and outputs, and `TorchInstrumentation` (`RecordActivity`) for layers. No tool wraps
a method of the subject.

Targets are layer paths or `Selection` objects, the same ones used in [Perturb and select units](perturb_and_select_units.md).

## Reopen a saved run

A run kept with `output_dir` or `result.save(path)` can be read again later, in a new session:

```python
from brainscore.experiments import ExperimentResult

result = ExperimentResult.open('runs/my-study')
trials = result.trials()           # one row per trial
it = result.neural('IT', condition='lesioned')
```

`trials()` has `condition`, `repetition` and `stimulus_id`, then the stimulus set's own
columns (a noise level, a word length), then each saved answer: `choice`, `p(<label>)`,
`rt`, `rating`, `text`. For an answer read from a reply, `<answer>_valid`, `<answer>_raw`
and `<answer>_error` say whether it could be read, what the reply was, and why not, so
an analysis can count or drop invalid trials:

```python
valid = trials[trials['choice_valid']]
valid.groupby(['condition', 'noise'])['choice'].apply(lambda c: (c == 'animal').mean())
```

## Read and replay

```python
for event in result.record.events():        # or RunRecord('runs/silence-layer3')
    print(event['condition'], event['repetition_id'], event['kind'])
outputs = list(result.record.outputs())

from brainscore.experiments import replay_sessions, compare_outputs
replay = Experiment(subject=subject, protocol=replay_sessions(result.directory),
                    tools=[RecordInputsOutputs()], output_dir='runs/replay').run()
compare_outputs(result.directory, replay.directory)
```

Replay sends the saved inputs again. It does not re-run a simulator; in closed-loop tasks the new actions do not change the saved observations. Reattach intervention tools explicitly when replaying.

The output folder holds `experiment.json` (setup, status, provenance), `result.json`, and `inputs_outputs/` (events and arrays). Without `output_dir`, that folder is temporary and is deleted with the result; replay and inspection need a kept run, so pass `output_dir` or call `result.save(path)` first. A failed run stays readable for diagnosis in a folder you named; `RunRecord.inspect(path)` reports what survived an interrupted run.

## Wrap an existing evaluator

```python
from brainscore.experiments import CallableProtocol
from brainscore.harnesses.adapter import run_request

def evaluate(subject, context):
    answers = []
    for trial in range(3):
        # One request is one session, recorded as record repetition `trial`.
        session = run_request(subject, context, trial, stimuli, ['behavior'], task)
        answers.append(session.collect('behavior'))
    return my_score(answers)

protocol = CallableProtocol(identifier='classic-benchmark', evaluate=evaluate,
                            repetitions=[0, 1, 2])
```

The evaluator keeps its own loop, resets and scoring; the tools still attach.
`run_request` records each session between its own repetition boundaries, so
`replay_sessions` rebuilds it. For a system that is not a subject (a policy server,
an external library object), name the methods to record instead, for example
`CallableProtocol(..., methods=['infer'])`. There is no default method.

### Connect an external experiment library (an adapter)

An adapter maps an external library (CogGym, EWoK) onto the common channels
(Contract Rules 3.8, 3.9). `brainscore.harnesses.adapter` holds what every adapter
shares, so a new adapter writes only what is specific to its library:

1. How the library states a trial: one stimulus row (its parts in the library's
   order) and a `TaskContext` (the system text as the instruction, the generation
   settings on the response).
2. One `run_request(subject, context, repetition, stimuli, outputs, task)` per
   library request, and how the library parses the answer.
3. How the library scores the answers, as one number.
4. The benchmark: `required_input_channels`, `requested_output_channels`,
   `trial_memory`, `ceiling=not_estimated()`, a `protocol()` that returns a
   `CallableProtocol`, a `__call__` that returns
   `run_recorded(subject, self.protocol(), output_dir)`, and the score from
   `adapter_score(self, value, metric=..., input_adaptation=..., context=context)`,
   where `input_adaptation` says why the number can differ from the library's own.

The adapter checks no compatibility itself: `score()` checks the pair once before
any compute.
