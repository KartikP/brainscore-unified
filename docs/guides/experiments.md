# Experiments

A benchmark gives one score. A study usually needs more: several conditions, recordings of activity and behavior, and a saved record to analyze later. `Experiment` runs a study the way a lab does: design the task, invite a subject, attach the equipment, run.

| Term | Meaning |
| --- | --- |
| **Protocol** | The procedure. `TaskProtocol` presents stimuli under a task; `SessionProtocol` runs sessions you build yourself; `CallableProtocol` wraps an evaluator that already exists. |
| **Condition** | A setting being compared, such as `baseline` or `silenced`. |
| **Trial** | One independent repetition of a condition. The subject is reset before and after each trial. |
| **Tool** | Equipment attached to the run: recorders and interventions. The subject is not modified. |

## A task with recorders

```python
from brainscore_core import Perturbation, Selection, StateChange, TaskContext
from brainscore.experiments import (Experiment, TaskProtocol, RecordNeural,
                                    RecordBehavior, Intervene)

model = load_model('qwen3.6-27b')                                    # 1. the subject
stimuli = ...                                                         # 2. a StimulusSet
task = TaskContext(task_type='label', label_set=['real', 'pseudo'],  # 3. the task
                   instruction='Is this a real word?', prefer_path='generation')
lesion = StateChange(kind='ablation', target=Selection(layer='blocks.12'),
                     perturbation=Perturbation(kind='zero'))

result = Experiment(
    subject=model,
    protocol=TaskProtocol(stimuli, task, conditions=['baseline', 'lesioned']),
    tools=[RecordNeural(['IT', 'language_system']),                   # 4. neural recorder
           RecordBehavior(),                                          # 5. behavior recorder
           Intervene(lesion, conditions=['lesioned'])],
    output_dir='runs/lexical-decision',                               # must be new
).run()                                                               # 6. run

result.neural('IT', condition='lesioned')       # NeuroidAssembly
result.behavior(condition='lesioned')           # BehavioralAssembly
```

The task says what the subject does; the recorders say what is measured. Each session requests exactly the recorders' channels, and each stimulus is presented once per trial, so the recording is the activity that produced the answer.

| Task | Recorders | Saved |
| --- | --- | --- |
| `TaskContext(task_type='passive')` | `RecordNeural` | activity during plain viewing |
| a task with answers | `RecordNeural` | activity during the task (answers are computed, not saved) |
| a task with answers | `RecordBehavior` | answers |
| a task with answers | both | both, from the same presentation |

`Experiment.validate()` fails before any compute when a recorder's channel cannot be produced: a region the subject does not declare, `RecordBehavior` on a passive task, or a protocol that does not request the channel.

How the activity is taken depends on how the subject answers:

- **With `answer`** (a model that generates a reply), each extractor records the forward pass that reads the stimulus and instruction (`RecordNeural(..., during='prompt')`). The passes that generate the reply are not kept. The extractor needs a `record(layers)` method; `TextWrapper` and `VisionWrapper(kind='vlm')` have one.
- **With a readout**, the region layers are extracted in the same call as the readout layer. In a condition with an intervention, the readout is fitted on the intact network and tested on the changed one, as an animal trained before a lesion is tested after it. `TaskProtocol(..., refit_readout=True)` fits it on the changed network instead, which measures what information survives.

`result.neural(region, condition=...)` returns a `NeuroidAssembly` whose `attrs['recording']` says how it was made (task type, path, forward pass, readout fit, condition), so a recording made during a task is never mistaken for one made during plain viewing.

`Intervene` applies a change through the subject's own `intervention` at the start of each matching trial and undoes it at the end. Its targets use the same layer paths as `region_layer_map`.

## Lower-level tools

When the recorders do not cover what you need, build sessions yourself and attach tools that reach into the network:

```python
from contextlib import contextmanager
from brainscore_core.streaming import BufferedSession
from brainscore.experiments import (Experiment, SessionProtocol, RecordInputsOutputs,
                                    RecordActivity, Ablate, TorchInstrumentation)

@contextmanager
def make_session(trial):                     # trial.condition, trial.identifier
    yield BufferedSession(stimulus_set=stimuli, requested_output_channels=['behavior'])

protocol = SessionProtocol(identifier='silence-layer3', session_factory=make_session,
                           conditions=['baseline', 'silenced'], trials=[0, 1, 2],
                           input_channels=['vision'], output_channels=['behavior'])

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
| `RecordBehavior()` | Requests `behavior` and saves it. Read with `result.behavior(condition=...)`. |
| `Intervene(change, conditions=...)` | Applies a `StateChange` through the subject's `intervention`, undone after each trial. |
| `RecordInputsOutputs` | Saves every input, output and lifecycle event with its condition and trial. Needs no access to model internals. |
| `RecordActivity(targets)` | Records layer outputs through instrumentation. |
| `Ablate(targets, conditions=...)` | Zeros layer outputs or a `Selection` of units, in chosen conditions. |
| `ScaleActivity(targets, factor=0.5)` | Multiplies outputs by a factor. |
| `PatchActivity` | Replaces activity with activity recorded on another input. |
| `ObserveCalls` | Attaches an existing observer to method calls. |

Targets are layer paths or `Selection` objects, the same ones used in [Perturb and select units](perturb_and_select_units.md).

## Read and replay

```python
for event in result.record.events():        # or RunRecord('runs/silence-layer3')
    print(event['condition'], event['trial_id'], event['kind'])
outputs = list(result.record.outputs())

from brainscore.experiments import replay_sessions, compare_outputs
replay = Experiment(subject=subject, protocol=replay_sessions(result.directory),
                    tools=[RecordInputsOutputs()], output_dir='runs/replay').run()
compare_outputs(result.directory, replay.directory)
```

Replay sends the saved inputs again. It does not re-run a simulator; in closed-loop tasks the new actions do not change the saved observations. Reattach intervention tools explicitly when replaying.

The output folder holds `experiment.json` (setup, status, provenance), `result.json`, and `inputs_outputs/` (events and arrays). A failed run stays readable for diagnosis; `RunRecord.inspect(path)` reports what survived an interrupted run.

## Wrap an existing evaluator

```python
from brainscore.experiments import CallableProtocol

protocol = CallableProtocol(identifier='classic-benchmark',
                            evaluate=lambda subject, context: benchmark(subject),
                            methods=['look_at'])     # calls to observe
```

The evaluator keeps its own loop, resets and scoring; the tools still attach.

See [Lesson 10](../../notebooks/lessons/10_run_an_experiment.ipynb), [Experiment 4](../../notebooks/experiments/04_silencing_units_in_a_digit_classifier.ipynb) and [Experiment 9](../../notebooks/experiments/09_recording_and_silencing_resnet18.ipynb).
