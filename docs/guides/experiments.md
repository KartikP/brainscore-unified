# Experiments

A benchmark gives one score. A study usually needs more: several conditions, repetitions, recordings of internal activity, and a saved record to analyze later. `Experiment` provides this without changing the subject.

| Term | Meaning |
| --- | --- |
| **Protocol** | The procedure. `SessionProtocol` creates a fresh session for every trial; `CallableProtocol` wraps an evaluator that already exists. |
| **Condition** | A setting being compared, such as `baseline` or `silenced`. |
| **Trial** | One independent repetition of a condition. The subject is reset before and after each trial. A trial can contain many stimuli or a whole episode. |
| **Tool** | Equipment attached to the run. It observes or changes the run; the subject is not modified. |
| **Instrumentation** | The connection that lets tools reach a model's layers, such as `TorchInstrumentation(net)`. |

## Set up and run

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
           RecordActivity(['layer3'])],                   # record layer3 after the change
    instrumentation=TorchInstrumentation(net),
    output_dir='runs/silence-layer3',                    # must be new
    metadata={'checkpoint': 'v1.2', 'seed': 7},
)
experiment.validate()          # checks channels and tools without running the model
result = experiment.run()
```

List tools in this order: recorders, activity recorded *before* an intervention, interventions, activity recorded *after*.

## Tools

| Tool | Does |
| --- | --- |
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
