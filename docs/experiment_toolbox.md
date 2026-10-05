# Set up an experiment

Use one `Experiment` API to combine a subject, protocol, tools, and output folder.
The protocol owns the task sequence. Tools record or intervene without changing
the subject's core contract. These APIs are a feature-branch candidate.

```python
from brainscore.experiments import (
    Experiment, RecordInputsOutputs, RecordActivity, Ablate, TorchInstrumentation,
)

experiment = Experiment(
    subject=subject,
    protocol=protocol,
    tools=[
        RecordInputsOutputs(),
        Ablate(['population'], trials=['intervention']),
        RecordActivity(['population']),
    ],
    instrumentation=TorchInstrumentation({'population': model.encoder}),
    output_dir='runs/my-experiment',
    metadata={'checkpoint': 'exact revision', 'seed': 7},
)
experiment.validate()  # Checks compatibility without running inference.
result = experiment.run()
```

Supply the subject, protocol, and model module. The protocol must contain a trial
named `intervention`. For recording alone, omit the last two tools and the
instrumentation provider. The [CPU example](../examples/experiment_toolbox/run.py)
is complete and runnable.

## Choose a protocol

| Protocol | Use it for | Who drives the experiment? |
| --- | --- | --- |
| `SessionProtocol` | Subjects implementing `interact(session)` | Each supplied session determines the next input, including feedback from outputs. |
| `CallableProtocol` | Existing benchmarks or external evaluators | The original evaluator keeps its loop, resets, task definitions, and score. |

For native sessions, supply a context-managed factory. Each trial gets a fresh
session; the subject resets before and after it. The factory closes any simulator,
file handles, or other resources it owns.

```python
from contextlib import contextmanager
from brainscore.experiments import SessionProtocol
from brainscore_core.streaming import InMemorySession, StreamEvent

@contextmanager
def make_session(trial_id):
    yield InMemorySession([
        StreamEvent('observation', {'text': 'Which object moved?'}, t_ms=0),
    ])

protocol = SessionProtocol(
    'my-protocol', make_session,
    trials=['baseline', 'intervention'],
    input_channels=['observation'], output_channels=['answer'],
)
```

The channel names must match your subject's declarations. Payload schemas and
units belong to those channels and adapters. The runner does not infer them from
labels such as vision, language, VLM, or robotics. A stateful subject can emit
both activity and behavior in response to one input.

For an existing benchmark:

```python
from brainscore.experiments import CallableProtocol

protocol = CallableProtocol(
    'existing-benchmark',
    lambda subject, context: benchmark(subject),
    methods=['look_at'],  # Or digest_text, process, or an explicit provider API.
    metadata={'benchmark_revision': 'exact revision'},
)
```

Selected methods must exist. Nested calls are observed once. An inference error
caught by the evaluator still marks the experiment failed. The evaluator owns
reset semantics; use separate executions for baseline/intervention conditions
unless its protocol explicitly handles them. The runner never changes its score.

## Attach tools

| Tool | What it does | Requirement |
| --- | --- | --- |
| `RecordInputsOutputs` | Saves events, trial/call IDs, activity events, and intervention history | Events or selected public method calls |
| `RecordActivity` | Publishes snapshots of selected internal outputs | Instrumentation provider and `RecordInputsOutputs` |
| `Ablate` | Zeros selected internal outputs during selected trials | Instrumentation provider |
| Your own `Tool` | Subscribes to events or manages experimental resources | Public `validate`, `describe`, and `attach` methods |

Order tools explicitly: observers first, then activity recording with
`when='before'`, then interventions, then activity recording with `when='after'`.
Use different names for multiple activity recorders. Tools detach in reverse
order. Interventions attach after a native trial's initial reset and detach
before its final reset, including on errors and cancellation.

`TorchInstrumentation` accepts named PyTorch modules. It currently records
**tensor outputs** and ablates **whole module outputs**. It does not support
arbitrary tuple/dict outputs, individual-unit selection, weight edits, training,
or internal access through a remote API. Unsupported output types fail when
encountered; preflight cannot discover them without running the model. Existing
model hooks remain installed and can affect values observed by these tools.

An external provider can implement `describe()`, `validate(operation, targets)`,
`record(targets, callback)`, and `ablate(targets)`. The last two return context
managers and must clean up partially attached resources. No core edit is needed.
For a remote model, put instrumentation inside its server or expose it explicitly;
do not claim that an input/output connection provides internal access.

## Inspect the outputs

- `experiment.json`: actual protocol, tools, provenance, status, and artifact producers/hashes.
- `result.json`: evaluator return value encoded with the existing record codec.
- `inputs_outputs/`: verified event log and content-addressed arrays/files.
- `external/`: explicitly imported artifacts, credited to their producer.

```python
from brainscore.experiments import read_events

for event in read_events('runs/my-experiment'):
    print(event['trial_id'], event['kind'], event['source'])
```

Input/output events carry the original payload. Activity and lifecycle events
have distinct kinds. `event_id` links observations, activity, and responses where
the protocol boundary permits it. Multiple outputs may follow one input. Event
`t_ms` is experiment time; `elapsed_s` is wall-clock duration. Neither is inferred
from the other. For an external evaluator, finer trial boundaries must be emitted
by its adapter; the default scope is the whole evaluator call.

External evaluators can call
`context.import_artifact(path, name='trial.mp4', producer='OpenPI LIBERO evaluator', description='Trial video')`.
This copies and hashes an existing video; it does not claim Brain-Score created it.
Supply checkpoint hashes, seeds, device/runtime details, and scientific settings
in metadata. The runner records what you provide; it cannot discover every hidden
setting in an external process.

A failed or interrupted run is not a valid completed measurement. Output folders
must be new. Errors propagate and cleanup runs before final status is written.
Large records still inherit `RunRecord`'s memory/materialization limits. Use
bounded runs. Each running experiment needs its own subject, tools, and provider;
concurrent reuse and distributed execution are not supported by this runner.

## Replay saved inputs

```python
from brainscore.experiments import replay_sessions, compare_outputs

replay = Experiment(
    subject=fresh_subject,
    protocol=replay_sessions('runs/my-experiment'),
    tools=[RecordInputsOutputs()],
    output_dir='runs/replay',
).run()
comparison = compare_outputs('runs/my-experiment', replay.directory)
```

Reattach interventions explicitly if reproducing an intervention condition.
Replay never silently executes saved tool configuration. `replay_sessions` feeds
saved inputs into a model; it does not rerun physics or regenerate feedback.
`replay_calls` handles saved method calls with an explicit method allowlist and
caller-supplied reset function. Check model revisions, random state, precision,
and environment when interpreting differences.

`compare_outputs` compares outputs in sequence exactly, including event channels,
timestamps, and metadata. It is an inspection utility, not a task score,
statistical test, or proof of equivalent closed-loop behavior. Use domain metrics
for unsupported payloads or scientific comparisons.

## Add a tool from your own package

Subclass `Tool`, describe its configuration, validate its requirements, and
return a context manager from `attach(context)`. Subscribe with
`context.subscribe(callback)` and register output files with `context.artifact`.
Callbacks must snapshot data before returning and must not mutate live payloads.

[CountEvents](../examples/experiment_toolbox/partner_tool.py) is a complete example
using public imports only. Pass `CountEvents()` alongside the other tools. No
registry changes or edits to `Subject`, `Session`, or Brain-Score are required.

## What the examples prove

The five CPU demonstrations cover image, text, image-plus-text, feedback-control,
and recurrent activity/behavior payloads with the same API. They use small,
untrained models and synthetic data. They verify plumbing and intervention
cleanup, not scientific performance or a whole-brain model.

The [OpenPI call example](../examples/libero/toolbox_calls.py) connects the same
runner to saved LIBERO requests and a policy server. Its websocket test uses a
synthetic policy. The ongoing trained LIBERO qualification uses the earlier
bridge; it does not qualify this runner or demonstrate an internal π₀.₅ ablation.
