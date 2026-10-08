# Set up an experiment

An `Experiment` combines a **subject** (the model-facing interface), a **protocol** (what to run), and **tools** (what to record or change). These use the same subjects, layer selections, and saved records as the rest of UMI.

Start with [pretrained ResNet-18](../notebooks/18_resnet_experiment_toolbox.ipynb) or the [small digit classifier](../notebooks/17_experiment_toolbox.ipynb). Both run on CPU.

## How the pieces fit

| Part | Responsibility |
| --- | --- |
| `Experiment` | Run the subject with one protocol and the selected tools; save results and clean up. |
| `SessionProtocol` / `CallableProtocol` | Define the procedure: run sessions, or wrap an existing evaluator. |
| `RecordInputsOutputs`, `RecordActivity`, `Ablate` | Ready-to-use tools attached to the run. |
| `TorchInstrumentation` | Connect activity/intervention tools to the actual PyTorch network. |
| `OpenPIInstrumentation` / `RemoteOpenPIInstrumentation` (feature branch) | Connect the same tools to the pinned local/remote OpenPI JAX policy. |
| `observe`, `ActivationWindow`, `RunRecorder` | Building blocks used underneath the tools; also usable directly. |
| `build_trace_subject` | Adapt a generated-response provider into a subject. It produces traces; recorders save them. |

`ObserveCalls` wraps `observe` for experiments. `RecordActivity` uses `ActivationWindow` through instrumentation. `RecordInputsOutputs` writes through `RunRecorder`. These are shared implementations at different levels of control.

For closed-loop episodes, see [environment sessions](environment_sessions.md).

For exposed CoT and streamed responses, see [reasoning recording](reasoning_recording.md). For policies, see [robotics instrumentation](policy_instrumentation.md).

## Choose the protocol

| Protocol | Use it when | Who supplies the next input? |
| --- | --- | --- |
| `SessionProtocol` | Your subject implements `interact(session)` | The session, which can respond to model outputs. |
| `CallableProtocol` | You already have a benchmark or evaluator | The existing evaluator, with its scoring and reset rules. |

For sessions, name the **conditions** you want to compare and the **trials** that repeat each condition:

```python
from contextlib import contextmanager
from brainscore_core.streaming import InMemorySession, StreamEvent
from brainscore.experiments import SessionProtocol

@contextmanager
def make_session(trial):
    # trial.condition names the setting; trial.identifier identifies the repetition.
    # Supply already prepared inputs and collect the model's responses.
    yield InMemorySession([StreamEvent('image', images.copy(), t_ms=0)])

protocol = SessionProtocol(
    'compare-conditions', make_session,
    conditions=['normal', 'silenced', 'restored'],
    trials=[0],  # One repetition of each condition.
    input_channels=['image'], output_channels=['prediction'],
)
```

Supply `images` and a subject with matching channels. The protocol resets the subject before and after each trial. The factory creates a fresh session and closes any resources it owns. `@contextmanager` lets Python manage that setup and cleanup around `yield`.

`session.next_input()` returns `None` only when the session has ended. Inputs may be images, text, robot observations, or other documented payloads; channel names alone do not define their shape or units.

For an existing evaluator:

```python
from brainscore.experiments import CallableProtocol

protocol = CallableProtocol(
    'existing-benchmark',
    lambda subject, context: benchmark(subject),
    methods=['look_at'],  # Select the public calls to observe.
)
```

Use the methods your evaluator calls, such as `look_at`, `digest_text`, or `process`. Nested calls are observed once. The evaluator retains its task, reset, and score semantics. Use separate experiments for its baseline and intervention conditions unless the evaluator provides those boundaries itself.

## Choose the tools

```python
from brainscore.experiments import (
    Experiment, RecordInputsOutputs, RecordActivity, Ablate, TorchInstrumentation,
)

experiment = Experiment(
    subject=subject,
    protocol=protocol,
    tools=[
        RecordInputsOutputs(),  # Save inputs, outputs, and tool measurements.
        Ablate(['layer3.0.bn2'], conditions=['silenced']),  # Zero this layer output.
        RecordActivity(['layer3.0.bn2']),  # Measure activity after the change.
    ],
    instrumentation=TorchInstrumentation(model),  # Use the model's actual layer paths.
    output_dir='runs/my-experiment',
    metadata={'checkpoint': 'exact revision', 'seed': 7},
)
experiment.validate()  # Check compatibility without running the model.
result = experiment.run()  # Run, save results, and detach tools.
```

This example assumes the session protocol above and a ResNet model. For recording alone, omit the activity/ablation tools and instrumentation.

| Tool | Purpose |
| --- | --- |
| `RecordInputsOutputs` | Save input/output, activity, and lifecycle events with condition, trial, and call IDs. |
| `RecordReasoning` (feature branch) | Save exposed reasoning and response context in a focused record; requires `RecordInputsOutputs`. |
| `RecordActivity` | Record selected layer outputs through `ActivationWindow`. Requires `RecordInputsOutputs`. |
| `Ablate` | Zero selected outputs using the same implementation as `StateChange` interventions. |
| `ObserveCalls` | Attach an existing `on_start` / `on_result` / `on_error` observer. |
| Your own `Tool` | Observe events or manage resources from your own package. |

Order tools as: observers, activity with `when='before'`, interventions, activity with `when='after'`. Give multiple recorders distinct names. Tools detach in reverse order, including on errors. Other hooks already installed on the model remain in place.

### Select layers and units

Use the same dotted layer paths throughout UMI, such as `layer3.0.bn2`. To select units, pass the existing `Selection` type:

```python
from brainscore_core.events import Selection

selected = Selection('encoder', indices=[0, 3])
# These indices address the last output axis, as they do in StateChange.
activity = RecordActivity([selected])
ablation = Ablate([selected], conditions=['silenced'])
```

For a linear layer, the last axis contains its units. For a convolutional output, it is usually image width, **not channels**. Check the output shape before choosing indices.

`TorchInstrumentation` records tensor outputs, the first tensor in a tuple/list, or supported transformer output fields. Ablation supports tensors and tuples beginning with a tensor; other tuple entries are unchanged. Unsupported outputs raise errors when encountered. This does not provide weight editing or internal access to a remote model.

An external instrumentation provider implements `describe`, `validate`, `record`, and `ablate`. The last two return context managers and must clean up after partial attachment failures. No core changes are needed.

## Read saved results

```python
# RunRecord is the shared reader for standalone recordings and experiments.
for event in result.record.events():
    print(event['condition'], event['trial_id'], event['kind'])

for response in result.record.outputs():
    print(response)  # Only model responses, excluding activity and lifecycle events.
```

You can also open `RunRecord('runs/my-experiment')`. Events retain their original payloads. `event_id` connects input, activity, and output where the protocol permits it. `t_ms` is experiment time; `elapsed_s` is wall-clock duration.

| Location | Contents |
| --- | --- |
| `experiment.json` | Setup, status, provenance, and artifact hashes/producers. |
| `result.json` | The protocol's return value. |
| `inputs_outputs/` | Event log and saved arrays/files. |
| `external/` | Artifacts explicitly copied from an external evaluator. |

For external videos or reports, use `context.import_artifact(path, name='trial.mp4', producer='LIBERO evaluator', description='Trial video')`. This records the actual producer. Supply checkpoint, seed, precision, and environment details in metadata; UMI cannot discover every setting in an external process.

Failed runs remain readable for diagnosis but cannot be replayed as completed measurements. Use a new output directory and separate subject/tools for each experiment. This synchronous runner does not support concurrent reuse or distributed execution. Keep recordings bounded; the codec materializes individual assets in memory.

## Replay saved inputs

```python
from brainscore.experiments import replay_sessions, compare_outputs

replay = Experiment(
    subject=subject,
    protocol=replay_sessions(result.directory),  # Preserve conditions, trials, and inputs.
    tools=[RecordInputsOutputs(), Ablate(['layer3.0.bn2'], conditions=['silenced'])],
    instrumentation=TorchInstrumentation(model),
    output_dir='runs/replay',
).run()
comparison = compare_outputs(result.directory, replay.directory)
```

Reattach interventions explicitly: saved settings never execute automatically. `replay_sessions` sends saved inputs to the model; it does not rerun a simulator or regenerate feedback. For saved method calls, `replay_calls` requires an explicit method list and reset callback.

`compare_outputs` checks exact output agreement, including channels, timestamps, and metadata. It is not a benchmark score or evidence of equivalent closed-loop behavior. Model versions, random state, precision, and runtime can affect agreement.

## Add a tool

```python
from brainscore.experiments import Tool

class CountOutputs(Tool):
    name = 'output_count'

    def __init__(self):
        self.count = 0

    def on_event(self, event):
        # Count responses without changing the data sent to the model.
        if event['kind'] == 'output':
            self.count += 1
```

Pass `CountOutputs()` in the tool list. For resources or saved files, override `attach(context)` with a context manager; see [CountEvents](../examples/experiment_toolbox/partner_tool.py). Add `validate` for requirements and `describe` for configuration. Copy live payloads before retaining them, and do not mutate them.

For existing method observers, use `ObserveCalls(observer, methods=['process'])`. Method callbacks describe calls; `on_event` also sees session and lifecycle events. These are distinct boundaries, not interchangeable payloads. See [tool authoring](tool_authoring.md) for lower-level control.

## Evidence limits

The [five CPU examples](../examples/experiment_toolbox/run.py) exercise image, text, image-plus-text, feedback-control, and recurrent activity/behavior with synthetic data. They verify integration, not scientific performance. Notebooks 17 and 18 add trained image-model demonstrations.

The [OpenPI example](../examples/libero/toolbox_calls.py) uses saved LIBERO requests and a policy server. Its websocket test uses a synthetic policy. The trained LIBERO qualification uses the separate bridge; it does not qualify internal policy ablation. The feature-branch [OpenPI providers](policy_instrumentation.md) also have [trained L4 and ten-task LIBERO checks](qualification/2026-10-06-openpi-tools.md), including exact recording-only actions under controlled compiler settings.
