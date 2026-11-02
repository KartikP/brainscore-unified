# Build tools and integrations with UMI

This guide describes the v2 source candidate and its coordinated peer packages.
The public release remains subject to the [release policy](production_release.md).

When interpreting small measured effects, read the [numerical policy](numerical_policy.md).
Record precision, device and execution configuration with your tool's results.
The fixed benchmark parity budget does not establish an error bound for a new
experiment or intervention; compare its effect with matched numerical controls.

For a single setup that lists a protocol and its tools, start with the
[experiment toolbox](experiment_toolbox.md). It composes these interfaces without
changing the core contract.

## Shared vocabulary

Use the [UMI vocabulary and conventions](conventions.md) across integrations. The experiment tools reuse these interfaces: `RecordActivity` uses `ActivationWindow`, `Ablate` uses the `StateChange` PyTorch implementation, and `result.record` returns `RunRecord`. An existing call observer can be included as `ObserveCalls(observer, methods=['process'])`.

## Choose the extension point

| Your contribution | Public entry point | What you own |
| --- | --- | --- |
| Observe model calls | `brainscore.instrumentation.observe` | `on_start`, `on_result`, `on_error` callbacks |
| Record an experiment | `RunRecorder` plus `observe` or `ObservedSession` | Model, data, protocol and configuration provenance |
| Analyze saved measurements | `RunRecord.events`, `outputs`, `evaluate` | Analysis/metric and target data |
| Collect generated responses | `build_trace_subject` | Provider callable and explicit answer parser |
| Measure internal activity | `ActivationWindow` / `PerceptWindow` | Supported model module and selected layers |
| Change model activity | `intervene` and `StateChange` | Model-supported perturbation and condition design |
| Estimate and verify unit influence | `TorchInstrumentation.attribute`, `PatchActivity` | Paired inputs, differentiable objective, independent validation and controls; [guide](attribution_patching.md) |
| Add a sensory domain | `register_channel(..., columns=...)` and an extraction wrapper | Payload validation, materialization and model extraction |
| Add an output or session | `Capability` with `StreamEvent` | Schema, channel declarations, dispatch and lifecycle |
| Evaluate a robotics policy | `ActionSpec`, `droid_steps`, `evaluate_droid_episode` | Action semantics, policy, data and scientific protocol |
| Drive a controlled environment | `EnvironmentSession` | Environment reset/step, action validation, timing and safety |

Robotics helpers are exported from `brainscore.harnesses.robotics`. For a policy
that returns action chunks, follow [Connect a DROID policy](droid_integration.md).
The [first-author trial](partner_trial.md) helps check the guide with a new user.

Put these implementations in your own Python package. Explicitly import its
registration module before constructing models. No edits to Subject, core event
unions, BrainScoreModel or Brain-Score repositories are needed. Automatic plugin
discovery is not required. Treat registration as process initialization; do not
replace registrations while existing models are running.

For an installable package, copy [examples/partner_tool](../examples/partner_tool/README.md).
It runs outside the source tree and combines new channels, a capability, an observer,
record/replay and reset using only public imports.

## Your first independent tool

```python
from brainscore.instrumentation import observe

class LatencyTool:
    def __init__(self):
        self.seconds = []

    def on_result(self, call, result):
        self.seconds.append(call.duration_s)

tool = LatencyTool()
with observe(model, tool):
    score = benchmark(model)
```

This covers `process`, legacy `look_at`, and `digest_text`. Nested calls on one
subject count once per observer; independent nested observers each see the call.
Exact prior method overrides are restored on exit, including exceptions. Exit
contexts in reverse attachment order. Use a separate model instance for each
concurrent experiment. Observer failures propagate; an incomplete record must
not silently become a valid result. A tool that needs immutable inputs must copy
them in `on_start`, before inference can mutate them.

For native `interact` implementations, pass `ObservedSession(session, recorder)`
to observe input/output events. `observe` records only calls made on the object
to which it is attached. Methods with arbitrary custom names can be included via
the `methods` argument.

## Save and replay measurements

```python
from brainscore.instrumentation import observe
from brainscore.run_record import RunRecorder, RunRecord

with RunRecorder('new-run-directory', metadata={
    'model': 'my-model', 'checkpoint': 'exact-revision',
    'benchmark': 'my-benchmark', 'protocol': 'fixed-context-v1',
    'seed': 0, 'generation': {'temperature': 0},
}) as record:
    with observe(model, record):
        result = benchmark(model)

saved = RunRecord('new-run-directory')
for measurement in saved.outputs():
    print(type(measurement))
```

The schema is versioned. Input snapshots, raw outputs, call order, durations,
errors, numeric arrays and supported assembly coordinates are retained. Arrays
and file assets have SHA-256 identities; event logs have a checksum when closed.
Unsupported schemas/payload types fail explicitly. No pickle or dynamically
imported payload classes are used. An exception marks the run failed, including
model exceptions caught by the calling experiment. Interrupted records remain
`open` and are diagnostic evidence, not completed experiments.

A recording failure (an unsupported payload, an I/O error, a damaged existing
asset, or an exceeded budget) is raised unchanged and marks the record `failed`,
even if the caller catches it. Later calls refuse to record; nothing is retried.

To cap a record's size on disk, pass `max_bytes`:

```python
RunRecorder('new-run-directory', metadata={...}, max_bytes=500_000_000)
Experiment(..., tools=[RecordInputsOutputs(max_bytes=500_000_000)], ...)
```

The cap covers `events.jsonl` plus the unique array and file assets the record
stores, including assets in its metadata. A duplicate asset is counted once.
Bytes are reserved before each write, so stored bytes never exceed the cap. A
write that would exceed it raises `RecordingBudgetExceeded` and writes nothing.
The manifest reports `budget: {max_bytes, reserved_bytes, exceeded}`.
`reserved_bytes` can exceed the bytes actually stored when a write failed
partway, and assets kept from a failed event stay reserved. Under `Experiment`
the run fails and normal cleanup runs.

The manifest is outside this limit; disk errors can still prevent saving it.
The cap does not limit memory: each event is fully serialized in memory before
it is checked. It is also not an experiment-wide quota. `result.json`,
`external/`, `experiment.json` and other tools' directories (such as
`reasoning/`) are not counted.

`RunRecord.events()` checks the whole event log (checksum, JSON, count and order)
before yielding the first event, then streams it. Memory grows with the largest
single event, not with the number of events; collecting events into a list
(`list(record.events())`) still holds them all. Do not modify, replace or link a
closed record while reading it. The reader stops with an error if the log file's
identity, size, modification time or change time differs from when reading
began. This catches ordinary writes, including a write that restores the
modification time. It is not a defense against someone who can forge file
timestamps, or whose write falls within the filesystem's timestamp resolution.
Open (interrupted) records are read as an unchecked prefix for diagnosis only.

Pass local file assets as `pathlib.Path` objects to copy and hash their contents;
StimulusSet `stimulus_paths` are copied automatically. Arbitrary strings remain
strings and are not assumed to be file paths. A raw path string in another
payload is therefore not a self-contained observation. Store large continuous
streams in bounded runs; this initial codec materializes individual assets in
memory. Keep credentials out of metadata and choose access/storage permissions
appropriate to participant or provider data. Hashes detect corruption; they are
not signatures authenticating the experimenter.

`RunRecord.evaluate(metric, target, output_index=0)` runs an analysis on a stored
output. It does not call the model. Re-running observations through a model is a
different experiment and requires its own randomness, checkpoint, environment
and tolerance policy. A saved trace cannot establish hidden reasoning or causal
mechanism by itself.

## Preserve generated responses

```python
from brainscore.model_helpers.response_trace import build_trace_subject
from brainscore_core.streaming import StreamEvent

model = build_trace_subject(
    'my-api-model',
    provider=lambda request: {'text': '42', 'usage': {'output_tokens': 1}},
    parse=int,
    provenance={'provider': 'fixture', 'model_revision': 'v1'},
)
response = model.process(StreamEvent('generation_request', {'prompt': 'Answer'}, 0))
assert response.payload['answer'] == 42
assert response.payload['raw']['text'] == '42'
```

The provider returns a dict containing `text`. Preserve provider-exposed token,
reasoning, finish-reason, cache and usage fields there when available. Parser
ValueError/TypeError produces `valid=False`, `answer=None` and `parse_error`, while
retaining the raw response. Provider errors propagate. This explicit path leaves
legacy behavioral label/one-hot scoring behavior unchanged.

For a focused CoT record and streamed fragments, see [reasoning recording](reasoning_recording.md). `RecordReasoning` uses the same record format and retains links to the complete input/output log.

## New input/output domains

```python
from brainscore_core.extensions import register_channel, CatalogEntry

register_channel(CatalogEntry(
    'tactile', 'input', 'StimulusSet',
    'sensor-specific pressure array; document units and shape',
    'my_lab.tactile', owner='my_lab',
    shape_validator=lambda value: hasattr(value, 'shape'),
), columns=['tactile_sample'])
```

Pass the wrapper as `extractor={'tactile': wrapper}` to `BrainScoreModel`; configure input preparation in the wrapper. The column is now recognized
without changing the built-in mapping. Register channel **families**; addressed
instances use `family:address` and require an addressing policy. Duplicate channel,
column and capability registrations require explicit replacement. Stock legacy
`io_catalog.register` retains its historical replacement behavior; new plugins
should use `register_channel`.

For new operations subclass `Capability` and implement `handles`/`process`.
Use `model.capability_config` for configuration and
`model.capability_state(self.identifier)` for per-model state. Declare
`input_channels`/`output_channels`; enable the capability only on configured
models. Return a registered `StreamEvent` for extension outputs. StreamEvent
payloads are validated in both directions. The wrapper and its preparation callable must validate/materialize
their own StimulusSet cells; column registration alone does not validate files.

Implement `setup` and `reset` when allocating state or resources. Reset is followed
by setup on the next execution. `supports_session(model, channels)` accepts a
complete output combination; `interact(model, session)` executes it. Multiple
handlers accepting the same combination are rejected. Catalog declarations alone
do not implement execution. See the working `response_trace.py` implementation
and `core/tests/test_external_extensions.py` for complete examples.

## Cache behavior for custom tools

Custom extraction providers use `cache_config()` to describe state that affects their outputs. If that state cannot be represented, disable result caching for the experiment. Make weight edits through `.data` aliases or NumPy **between scoring runs**; these edits are not detected during a run. See [activation caching](caching.md) for supported wrappers and limits.

## Choose a recording or intervention interface

| Interface | What it does | When to use it |
| --- | --- | --- |
| `start_recording` + `process` | Requests benchmark-facing activity, with the subject's region mapping and assembly format | Benchmarks using `BrainScoreModel` or a compatible adapter |
| `observe` | Observes calls, returned values, timing and failures | Add a recorder or observer to an existing evaluator |
| `ActivationWindow` / `PerceptWindow` | Captures internal outputs / input tensors on accessible modules | Direct control of a recording context |
| `RecordActivity` | Uses instrumentation to emit activity into an experiment record | Combine recording with other `Experiment` tools; requires `RecordInputsOutputs` |
| `process(StateChange)` | Applies or removes a supported change through the subject | Explicit intervention control; caller owns cleanup |
| `intervene` | Applies a `StateChange` and removes its handle on context exit | Scoped intervention around an existing evaluator |
| `Ablate` | Silences selected units for chosen conditions/trials through instrumentation | Automatic attachment and cleanup within an `Experiment` |

These operations share components, but record different things. `RecordActivity` uses `ActivationWindow` through `TorchInstrumentation`; other backends supply their own provider. An observation of a model call does not expose hidden internal activity.

## Internal measurements and interventions

Use `ActivationWindow(network, layers=['layer.path'])` around the experiment to
capture actual forward-hook outputs; `PerceptWindow` captures model input tensors.
Closed API models cannot expose these internals unless their provider explicitly
supports them. Attach to an accessible module, not an invented neural channel.

```python
from brainscore.instrumentation import intervene
from brainscore_core.events import StateChange, Selection, Perturbation

change = StateChange('ablation', Selection('layer.path'), Perturbation('zero'))
with intervene(model, change) as applied:
    perturbed_score = benchmark(model)
```

The model must already implement the relevant `intervention`. The context
removes only its own handle on success or failure. Record baseline and perturbed
conditions separately, including selector, intervention, model revision and
protocol. Do not infer a scientific causal result from the fact that a hook ran.

## Runnable robotics example

Run `python examples/partner_integration.py --out /tmp/new-umi-demo` from this
repository after installing its coordinated package set. It uses a small local
PyTorch policy and synthetic DROID-shaped observations. It records actions and
layer activations, adds an independent latency observer, then analyzes saved
outputs. `summary.json` explicitly identifies synthetic data.

For real data, replace the episode with a staged RLDS/TFDS episode and replace
the policy. `staged_droid_episodes(path)` never downloads data. Select demonstration
actions explicitly with `action_source`; verify the exact action field ordering,
units, normalization and coordinate frame for the checkpoint and dataset version.
The [official DROID schema](https://droid-dataset.github.io/droid/the-droid-dataset.html)
includes camera images, robot state and action fields; do not infer physical
semantics from an unnamed seven-element vector.

Supply true observation timestamps with `timestamp_key` in milliseconds when
available. Otherwise the declared period is used and marked inferred. The adapter
withholds demonstration actions and rewards from the policy. It validates camera
arrays, proprioception, finite actions, bounds and episode boundaries. Inference
latency/deadline misses are diagnostic observations, not real-time guarantees.

`evaluate_droid_episode` measures prediction agreement along recorded observations.
It cannot measure task success under model control. For closed-loop environments,
use `EnvironmentSession(environment, require_specs=True)`. The environment wrapper
returns `EnvironmentStep` and declares `observation_spec()` and `action_spec()`.
An optional `action_validator` can add checks. See [environment sessions](environment_sessions.md)
for reset, recording, and timing. Hardware emergency stops, calibrated transforms,
watchdogs, and control frequency belong to the robotics integration and require
qualification on that hardware.

## Contribution acceptance

Run a successful example, malformed input, reset/reuse, failed inference and
cleanup, unsupported output combination, and record/replay test. Assert that
your tool sees unchanged legacy public calls and that a second observer still
works. Record the supported model/data profile. An unfamiliar author's onboarding
trial is a separate release gate; passing tests does not establish usability.
