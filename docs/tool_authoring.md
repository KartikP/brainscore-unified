# Build tools and integrations with UMI

This guide describes the 0.3.0rc1 candidate and its coordinated peer packages.
The public release remains subject to the [release policy](production_release.md).

When interpreting small measured effects, read the [numerical policy](numerical_policy.md).
Record precision, device and execution configuration with your tool's results.
The fixed benchmark parity budget does not establish an error bound for a new
experiment or intervention; compare its effect with matched numerical controls.

## Choose the extension point

| Your contribution | Public entry point | What you own |
| --- | --- | --- |
| Observe model calls | `brainscore.instrumentation.observe` | `on_start`, `on_result`, `on_error` callbacks |
| Record an experiment | `RunRecorder` plus `observe` or `ObservedSession` | Model, data, protocol and configuration provenance |
| Analyze saved measurements | `RunRecord.events`, `outputs`, `evaluate` | Analysis/metric and target data |
| Collect generated responses | `build_trace_subject` | Provider callable and explicit answer parser |
| Measure internal activations | `ActivationWindow` / `PerceptWindow` | Supported model module and selected layers |
| Change model activations | `intervene` and `StateChange` | Model-supported perturbation and condition design |
| Add a sensory domain | `register_channel(..., columns=...)` and a preprocessor | Payload validation, materialization and model extraction |
| Add an output or session | `Capability` with `StreamEvent` | Schema, channel declarations, dispatch and lifecycle |
| Evaluate a robotics policy | `ActionSpec`, `droid_steps`, `evaluate_droid_episode` | Action semantics, policy, data and scientific protocol |
| Drive a controlled environment | `EnvironmentSession` | Environment reset/step, action validation, timing and safety |

Robotics helpers are exported from `brainscore.harnesses.robotics`.

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

Add a `tactile` preprocessor to BrainScoreModel. The column is now recognized
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
payloads are validated in both directions. A preprocessor must validate/materialize
its own StimulusSet cells; column registration alone does not validate files.

Implement `setup` and `reset` when allocating state or resources. Reset is followed
by setup on the next execution. `supports_session(model, channels)` accepts a
complete output combination; `interact(model, session)` executes it. Multiple
handlers accepting the same combination are rejected. Catalog declarations alone
do not implement execution. See the working `response_trace.py` implementation
and `core/tests/test_external_extensions.py` for complete examples.

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

The model must already implement the relevant `state_change_fn`. The context
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
use `EnvironmentSession(environment, action_validator=spec.validate)`, with your
environment returning EnvironmentStep and accepting validated actions. Hardware
emergency stops, calibrated transforms, watchdogs and control frequency belong to
the robotics harness and require qualification on that hardware.

## Contribution acceptance

Run a successful example, malformed input, reset/reuse, failed inference and
cleanup, unsupported output combination, and record/replay test. Assert that
your tool sees unchanged legacy public calls and that a second observer still
works. Record the supported model/data profile. An unfamiliar author's onboarding
trial is a separate release gate; passing tests does not establish usability.
