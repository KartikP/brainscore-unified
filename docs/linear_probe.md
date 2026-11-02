# Fit and attach a linear probe

A linear probe measures which labels can be decoded from activity. It does not
change the model, and high accuracy alone does not establish that the model uses
the decoded information. Select discovery and evaluation examples separately.

This tool lives in `unified`. It uses the existing experiment instrumentation,
layer selections, assemblies, records, and cleanup. It adds no subject methods
or channels and does not change the existing behavioral readout.

For a self-contained CPU example with a fixed synthetic text encoder:

```bash
python examples/linear_probe.py --out /tmp/new-linear-probe-demo
```

The example fits a color probe, saves and reloads it, attaches two instances
around an ablation, then reuses it in a direct scope and checks restoration.
Its phrases and encoder are synthetic; it demonstrates integration only.

## Fit, save, and reload

```python
from brainscore.tools import load_tool

probe = load_tool("linear-probe", target="language", name="gender")
probe = probe.fit(
    discovery_activity,
    labels=discovery_labels,
    provenance={"checkpoint": "exact model revision", "dataset": "discovery-v1"},
)
probe.save("gender-probe.json")

loaded = load_tool("linear-probe", artifact="gender-probe.json", name="gender")
probabilities = loaded.predict(evaluation_activity)
```

`fit` returns a new fitted tool; the original learner is unchanged. Activity must
have dimensions `(presentation, neuroid)`, a `layer` coordinate matching the
target, and original integer unit addresses in `unit_index`,
`neuron_number_in_layer`, or `neuroid_num`. MultiIndex coordinates are supported.
An explicit subset `Selection` must match these addresses in order.

The method standardizes discovery features and fits scikit-learn logistic
regression. `classifier_c` is its inverse regularization parameter (default 1).
Evaluation uses the saved scaler, coefficients, intercepts and class order;
it never refits. Labels are nonempty strings and must include at least two classes.
Prediction returns a `BehavioralAssembly` and retains presentation coordinates.

Artifacts are versioned JSON with a checksum, not pickle. They contain the
target and unit coordinates, fitting settings, parameters, scikit-learn version,
fitting-example count and a fingerprint of discovery features, labels and unit
addresses. Supply checkpoint and dataset provenance explicitly. Unit/layout
checks do not establish checkpoint identity; connect the same model revision
when reusing the artifact. Existing files are never overwritten.

## Use several tools in a protocol

Bind instrumentation to the actual network once, using its named modules or an
explicit mapping from logical population names to modules. This avoids private
model inspection. Reuse that provider for subsequent experiments.

```python
from brainscore.experiments import (
    Ablate, Experiment, RecordInputsOutputs, TorchInstrumentation,
)
from brainscore_core.events import Selection

instrumentation = TorchInstrumentation({"language": language_module})
before = load_tool("linear-probe", artifact="gender-probe.json",
                   name="gender_before", when="before")
after = load_tool("linear-probe", artifact="gender-probe.json",
                  name="gender_after", when="after")

result = Experiment(
    subject=subject,
    protocol=protocol,
    tools=[
        RecordInputsOutputs(),
        before,
        Ablate([Selection("language", [0])], conditions=["lesioned"]),
        after,
    ],
    instrumentation=instrumentation,
    output_dir="runs/probe-comparison",
).run()

for event in result.record.events():
    if event["kind"] == "probe":
        print(event["source"], event["condition"], event["payload"])
```

The supplied protocol defines its conditions and trial boundaries. List observers
first, then before-intervention tools, interventions, and after-intervention tools.
Within each phase, list order is attachment order. Names must be unique, including
when attaching two instances of the same method. Tools detach in reverse order.
All tools validate before the run creates files or attaches hooks.

Each probe publishes `probe_artifact` and `probe` events. `RecordInputsOutputs`
saves the actual fitted parameters and predictions, not only their hash. The
existing condition, trial, source and event identifiers provide the context for
these measurements. Probes do not accumulate a second in-memory history.

## Attach around an existing procedure

```python
from brainscore.tools import attach_tools

with attach_tools(
    subject,
    tools=[RecordInputsOutputs(), after],
    instrumentation=instrumentation,
    output_dir="runs/direct-probe",
):
    responses = behavioral_response(subject, task)
```

This scope uses the experiment runner's validation, records and cleanup. It
does not own model execution or resets and does not intercept session inputs or
outputs. It records emitted tool measurements and lifecycle events. Use
`SessionProtocol` for recordings that also capture session inputs/outputs and
support input replay. A benchmark can use this scope around its own sessions
without changing its scoring or reset rules.

## Position and backend boundaries

Prediction on supplied assemblies supports leading position dimensions. Hooked
predictions classify each batch/token/spatial position independently; there is
no implicit averaging, last-token selection, or padding mask. Hook records name
unidentified leading axes `position_0`, `position_1`, etc.; these are tensor
indices, not token identities or physical time. For a pooled or masked analysis,
record activity, explicitly transform it, and call `predict` offline.

`Selection.axis` and `expected_shape` retain their existing meanings. A channel
selection on a CNN and a flattened spatial selection are different populations.
The instrumentation provider must support `record`; access to model outputs
alone is insufficient. Fitting requires scikit-learn; applying a fitted probe
uses NumPy/SciPy. Hooks reuse the provider's snapshots and do not modify tensors,
parameters or parameter gradients.

The selected site must execute at least once per attached trial/scope; otherwise
the run fails rather than reporting an unmeasured probe as successful. Cached
responses do not run model hooks. Use an uncached forward path when recording
new probe measurements; this tool does not change global cache settings.

Tests cover binary/multiclass predictions against an independent scikit-learn
reference on held-out examples, artifact integrity, original unit coordinates,
position axes, two probes alongside recording/ablation, and cleanup after
failure or interruption. These are fixture checks, not real-model scientific
qualification. No other new interpretation methods are included in this change.
