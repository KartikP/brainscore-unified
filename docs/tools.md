# Load and compose tools

Load a fresh tool by identifier, then attach it directly or pass it to an
`Experiment`. Both paths use the same tool implementation, validation, ordering,
and cleanup. `Experiment` owns the run; its protocol owns the procedure. Neither
is itself a tool.

See [instrumentation, tools and experiments](tools_and_experiments.md) for a
short explanation of their responsibilities and a diagram.

[Pretrained-model qualification](qualification/2026-11-02-interpretability-tools.md)
covers GPT-2 and CLIP text on CPU float32, with independent numerical references,
multiple tools and exact baseline restoration. Scientific causal claims require
their own controls and evidence.

```python
from brainscore.tools import list_tools, load_tool

print(list_tools())
lesion = load_tool("ablate", targets=["language"], name="language_lesion")
```

| Identifier | Implementation | Configuration |
| --- | --- | --- |
| `ablate` | Existing `Ablate` tool | `targets`, optional `conditions`, `trials`, `name` |
| `contrastive-steering` | Fitted positive-minus-negative activity direction | `target`, optional `strength`, or `artifact`; [guide](contrastive_steering.md) |
| `linear-probe` | Fitted logistic readout | `target` and fitting settings, or `artifact`; [guide](linear_probe.md) |
| `leace` | Fitted categorical linear concept erasure | `target` and fitting settings, or `artifact`; [guide](leace.md) |
| `observe-calls` | Existing `ObserveCalls` adapter | `observer`, optional `methods`, `name` |
| `patch-activity` | Existing `PatchActivity` replacement | `targets`, `reference`, optional `conditions`, `trials`, `name` |
| `record-activity` | Existing `RecordActivity` internal recorder | `targets`, optional `when` (`before`/`after`) and `name` |
| `record-io` | Existing `RecordInputsOutputs` event writer | Optional `max_bytes` |
| `record-reasoning` | Existing `RecordReasoning` exposed-text recorder | Optional `extract`, `name`; requires `record-io` |
| `scale-activity` | Existing `ScaleActivity` intervention | `targets`, `factor`, optional `conditions`, `trials`, `name` |

The registry is explicit. An external package may call
`register_tool(identifier, factory)` without changing the runner. Import its
registration module first; replacing an existing identifier requires
`replace=True`. Loading tools does not automatically download models or fit
anything on evaluation data.

## Ablation in a protocol

```python
from brainscore.experiments import Experiment, TorchInstrumentation
from brainscore_core.events import Selection

instrumentation = TorchInstrumentation({"language": language_module})
result = Experiment(
    subject=subject,
    protocol=protocol,
    tools=[
        load_tool("record-io"),
        load_tool("record-activity", targets=["language"], name="before", when="before"),
        load_tool("ablate", targets=[Selection("language", [0, 3])],
                  conditions=["lesioned"], trials=[1], name="language_lesion"),
        load_tool("record-activity", targets=["language"], name="after"),
    ],
    instrumentation=instrumentation,
    output_dir="runs/loaded-ablation",
).run()
```

The supplied protocol must declare the condition `lesioned` and trial `1`.
This intervention zeros units 0 and 3 only in that condition and trial. Omit
`conditions` or `trials` to apply it to all corresponding entries. Omitting
unit indices zeros the whole selected output. `Selection.axis` and
`expected_shape` keep their existing coordinate meanings.

Loading `ablate` constructs the existing `Ablate`, not a second implementation.
Existing `Ablate(...)` calls retain their defaults and behavior. The optional
`name` distinguishes multiple instances, including disjoint unit selections
within the same population. Overlapping ablations also zero the overlap;
they do not double the intervention dose.

## Direct attachment and composition

```python
from brainscore.tools import attach_tools

with attach_tools(
    subject,
    tools=[
        load_tool("ablate", targets=[Selection("language", [0])], name="first"),
        load_tool("ablate", targets=[Selection("language", [3])], name="second"),
    ],
    instrumentation=instrumentation,
    output_dir="runs/two-ablations",
):
    responses = behavioral_response(subject, task)
```

A direct scope has condition `default` and trial `direct`; leave filters unset
or use those values. The caller owns execution and resets. Attachments are
removed in reverse order, including on failure or interruption, and pre-existing
hooks remain installed. Use `load_tool("record-io")` to save tool events; direct
attachment does not itself intercept session inputs or outputs.

Each collection needs unique instance names. List observers first, then
before-intervention measurements, interventions, and after-intervention
measurements. Within a phase, list order determines attachment order. A loaded
ablation can sit between two [named probes](linear_probe.md) or activity
recorders. The entire collection validates before any hooks attach.

Instrumentation must expose the actual named sites and support ablation. For
benchmark use, map functional population names to model sites explicitly; do
not add model-specific layer paths to a benchmark. Hooks require real forward
passes; cached responses do not execute interventions. Existing backend and
cache restrictions still apply. See the [experiment guide](experiment_toolbox.md).

## Event recording

`load_tool("record-io")` constructs the existing event writer. Put one writer
first in the tool list; it is the shared sink for session events, measurements
from other tools, and lifecycle/failure events. It retains the name
`inputs_outputs` and the standard `inputs_outputs/` directory, so existing
`RunRecord`, `result.record`, and session-replay readers work unchanged.

Use `max_bytes` to bound the event log and its stored arrays/files. Exceeding
the budget fails the run and leaves an explicitly failed record. Omitting
the budget retains existing behavior. Other experiment files are outside
this budget. The existing writer's serialization and memory limits still apply.

`SessionProtocol` supplies session inputs/outputs to the recorder.
`CallableProtocol` supplies the public calls selected by its evaluator.
`attach_tools` supplies tool measurements and lifecycle events; it does not
automatically observe calls inside its body. Loading the writer does not grant
access to hidden activity or change these recording boundaries.

## Internal activity recording

`load_tool("record-activity", targets=[...])` constructs `RecordActivity` and
requires both instrumentation supporting `record` and a `record-io` writer in
the same collection. Loading a tool does not discover or unwrap private models.
Bind instrumentation explicitly to the network or named population mapping.

Use distinct names for multiple recorders and `when="before"` or `"after"` to
choose their attachment phase. These phases describe activity before/after the
interventions in this collection, not an extra model execution. Both can record
the same site in one forward pass. Existing `Selection` coordinates, axes and
expected shapes are preserved; no pooling or token alignment is added.

Measurements are `activity` events in the shared record. Each retains its tool
name in `source`, condition/trial context, target, position relative to the
intervention, values, dtype/device and unit-selection coordinates. The recorder
does not alter tensors or replace subject outputs. Hooks detach on normal exit,
failure or interruption. Existing model hooks remain installed.

For a runnable CPU fixture composing loaded recording, ablation and fitted probes:

```bash
python examples/linear_probe.py --out /tmp/new-loaded-tools-demo
```

This demonstrates tool integration on a fixed synthetic encoder; it is not a
real-model qualification or evidence of a biological mechanism.

## Scale activity

```python
scale = load_tool("scale-activity", targets=[Selection("language", [0, 3])],
                  factor=0.5, conditions=["damped"], name="damp_language")
```

Pass this tool between before/after measurements in an `Experiment`, or inside
`attach_tools`. It multiplies selected activity during real forward passes;
it does not modify trained weights. Conditions/trials and `Selection` coordinates
have the same meanings as for ablation. The protocol must declare any filtered
condition or trial.

Named instances may affect different or overlapping selections. Scales on the
same units multiply in list order, so two factors of 0.5 give a factor of 0.25.
Zero and negative factors are allowed; nonfinite factors fail during loading.
The backend must support `scale` (currently `TorchInstrumentation`); recording
access alone does not imply scaling support. Cleanup removes only these hooks.

## Patch activity

```python
patch = load_tool("patch-activity", targets=[Selection("language", [0, 3])],
                  reference=analysis["reference"], name="restore_language")
```

`reference` is the saved donor snapshot returned by
`TorchInstrumentation.attribute`; see [attribution and patching](attribution_patching.md).
The loaded tool owns a copy, saves the actual reference in the run, and checks
coordinates, shapes and dtypes. It does not align tokens automatically.

Several named patches can affect different selections. On overlapping units,
later patches replace earlier values. Scaling followed by patching differs
from patching followed by scaling: list order is explicit and preserved.
Each active patch target must execute exactly once per trial/direct scope;
missing or repeated calls fail and detach hooks. Autoregressive generation loops
remain unsupported by this exact-patch scope. The backend must support `patch`
(currently local PyTorch); this loader does not extend backend coverage.

## Observe public calls

```python
observer_tool = load_tool("observe-calls", observer=my_observer,
                          methods=["interact"], name="call_counts")
```

An observer implements any of `on_start(call)`, `on_result(call, result)` and
`on_error(call, error)`. Use `methods=["interact"]` for a native subject, or
explicit public methods exposed by an existing legacy evaluator. The default
remains `("process",)` for compatibility; it is not required by `Subject`.
There is no automatic method or capability discovery.

Put observers before activity and intervention tools. Multiple named observers
receive independent notifications. Nested selected methods count once per
observer, and exact prior instance attributes are restored after the scope.
Observer failures propagate. Call arguments are live objects; callbacks that
retain them must take their own snapshots. This adapter does not serialize
callback results or turn arbitrary call arguments into session events.

## Record exposed reasoning

```python
reasoning = load_tool("record-reasoning", name="reasoning")
summaries = load_tool("record-reasoning", name="summaries", extract=extract_summary)
```

Put these observer tools after the shared `record-io` writer. The default reads
the explicit reasoning field of `response_trace`; a custom `extract(output)`
adapts another exposed format and returns text, text blocks, or `None`. Answer
text is not automatically labeled reasoning, and hidden reasoning is not recovered.

Each named instance writes its own directory, readable with
`RunRecord(result.directory / name)`, linked to the shared inputs and original
responses. Names must be single, unreserved directory components. Default naming,
record schemas, unavailable-versus-empty distinctions, and streamed
delta/snapshot formats remain unchanged. Failed extraction fails the run and
leaves failed records for diagnosis. `record-io`'s byte budget does not cap these
separate reasoning records. See [reasoning recording](reasoning_recording.md).
