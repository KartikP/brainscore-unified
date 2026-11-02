# Load and compose tools

Load a fresh tool by identifier, then attach it directly or pass it to an
`Experiment`. Both paths use the same tool implementation, validation, ordering,
and cleanup. `Experiment` owns the run; its protocol owns the procedure. Neither
is itself a tool.

```python
from brainscore.tools import list_tools, load_tool

print(list_tools())
lesion = load_tool("ablate", targets=["language"], name="language_lesion")
```

| Identifier | Implementation | Configuration |
| --- | --- | --- |
| `ablate` | Existing `Ablate` tool | `targets`, optional `conditions`, `trials`, `name` |
| `linear-probe` | Fitted logistic readout | `target` and fitting settings, or `artifact`; [guide](linear_probe.md) |
| `record-io` | Existing `RecordInputsOutputs` event writer | Optional `max_bytes` |

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
        load_tool("ablate", targets=[Selection("language", [0, 3])],
                  conditions=["lesioned"], trials=[1], name="language_lesion"),
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
