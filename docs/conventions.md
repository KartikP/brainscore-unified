# UMI vocabulary and example style

Use these terms across code, guides, and notebooks. Explain a term once, then use the same name.

| Term | Meaning |
| --- | --- |
| Model | The network or service performing the computation. |
| Subject | The interface through which an experiment or benchmark uses a model. `BrainScoreModel` is a configurable implementation. |
| Adapter | Connects an existing interface to another; name the interfaces, such as `VisionModelAdapter` bridging a vision plugin to UMI. |
| Environment harness | Connects an external environment's observations, actions and lifecycle to the subject. |
| Preprocessor | Prepares an input in the format a model expects. |
| Wrapper | Adapts a model or extractor to an expected interface. It can include preprocessing. |
| Activations | Internal values produced by the model. “Activity” is the plain-language name for these measurements. |
| Layer path | The model's own name for a component, such as `layer3.0.bn2`. |
| Selection | A layer path and optional unit indices. For PyTorch interventions, indices address the last output axis. |
| Region mapping | A proposed correspondence between brain regions and model layers. A mapping alone is not evidence of biological similarity. |
| Assembly | An array with labels describing its samples, units, and other axes. |
| Neuroid | An entry on a recorded-unit axis: a model feature or biological recording site, depending on the data. |
| Modality | An input domain, such as vision, text, or audio; channel names also describe outputs. |
| Channel | A named kind of input or output, with a documented payload schema. |
| Event | One timestamped input or output. Experiment logs also include activity and lifecycle events. |
| Session | The exchange of inputs and outputs. `None` from `next_input()` means it has ended. |
| Protocol | The procedure that supplies inputs and controls the experiment. |
| Condition | A setting being compared, such as normal or silenced. |
| Trial | One repetition within a condition. |
| Experiment | One execution combining a subject, protocol, tools, and output location. |
| Tool | A component that observes a run or changes selected model behavior. |
| Instrumentation | The backend connection that lets tools record or change internal model activity; `TorchInstrumentation` supplies it for PyTorch. |
| Reasoning trace | Exposed model-generated reasoning text. Preserve whether it is a complete block, a streamed delta, a cumulative snapshot, or a provider summary. It is not automatically a faithful explanation of internal computation. |
| Benchmark | A defined evaluation procedure that returns a score. Not every experiment is a benchmark. |
| RunRecord | The shared reader for saved measurements. Reading a record does not run the model. |
| Replay | Sending saved inputs through a model again. This does not regenerate an environment's feedback. |

## One learning path

- Use `subject` for the UMI-facing object and `model` or `network` for the underlying network when both appear together.
- Use real layer paths in examples. Named-module mappings remain supported when an integration needs them.
- Use `conditions` for comparisons and `trials` for repetitions. `Ablate` can filter either.
- Use `RunRecord` or `result.record` to read measurements. `read_events` remains a convenience for complete experiment records.
- Prefer `Experiment` when assembling several tools. Teach direct `observe`, `ActivationWindow`, and `intervene` when the example needs that level of control; explain their relationship to the tools.
- Preserve public entry points and compatibility. Do not rename working APIs only to make their spelling match.

Use “silence” for setting selected activity to zero and “intervention” for the broader category of changes. Keep API names such as `Ablate`, `StateChange`, and `Perturbation`. Explain `drive` as adding a value. Preserve meaningful controls such as recording-only, and retain condition names in saved records.

Prefer “activity” in introductory prose. Use “features” for values supplied to an analysis and “representations” for patterns being compared; explain the distinction where needed. Use “v2 source candidate” for the current release status. Keep exact version numbers where installation or evidence requires them.

## Notebook structure

Start with the question, model/data, requirements, and evidence limits. Then use short numbered steps: prepare, connect, run, inspect, and optionally replay. Keep each cell to one task. Add a concise comment explaining every Brain-Score operation and unfamiliar Python helper.

Use ordinary Python and explicit variables. Avoid compressed imports, semicolon-separated statements, blanket warning suppression, and helpers used only to hide a few lines. Put advanced methodology after the main walkthrough, while retaining information needed to interpret the result.

Figures need a descriptive title, labeled axes and units, and a legend or colorbar where needed. Use blue for baseline/model, orange for intervention, green for restored, and gray for controls. Use shared color scales when comparing panels; label any deliberate scale differences. Identify synthetic, measured, and previously saved results explicitly.

State what a result supports. A demonstration is not a qualification; a selected example is not an aggregate result; an output change is not evidence about biological causality. Avoid promising a direction of effect before measuring it.
