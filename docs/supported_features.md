# UMI features and environments

UMI connects models to benchmarks and experiments through a shared subject interface. Use it to score models, record their responses, inspect internal activity, intervene, and analyze or replay a run.

These features are available in the **v2 source candidate**. Model, data, and environment combinations have different validation coverage; general-release commitments are defined in the [release policy](production_release.md).

## Capabilities by domain

The same experiment tools work across domains when the subject exposes the required inputs, outputs, and model access.

| Domain | Record inputs and outputs | Record activity / intervene | Evaluate | Example and scope |
| --- | --- | --- | --- | --- |
| Vision | Images and model responses | Selected layers or units in an accessible backend | Brain-response prediction, representation similarity, behavioral tasks | [ResNet-18 tools](../notebooks/18_resnet_experiment_toolbox.ipynb); [measured IT responses](../notebooks/10_brain_alignment.ipynb) |
| Language | Text, model responses, provider-exposed traces | Selected layers or units in an accessible backend | Brain-response prediction, behavioral tasks | [GPT-2 / LeBel](../notebooks/16_whole_brain_encoding.ipynb): measured full-cortex predictions |
| Vision-language | Image/text inputs and responses | Accessible visual or language components | Neural and behavioral benchmarks with compatible model mappings | [CLIP registration](../brainscore/models/clip_vit_b_32/model.py); shared-tool integration also has a synthetic example |
| Audio / video | Audio clips, frames, temporal inputs and responses | Accessible layers; time-resolved extraction where the wrapper provides it | Registered naturalistic fMRI benchmarks | [Wrapper selection](getting_started.md#choose-an-extraction-wrapper); real-data runs need model-specific dependencies and datasets |
| Robotics | Observations, instructions, state, actions and action chunks | Local, accessible policy backend; remote internals require server support | Recorded-action comparisons; external simulator task success | [DROID](droid_integration.md): untrained-policy transport/tool demonstration. [LIBERO](robotics_benchmark_integration.md): trained-policy smoke comparison |
| Recurrent neural/behavioral models | Sequential inputs, activity and behavioral outputs | Accessible components, with explicit state/reset behavior | User-defined protocols and metrics | [Recurrent example](../examples/experiment_toolbox/run.py): synthetic integration, not a validated whole-brain model |

**Backend access** means the tool can attach to the underlying model. Internal recording and ablation use `TorchInstrumentation`, or the OpenPI/JAX providers below. Other backends need a provider. A remote model's input/output access does not provide access to its layers.

## Run and score

| What you can do | Entry point | Boundary |
| --- | --- | --- |
| Score models | `brainscore.score` | Registered identifiers or model/benchmark objects; compatible inputs/outputs |
| Reuse cached activations | [Content checks](caching.md) | Standard vision, text, VLM, audio, and unified video wrappers check weights, settings, and input contents; excludes vision's separate temporal extractor |
| Check storage and data early | `score()`, `python -m brainscore.doctor` | Scoring checks cache storage before loading benchmarks/models; unified also checks declared local assets. Lazy or remote failures can occur later |
| Map activity to brain regions | `region_layer_map`, recording helpers | One layer, composite selections, or all layers; the benchmark determines fitting/scoring |
| Reuse legacy models | `look_at`, `digest_text`, adapters | Existing vision/language task and recording workflows |
| Assemble an experiment | `Experiment` | Combines a subject, one protocol, tools, and an output folder |
| Choose the procedure | `SessionProtocol`, `CallableProtocol` | Sessions manage conditions/trials; an existing evaluator keeps its own procedure |
| Control delivery | Sessions and streaming helpers | Batches, individual inputs, windows, and clock policies; no hard real-time guarantee |

## Ready-to-use experiment tools

Pass these in `Experiment(tools=[...])`.

| Tool | What it does | Requires |
| --- | --- | --- |
| `RecordInputsOutputs` | Save inputs, outputs, activity, errors, and lifecycle events | Supported payloads |
| `RecordActivity` | Capture selected internal activity before/after intervention | `RecordInputsOutputs` and instrumentation |
| `Ablate` | Temporarily zero selected outputs, optionally in selected conditions/trials | Instrumentation |
| `ScaleActivity` | Multiply selected outputs by a fixed factor, optionally in selected conditions/trials | `TorchInstrumentation`; weights stay unchanged |
| `ObserveCalls` | Include an existing call observer in the experiment | Observer callbacks and selected subject methods |
| `RecordReasoning` | Save exposed CoT, summaries, and streamed fragments with their original response | `RecordInputsOutputs` and explicit reasoning fields or an extractor; [guide](reasoning_recording.md) |

## Adapters and building blocks

Use these to connect a model or build a tool. They also underpin the tools above.

| Component | Responsibility | Relationship to experiment tools |
| --- | --- | --- |
| `TorchInstrumentation` | Access the actual PyTorch network's named layers | Supplies internal recording/ablation; [policy backends](policy_instrumentation.md) |
| `OpenPIInstrumentation` | Record/ablate three sites in the pinned Pi0/Pi0.5 JAX sampler | Selected units and denoising iterations; [trained L4 and ten-task smoke checks](qualification/2026-10-06-openpi-tools.md) |
| `RemoteOpenPIInstrumentation` | Use the same tools through `OpenPIToolServer` and `OpenPIPolicyClient` | Request-scoped cleanup and separate activity delivery; [guide](policy_instrumentation.md) |
| `observe` | Notify callbacks about method starts, results, and errors | Used by `ObserveCalls`; does not save anything by itself |
| `ActivationWindow` / `intervene` | Scoped activity capture / intervention | The direct mechanisms behind activity and intervention workflows |
| `RunRecorder` | Write records, arrays, files, and metadata | Storage used by recording tools |
| `RunRecord` | Read saved measurements and apply metrics | Shared reader; no model execution |
| `build_trace_subject` | Adapt a provider and answer parser into a subject | Produces raw responses, streamed fragments, and parsed answers for recorders |
| `replay_sessions` / `replay_calls` | Send saved inputs through a model again | Explicit new execution; does not regenerate simulator feedback |
| `context.import_artifact` | Copy external videos/reports with producer metadata | Keeps evaluator-produced artifacts with the experiment |

`Selection` uses the same layer paths across recording and intervention tools. For PyTorch interventions, unit indices address the **last output axis**; on a convolutional output that is usually image width, not channels.

## Guides, tutorials and examples

“Guide” explains the workflow. “Notebook” walks through it with concise code comments and figures. “Example” is code to run or adapt. A dash means there is no dedicated notebook linked here.

| Workflow | Guide / reference | Commented notebook | Runnable example |
| --- | --- | --- | --- |
| Connect and score a model | [Getting started](getting_started.md), [API](umi_api_reference.md) | [01: layer mapping](../notebooks/01_quickstart_layer_mapping.ipynb), [07: your model](../notebooks/07_bring_your_model.ipynb) | [Model template](../templates/new_model) |
| Use existing Brain-Score models | [Legacy integration](from_brain_score.md) | [10: brain alignment](../notebooks/10_brain_alignment.ipynb) | [Guide’s scoring example](from_brain_score.md#score-an-existing-model) |
| Record, ablate, restore and replay | [Experiment toolbox](experiment_toolbox.md) | [17: digits](../notebooks/17_experiment_toolbox.ipynb), [18: ResNet-18](../notebooks/18_resnet_experiment_toolbox.ipynb) | [Five-domain tool examples](../examples/experiment_toolbox/README.md) |
| Work directly with internal activity | [Tool authoring](tool_authoring.md#internal-measurements-and-interventions) | [04: several regions](../notebooks/04_multiregion_geometry.ipynb), [11: ablation](../notebooks/11_state_change_ablation.ipynb), [14: interventions](../notebooks/14_intervention_spectrum.ipynb) | [Experiment tools](../examples/experiment_toolbox/run.py) |
| Record generated reasoning | [Reasoning recorder](reasoning_recording.md) | — | [Streamed response example](../examples/experiment_toolbox/reasoning.py) |
| Handle time and streaming | [Streaming API](umi_api_reference.md#streaming-helpers) | [13: temporal alignment](../notebooks/13_temporal_multimodal.ipynb), [15: delivery](../notebooks/15_streaming_delivery.ipynb) | [Session examples](../examples/experiment_toolbox/run.py) |
| Display brain measurements | [Local data](local_data.md) | [08: illustrative maps](../notebooks/08_brain_visualization.ipynb), [16: measured predictions](../notebooks/16_whole_brain_encoding.ipynb) | [Measured figure scripts](../notebooks/figure_sources/README.md) |
| Inspect an OpenPI policy | [Policy tools](policy_instrumentation.md) | [19: trained-policy measurements](../notebooks/19_openpi_experiment_toolbox.ipynb) | [Server](../examples/libero/serve_tools.py), [qualification](../examples/libero/qualify_tools.py) |
| Run a native environment session | [Environment sessions](environment_sessions.md): spaces, reset, clocks, execution records | — | Grid and rendered discrete Gymnasium environments |
| Connect a robotics evaluator | [DROID](droid_integration.md), [LIBERO](robotics_benchmark_integration.md) | — | [LIBERO bridge and evaluator](../examples/libero/README.md) |
| Add a tool or domain | [Tool authoring](tool_authoring.md), [extension guide](../EXTENDING.md) | — | [External package](../examples/partner_tool/README.md), [custom experiment tool](../examples/experiment_toolbox/partner_tool.py) |

See the [notebook index](../notebooks/README.md) for downloads, compute requirements, and whether each example uses synthetic, measured, or saved results.

## Add features from your own package

| Extension | Implement | What stays shared |
| --- | --- | --- |
| Input domain | Channel schema, optional stimulus-column mapping, and preprocessor | Subject, sessions, recording and scoring interfaces |
| Output or operation | Output channel and a registered `Capability` | Dispatch and event validation |
| Session or experiment | Session/protocol; capability session handling when using capability dispatch | Subject lifecycle and experiment tools |
| Research tool | `Tool`, call observer, or instrumentation provider | Tool attachment, cleanup, events and artifacts |
| Benchmark or metric | Evaluation callable; optional registry entry | Model interface and result handling |
| Robotics policy | Observation/action adapter, action semantics and reset behavior | Model calls, records and analysis |

Import your registration module before constructing the model. These paths do not require editing the `Subject` contract or Brain-Score's dispatch code. You still implement domain-specific execution and test its semantics; registration alone does not supply them.

## Environments

Install the four packages together using the [peer revisions and constraints](../install/README.md). Python **3.11** is required. Package dependency ranges do not establish that every permitted version has been tested.

| Environment | Current coverage | Boundary |
| --- | --- | --- |
| Ubuntu 24.04 x86_64, CPU | Coordinated installation and offline integration CI | Scientific results need their own model/data profile |
| macOS 14 ARM64, CPU | Coordinated installation and offline integration CI | Other macOS versions need separate evidence |
| Ubuntu 22.04, NVIDIA L4, CUDA 13.0, FP32 | Recorded scientific candidate comparisons | Historical profile; final-artifact requalification remains required |
| macOS, Apple MPS | Selected measured runs, including GPT-2/LeBel | Experimental backend; one result is not general numerical qualification |
| Robotics policy servers | Integration-specific environments, separated from UMI when necessary | Follow the pinned evaluator/policy instructions |
| Windows, Intel macOS, ROCm, Linux ARM64, other CUDA stacks | No general qualified profile | Python versions outside 3.11 are rejected by package metadata |

Use the [numerical policy](numerical_policy.md) for exactness and error budgets. Use separate subject/tool instances for concurrent experiments; the experiment runner is synchronous. Custom object serialization, physical-robot safety, distributed scheduling, a replay UI and automatic circuit discovery are not built-in guarantees.

[Release requirements](production_release.md) · [Qualification records](qualification/) · [Shared vocabulary](conventions.md)
