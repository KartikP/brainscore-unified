# Glossary

Short definitions of UMI terms. For how the pieces fit together, see [Concepts](concepts.md); for naming and example style, see [Conventions](conventions.md).

## Subjects and models

| Term | Meaning |
| --- | --- |
| **Model** | Any system that computes responses, such as a trained network, a robot policy or a remote service. A registration loads it; for your own model, you supply it. |
| **`Subject`** | The interface an experiment talks to: `identifier`, `in_channels`, `out_channels`, optional `required_channels`, one evaluation method and `reset()`. The interface alone supplies no checkpoint. |
| **`participate(session)`** | The one evaluation method on `Subject`: read inputs, send responses, return when input ends. Passive and closed-loop experiments use it alike. |
| **`reset()`** | Clears run state, such as history or active interventions, between independent runs. It does not retrain the model or erase learned weights. |
| **`UnifiedModel`** | A `Subject` base that runs sessions through typed methods (`process`, `start_recording`, `start_task`). Legacy adapters build on it. |
| **`BrainScoreModel`** | A ready-made `UnifiedModel` assembled from parts that a registration or your code configures, including any network and weights. It also keeps state such as a fitted readout. |
| **`process(input_event)`** | A typed convenience on `UnifiedModel` and `BrainScoreModel`, not part of the `Subject` contract. |
| **Adapter / legacy facade** | An adapter presents an existing vision or language model as a subject; the facade runs any subject on a legacy benchmark. Both are permanent. |
| **`build_*` function** | A factory returning a configured subject or helper, such as `build_trace_subject`. |

## Inputs and outputs

| Term | Meaning |
| --- | --- |
| **Domain** | An area of study, such as vision, language or robotics; not an input type or channel. A robotics study may present images, instructions and joint positions. |
| **Modality** | A kind of sensory input: `vision` (including video), `text` or `audio`. On `BrainScoreModel`, `extractor` keys declare modalities, which become input channels. |
| **Channel** | A named route with a catalog entry, written `family` or `family:address`, such as `text`, `neural:IT` or `motor`. Declaring a channel does not implement it. |
| **`StimulusSet`** | A table with one row per stimulus; `stimulus_id` links each row to its response. |
| **`StreamEvent`** | One item on a channel: `channel`, `payload`, `t_ms` (milliseconds, or `None` without a physical time scale) and `meta`. A **stream** is a sequence of events. |

## Running an evaluation

| Term | Meaning |
| --- | --- |
| **Session** | One run's exchange: `next_input()` returns the next event or `None` once input is exhausted, and `respond(event)` accepts a response. Exhaustion is not success; completion (`session.complete`) and failures are reported separately. |
| **Compatibility** | A pre-flight check from declarations alone. **Full**: the subject takes in every benchmark input; **partial**: it takes in some and the rest never enter it; **N/A**: a requested output or required input is missing, or no input is accepted, so the pair is not run and never scored zero. |
| **Benchmark** | A defined test: stimuli, requested outputs and a metric, returning a `Score`. |
| **Metric** | The calculation comparing model outputs with target data, such as accuracy. |
| **`Score`** | A metric result with metadata. **Raw** is the direct metric value; **ceiling-adjusted** (ceiled) is normalized by the benchmark's ceiling and can exceed 1.0. |
| **Ceiling** | An estimate, from measurement reliability, of how well the target data can be predicted. Not every benchmark has one. |
| **Null baseline** | A reference without the intended signal, such as chance or random features, used to interpret a score. |
| **Protocol** | The procedure an `Experiment` runs. |
| **`SessionProtocol`** | Creates a fresh session for each trial and resets the subject before and after it. |
| **`CallableProtocol`** | Wraps an existing `evaluate(subject, context)`, such as a benchmark or external runner, keeping its loop, resets and scoring. It is an alternative to `SessionProtocol`, not built on it. |
| **Condition** | A setting being compared, such as `baseline` or `silenced`. |
| **`Trial`** | One independent repetition within a condition, with a fresh session. It can contain many stimuli or a whole episode. |
| **`Experiment`** | One execution combining a subject, protocol, tools and a new output directory; it need not produce a score. |

## Parts of `BrainScoreModel`

| Term | Meaning |
| --- | --- |
| **Preprocessor** | Input preparation only, such as resizing images or tokenizing text. Configure it on the extractor. |
| **Extractor** | Records activity for a modality: a callable `(stimuli, layers=...)` returning a `NeuroidAssembly`, owning preparation, the forward pass, hooks and caching. |
| **Wrapper** | An extractor Brain-Score ships, such as `VisionWrapper` or `TextWrapper`. `PolicyWrapper` wraps a policy instead. |
| **`answer` / `policy` / `intervention`** | Callables that produce a behavioral answer, choose an action from an `EnvironmentStep`, or apply a `StateChange` and return its record and cleanup. |
| **`Capability`** | A registered handler inside `BrainScoreModel` that adds a dispatch path and its channels; it is part of the subject. |
| **`TaskContext`** | Task settings such as `instruction`, `label_set` and `fitting_stimuli`. |
| **`label_set`** | The answers a benchmark accepts, passed to `answer`; it does not constrain what a model generates. On the generation path, an out-of-list answer is recorded as invalid, with a warning: its probabilities are NaN, and the `raw_answer` and `answer_valid` coordinates keep the reply. |
| **Readout** | A classifier fitted on recorded features to produce behavior, a different measurement from generation. |

## Recording and measurement

| Term | Meaning |
| --- | --- |
| **Activation (activity)** | A value computed inside a network for one input. Output access alone, as through a typical API, does not reach it; a remote service exposes internals only when instrumented to do so. |
| **Layer path** | The model's own module name, such as `layer3.0.conv1`. |
| **`region_layer_map`** | Which model units to record for a requested brain region, such as `{'IT': 'layer4'}`. It is a hypothesis that benchmarks test. |
| **Selector / `Selection`** | `LayerSelector` and `CompositeSelector` are region-map values. A `Selection` is a concrete layer path, unit indices and axis, reused for recording and intervention. |
| **Assembly** | A labeled array. `NeuroidAssembly` has `presentation`, `neuroid` (recorded units) and optional `time_bin`; `BehavioralAssembly` axes depend on the task, such as `presentation` × `choice` for probabilities or `presentation` alone for reading times. |

A region mapping or an assembly's labels do not establish biological similarity; only a benchmark's comparison with measured data can support that claim.

## Environments

| Term | Meaning |
| --- | --- |
| **Environment** | The external simulator, game or robot, such as Gym or LIBERO, with its own observation and action formats. |
| **Environment wrapper** | The UMI-facing adapter for `EnvironmentSession`; its `reset()` and `step(action)` translate native results into `EnvironmentStep`. An established evaluator can keep its own loop through `CallableProtocol`. |
| **Spec** | An object with `validate(value)` and `describe()`, such as `ArraySpec` or `ActionSpec` (named dimensions, units, frame, bounds, `period_ms`). Validation never clips or casts. |
| **Step / episode** | A step is one observation and action, ordered by `step_num`, not physical time. An episode runs from reset to its end. |
| **Clock** | The environment sets `t_ms` when it has a time scale, otherwise `None`. `latency_ms` is the policy call's wall time. |

## Tools and records

| Term | Meaning |
| --- | --- |
| **Tool** | An object attached to an `Experiment` to observe or change a run. Unlike a capability, it needs no change to the subject. |
| **`RecordInputsOutputs`** | A tool that saves a run's inputs, outputs and lifecycle events. It needs no access to model internals. |
| **`RecordActivity`** | A tool that captures internal activity at named targets through instrumentation; `RecordInputsOutputs` saves what it captures. |
| **Instrumentation** | The backend connection that lets tools reach a model's internals, such as `TorchInstrumentation` for PyTorch. |
| **`OpenPIInstrumentation`** | Records or zeroes three named sites in one pinned OpenPI JAX sampler and refuses other versions. It is not a general JAX hook system; qualification covers only tested configurations. |
| **Recording** | Saving what happened during a run. `RunRecorder` writes; `RunRecord` reads without running the model. |
| **Provenance** | What produced a result: model revision, weights, settings, software versions and procedure. Keep it with responses and scores. |
| **Input replay** | `replay_sessions` or `replay_calls` send saved inputs to a model again, open loop: new outputs do not change saved observations. |
| **Simulation or video replay** | Re-executing saved actions in a simulator, or rendering saved frames; UMI's replay functions do neither. |
| **Reasoning record** | `RecordReasoning` saves only reasoning text a provider explicitly exposes. It does not reveal hidden computation, and the text need not explain the answer. |
