# Glossary

One line per term, alphabetical. For how the terms fit together, read [Core concepts](02_core_concepts.md).

| Term | Meaning |
| --- | --- |
| **Ablate** | An experiment tool that sets selected unit outputs to zero during chosen conditions. |
| **Adapter** | A permanent wrapper that presents a classic Brain-Score vision or language model as a subject. |
| **`answer`** | A `BrainScoreModel` part: `(stimulus_row, instruction, label_set) -> label`. Produces behavior by generating a reply. |
| **Assembly** | A labeled array. `NeuroidAssembly` has `presentation` (stimuli), `neuroid` (units) and optionally `time_bin`. `BehavioralAssembly` holds choices. |
| **Benchmark** | A fixed test: stimuli, requested outputs, a metric and measured data. Returns a score. |
| **`BrainScoreModel`** | A ready-made subject assembled from parts (extractor, region map, answer, policy, intervention). |
| **`BufferedSession`** | A session with fixed inputs that stores every response. |
| **`CallableProtocol`** | A protocol that wraps an existing evaluator and keeps its loop, resets and scoring. |
| **Ceiling** | An estimate of how well the measured data can be predicted, from its reliability. A ceiled score divides by it. |
| **Channel** | A named kind of input or output, written `family` or `family:address`, such as `vision` or `neural:IT`. |
| **Compatibility** | The pre-flight check, from declarations only, of whether a subject can take part in a benchmark. Failing pairs are N/A. |
| **`CompositeSelector`** | A region defined by units from several layers. |
| **Condition** | A setting compared within an experiment, such as `baseline` or `silenced`. |
| **`EnvironmentSession`** | A session in which each action changes the next observation. |
| **`EnvironmentStep` / `EnvironmentResponse`** | One observation from an environment, and the subject's action in reply. |
| **Event (`StreamEvent`)** | One item on a channel: `channel`, `payload`, `t_ms`, `meta`. |
| **Experiment** | A subject, a protocol, tools and an output folder, run together. Produces a saved record; need not produce a score. |
| **`extractor`** | A `BrainScoreModel` part, one per input type, that prepares inputs, runs the network and returns layer activity. |
| **Instrumentation** | The connection that lets tools reach inside a model, such as `TorchInstrumentation`. |
| **`intervention`** | A `BrainScoreModel` part: `StateChange -> (PerturbationApplied, undo_function)`. |
| **Layer path** | A module's name in the network, such as `layer3.0.conv1`; list them with `named_modules()`. |
| **`LayerSelector`** | One layer as a region; `modality=` picks the tower in a multi-tower model. |
| **Metric** | The calculation that compares model outputs with measured data. |
| **Modality** | A kind of sensory input: `vision` (including video), `text`, `audio`. |
| **N/A** | Not applicable: the pair was not run because declarations do not fit. Never the same as a zero score. |
| **Neuroid** | One recorded unit of a model or brain. |
| **Null baseline** | A reference without the signal of interest, such as chance or random features. |
| **`participate(session)`** | The one evaluation method every subject implements. |
| **`Perturbation`** | How to change selected units: `zero`, `scale` or `replace`. |
| **`policy`** | A `BrainScoreModel` part: `EnvironmentStep -> EnvironmentResponse`. |
| **`process(...)`** | A convenience method on `BrainScoreModel` and adapters for interactive use. Not part of the contract. |
| **Protocol** | The procedure of an experiment: which sessions, conditions and trials to run. |
| **Readout** | A classifier fitted on one layer's activity to produce choices (`behavioral_readout_layer`). |
| **`region_layer_map`** | Which layer (or units) stands in for each brain region. A hypothesis that benchmarks test. |
| **`reset()`** | Clears run state between independent runs. Does not retrain the model. |
| **`RunRecord`** | Reads a saved experiment without running the model. |
| **Score** | A metric result with metadata. Raw or ceiled. |
| **`Selection`** | A concrete set of units: layer path, optional indices, axis. Shared by recording and perturbation. |
| **Session** | One exchange of inputs and responses with one subject: `next_input()`, `respond(event)`, `collect()`. |
| **`SessionProtocol`** | A protocol that creates a fresh session for each trial and resets the subject around it. |
| **`StateChange`** | A request to apply a perturbation to a selection, or to undo one. |
| **`StimulusSet`** | A table with one row per stimulus; `stimulus_id` links each row to its response. |
| **Subject** | The participant: anything that declares channels and implements `participate`. |
| **`TaskContext`** | What the subject is asked to do (`instruction`, or `task_type='passive'`), the form of the answer (`response`: `Choice`, `Scale`, `Text`), labeled `fitting_stimuli` for a readout, and `prefer_path`. |
| **Visual form** | Whether a subject takes pictures (`vision:still`), video (`vision:clip`) or both, natively. Benchmarks show each stimulus in its own form when the subject has it, and record any adaptation on the score. |
| **Prompt** | One trial as a replying model sees it, in order: the instruction, the stimulus parts (in `present` order), then the response's directive. |
| **Reported / implied** | A reported answer is the model's parsed reply; an implied one is read from its internals (a readout, or a language model's likelihoods). Results say which in `attrs['measure']`. |
| **Tool** | Equipment attached to an experiment to observe or change the run, without changing the subject. |
| **Trial** | One independent repetition of a condition, with a fresh session. |
| **`UnitSelection`** | A recipe for choosing units (`IndexSelection`, `RandomSelection`, `FunctionalSelection`) that resolves to a `Selection`. |
| **Wrapper** | An extractor shipped with Brain-Score, such as `VisionWrapper` or `TextWrapper`. |
