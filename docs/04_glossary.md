# Glossary

One line per term, alphabetical. For how the terms fit together, read [Core concepts](02_core_concepts.md). The code, the error messages and these documents use one word for one thing; the words in this table are those words.

| Term | Meaning |
| --- | --- |
| **Ablate** | An experiment tool that sets selected unit outputs of a network that is not a subject to zero during chosen conditions. A subject is changed with `Intervene`. |
| **Adapter** | A permanent wrapper that presents a classic Brain-Score vision or language model as a subject. |
| **`answer`** | A `BrainScoreModel` part: `(stimulus_row, instruction, label_set) -> label`. Produces behavior by generating a reply. |
| **Assembly** | A labeled array. `NeuroidAssembly` has `presentation` (stimuli), `neuroid` (units) and optionally `time_bin`. `BehavioralAssembly` holds choices. |
| **Benchmark** | A fixed test: stimuli, requested outputs, a metric and measured data. Returns a score. It declares `required_input_channels` and `requested_output_channels`, and its `__call__(subject)` takes the subject. |
| **`BrainScoreModel`** | A ready-made subject assembled from parts (extractor, region map, answer, policy, intervention). |
| **`CallableProtocol`** | A protocol that wraps an existing evaluator and keeps its loop, resets and scoring. |
| **Ceiling** | An estimate of how well the measured data can be predicted, from its reliability. A ceiled score divides by it. |
| **Channel** | A named kind of input or output, written `family` or `family:address`, such as `vision:image` or `neural:IT`. Names match exactly. |
| **Column table** | The one table that says which channel each stimulus column feeds, such as `image_path` to `vision:image` and `sentence` to `text`: `io_catalog.stimulus_columns()`. Sessions, pre-flight and `BrainScoreModel` read it; `register_channel(..., columns=...)` adds to it. |
| **Compatibility** | The pre-flight check, from declarations only, of whether a subject can take part in a benchmark. Failing pairs are N/A. |
| **`CompositeSelector`** | A region defined by units from several layers. |
| **Condition** | A setting compared within an experiment, such as `baseline` or `silenced`. |
| **Control events** | Three events on the `control` channel: `begin` opens an episode, `respond` asks the subject to answer, `end` closes the episode and says why (`terminal`, `truncated` or `failure`). A subject gets them only if it declares the `control` input. |
| **`EnvironmentSource`** | A source that drives an environment (`reset`, `step`): each action changes the next row. |
| **Event (`StreamEvent`)** | One item on a channel: `channel`, `payload`, `t_ms`, `meta`. |
| **Experiment** | A subject, a protocol, tools and an output folder, run together. Produces a saved record; need not produce a score. |
| **`extractor`** | A `BrainScoreModel` part, one per input type, that prepares inputs, runs the network and returns layer activity. |
| **Instrumentation** | The connection that lets tools reach inside a model, such as `TorchInstrumentation`. |
| **`intervention`** | A change to a subject while it runs (a lesion, a stimulation, a drug), and the `BrainScoreModel` part that applies it: `StateChange -> (PerturbationApplied, undo_function)`. The channel name (`lesion`, `stimulation`, `pharmacological`) says the kind. |
| **Layer path** | A module's name in the network, such as `layer3.0.conv1`; list them with `named_modules()`. |
| **`in_channels`, `out_channels`** | What a subject declares: the inputs it can take and the outputs it can give. |
| **`LayerSelector`** | One layer as a region; `modality=` picks the extractor in a model with more than one. |
| **Metric** | The calculation that compares model outputs with measured data. |
| **Modality** | A kind of sensory input: `vision` (pictures), `video`, `text`, `audio`. A modality is not a channel. Pictures arrive on `vision:image` and video on `vision:video`. |
| **N/A** | Not applicable: the pair was not run because declarations do not fit. Never the same as a zero score. |
| **Neuroid** | The assembly dimension that holds recorded units: model units, voxels or parcels. Outside an assembly, say layer or unit. |
| **Null baseline** | A reference without the signal of interest, such as chance or random features. |
| **`outputs`** | The channels a session requests, such as `['neural:IT']` or `['behavior']`: `stimulus_session(stimuli, outputs, task)`. A session rejects a response on a channel it did not request. |
| **`output_inputs`** | A subject declaration: for an output channel, the input channels it can be answered from. The subject answers the output when it takes in at least one of them from the benchmark. An output that is not listed can be answered from any input. |
| **Participant** | A person whose data a benchmark holds. Data coordinates keep the name the data has (for example the `subject` coordinate of an fMRI assembly). |
| **`participate(session)`** | The one evaluation method every subject implements. |
| **`Perturbation`** | How an intervention changes selected units: `zero`, `scale`, `drive` or `replace`. |
| **`policy`** | A `BrainScoreModel` part: `row -> {output channel: payload}`. |
| **`process(...)`** | A convenience method on `BrainScoreModel` and adapters for interactive use. Not part of the contract. |
| **Protocol** | The procedure of an experiment: which conditions, how many repetitions, where sessions begin and end, and when the subject is reset. |
| **`record`** | To read activity from a subject; a saved run is a `RunRecord`. What to record is a requested output (`neural:<region>`), never a separate argument. |
| **Readout** | A classifier fitted on one layer's activity to produce choices (`behavioral_readout_layer`). |
| **Region** | A brain area a benchmark names, such as `IT`. A benchmark requests it as the channel `neural:<region>`. |
| **`region_layer_map`** | Which layer (or units) stands in for each brain region. A hypothesis that benchmarks test. |
| **`required_input_channels`, `requested_output_channels`** | What a benchmark, a protocol or a tool declares: the inputs its stimuli give and the outputs it needs. |
| **Response** | An event a subject sends with `session.respond`. `TaskContext.response` is the form of the answer; `answer` and `reply` are part names. |
| **`reset()`** | Clears run state between independent runs. Does not retrain the model. |
| **`RunRecord`** | Reads a saved experiment without running the model. |
| **Score** | A metric result with metadata. Raw or ceiled. |
| **`Selection`** | A concrete set of units: layer path, optional indices, axis. Shared by recording and interventions. |
| **Session** | One continuous visit by one subject, between two resets: `next_input()`, `respond(event)`, `collect()`. You build it from a source, the outputs you request and the task. It holds many trials, or one continuous stream. |
| **Repetition** | One independent run of a condition: one session, with the subject reset before and after it. |
| **Source** | Where a session's inputs come from: a list or generator of events (`FixedSource`), or an environment (`EnvironmentSource`). |
| **`SessionProtocol`** | A protocol that creates a fresh session for each repetition and resets the subject around it. |
| **`StateChange`** | The payload of an intervention: apply a change to a resolved selection, or undo one by its handle. Sent with `apply_state_change` or the `Intervene` tool. |
| **Stimuli** | A `StimulusSet` given to a session or a helper: the argument is always `stimuli`. One stimulus is a row. |
| **`StimulusSet`** | A table with one row per stimulus; `stimulus_id` links each row to its response. |
| **Subject** | The thing evaluated: anything that declares channels and implements `participate`. Every function that takes it names the argument `subject`. A person in the data is a participant. |
| **Task** | The `TaskContext` of a session, given as `task`: what the subject is asked to do. |
| **`TaskContext`** | What the subject is asked to do (`instruction`, or `task_type='passive'`), the form of the answer (`response`: `Choice`, `Scale`, `Text`), labeled `fitting_stimuli` for a readout, and `prefer_path`. |
| **Visual form** | Whether a subject takes pictures (`vision:image`), video (`vision:video`) or both, natively. Benchmarks show each stimulus in its own form when the subject has it, and record any adaptation on the score. |
| **Prompt** | One trial as a replying model sees it, in order: the instruction, the stimulus parts (in `present` order), then the response's directive. |
| **Generation settings** | `max_tokens`, `temperature`, `top_p`, `seed` on a `Text` response. They belong to the task; a `reply` part reads them from `prompt.response`. |
| **Raw text, raw scores** | `Choice(raw_text=True)` shows the stimulus text exactly as given, with no added sentence; a part that scores options scores `f"{text} {option}"` as plain text. `Choice(raw_scores=True)` returns each option's summed log probability, not normalized. `Choice(labels='options')`, with a column name in place of the options, reads options that differ for each row from that stimulus column. |
| **Reported / implied** | A reported answer is the model's parsed reply; an implied one is read from its internals (a readout, or a language model's likelihoods). Results say which in `attrs['measure']`. |
| **Trial memory** | How long a subject keeps its trials: `none` (default; each trial is its own episode) or `history` (one episode; the subject keeps what it took in and answered on earlier trials; needs the trait `memory:trial_history`). |
| **Tool** | Equipment attached to an experiment. It watches the session (inputs and outputs) or the layers (`TorchInstrumentation`), or changes the subject through a declared channel (`Intervene`). No tool wraps a method of the subject. |
| **Row** | One closed-loop step: a mapping of channel to payload. A policy answers it with a mapping of output channel to payload. |
| **Trait `memory:trial_history`** | A subject declares it when it keeps earlier trials and its answers from `begin` until `end`. A `BrainScoreModel` declares it with `memory='trial_history'`. |
| **Typed helpers** | Shorthands that build one session and collect it: `neural_response(subject, stimuli, outputs)`, `behavioral_response(subject, stimuli, task)`, `apply_state_change(subject, state_change)`, `run_environment(subject, environment)`. They use the words of `stimulus_session`. |
| **Trial** | A unit within a session: one stimulus (one `StimulusSet` row, perhaps with several parts) followed by a response. |
| **`UnitSelection`** | A recipe for choosing units (`IndexSelection`, `RandomSelection`, `FunctionalSelection`) that resolves to a `Selection`. |
| **Wrapper** | An extractor shipped with Brain-Score, such as `VLMVisionWrapper` or `TextWrapper`. |
