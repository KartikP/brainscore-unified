# Glossary

One line per term, alphabetical. For how the terms fit together, read [Core concepts](02_core_concepts.md). The code, the error messages and these documents use one word for one thing; the words in this table are those words.

| Term | Meaning |
| --- | --- |
| **Act form** | How a subject acts on a screen, as a person does: `motor:key` (key acts) or `motor:pointer` (pointer acts). The payload is a list of acts, such as `[{'act': 'press', 'key': 'Space'}, {'act': 'wait', 'ms': 100}]`; positions are pixels of the last picture, origin at the top left. A benchmark asks for the form its participants used; a subject declares each form it has. Bare `motor` is an action in the form the environment defines. |
| **Ablate** | An experiment tool that sets selected unit outputs of a network that is not a subject to zero during chosen conditions. A subject is changed with `Intervene`. |
| **Adapter** | A permanent wrapper that presents a classic Brain-Score vision or language model as a subject. |
| **`answer`** | A `BrainScoreModel` part: `(stimulus_row, instruction, label_set) -> text`, called with keywords. It words its own prompt and returns the generated text; `BrainScoreModel` parses it with `Choice.parse`. It answers a plain `Choice`; a task with settings goes to `reply`. |
| **Assembly** | A labeled array. `NeuroidAssembly` has `presentation` (stimuli), `neuroid` (units) and optionally `time_bin`. `BehavioralAssembly` holds choices. |
| **Benchmark** | A fixed test: stimuli, requested outputs, a metric and measured data. Returns a score. It declares `requested_output_channels`; its inputs are read from the table it keeps (`stimulus_set`, and `task`), or declared in `required_input_channels` when it keeps no table. Its `__call__(subject)` takes the subject. |
| **`BrainScoreModel`** | A ready-made subject assembled from parts (extractor, region map, answer, policy, intervention). |
| **`BRAINSCORE_REGIONS_PER_PASS`** | An environment variable: `all` (default) records every requested region of an extractor that declares `independent_layers` in one pass; `one` gives each region its own pass. Any other value stops the session before any compute. |
| **`CallableProtocol`** | A protocol that wraps an existing evaluator and keeps its loop, resets and scoring. |
| **Ceiling** | An estimate of how well the measured data can be predicted, from its reliability. A ceiled score divides by it. |
| **Channel** | A named kind of input or output, written `family` or `family:address`, such as `vision:image` or `neural:IT`. Names match exactly. |
| **Column table** | The one table that says which channel each stimulus column feeds, such as `image_path` to `vision:image` and `sentence` to `text`: `io_catalog.stimulus_columns()`. Sessions, pre-flight and `BrainScoreModel` read it; `register_channel(..., columns=...)` adds to it. |
| **Compatibility** | The pre-flight check, from declarations only, of whether a subject can take part in a benchmark. Failing pairs are N/A. |
| **`CompositeSelector`** | A region defined by units from several layers. |
| **Condition** | A setting compared within an experiment, such as `baseline` or `silenced`. |
| **Control events** | Three events on the `control` channel: `begin` opens an episode, `respond` asks the subject to answer, `end` closes the episode and says why (`terminal`, `truncated` or `failure`). A subject gets them only if it declares the `control` input. |
| **Data plugin** | A folder `brainscore/data/<name>/` that registers the stimuli and recorded responses of one study (`stimulus_set_registry`, `data_registry`). Scripts that build its files, such as the frame, clip and sound preparation of the Friends movie data, are in its `data_packaging/` folder. A benchmark reaches the data only through `load_dataset` and `load_stimulus_set`. |
| **`EnvironmentSource`** | A source that drives an environment (`reset`, `step`): each action changes the next row. |
| **Event (`StreamEvent`)** | One item on a channel: `channel`, `payload`, `t_ms`, `meta`. |
| **`ExecutionNotFeasibleError`** | A compatible pair whose run is refused when a session or the benchmark starts, before model computation: the subject cannot run this task, these outputs or these settings. An error with no result. On an old benchmark, a refusal before any input reached the subject is N/A, with the refusal as the reason; a refusal after an input is an error. |
| **Experiment** | A subject, a protocol, tools and an output folder, run together. Produces a saved record; need not produce a score. |
| **`extractor`** | A `BrainScoreModel` part, one per input type, that prepares inputs, runs the network and returns layer activity: `extractor(stimuli, layers)`. Its `record(layers, stimuli)` records a region while another part runs the network; the recording's `take(row)` gives one assembly per layer. |
| **Generation settings** | `max_tokens`, `temperature`, `top_p`, `seed` on a `Text` response. They belong to the task; a `reply` part reads them from `prompt.response`. |
| **`in_channels`, `out_channels`** | What a subject declares: the inputs it can take and the outputs it can give. |
| **`independent_layers`** | What an extractor declares when its output for some layers, cut to one layer, equals its output for that layer alone. The regions of one session that it records then share one call. Every wrapper in `brainscore.model_helpers` declares it; any other extractor runs once per region. `BRAINSCORE_REGIONS_PER_PASS=one` gives every region its own pass. |
| **Instrumentation** | The connection that lets tools reach inside a model, such as `TorchInstrumentation`. |
| **`intervention`** | A change to a subject while it runs (a lesion, a stimulation, a drug), and the `BrainScoreModel` part that applies it: `StateChange -> (PerturbationApplied, undo_function)`. The channel name (`lesion`, `stimulation`, `pharmacological`) says the kind. |
| **Layer path** | A module's name in the network, such as `layer3.0.conv1`; list them with `named_modules()`. |
| **`LayerSelector`** | One layer as a region; `modality=` picks the extractor when the layer name alone does not: in a model with several extractor objects, the name is found in extractors of two input families (an extractor that cannot list its layers counts as having every layer). A tuple, such as `('text', 'vision')`, names the recording extractor first and then the other inputs that reach the layer. |
| **`likelihood`** | A `BrainScoreModel` part: `(prompt, candidates) -> {candidate: log probability}`. Scores each option as the reply, without generating; gives `behavior:log_probabilities`, and `behavior:probabilities` (`softmax`) and `behavior:choice` (`softmax_argmax`) when `answer_owners` names them. |
| **Metric** | The calculation that compares model outputs with measured data. |
| **Modality** | A kind of sensory input: `vision` (pictures), `video`, `text`, `audio`. A modality is not a channel. Pictures arrive on `vision:image` and video on `vision:video`. |
| **N/A** | Not applicable: the pair was not run because declarations do not fit. Never the same as a zero score. |
| **Neuroid** | The assembly dimension that holds recorded units: model units, voxels or parcels. Outside an assembly, say layer or unit. |
| **Null baseline** | A reference without the signal of interest, such as chance or random features. |
| **`output_inputs`** | A subject declaration: for an output channel, the input channels it can be answered from. The subject answers the output when it takes in at least one of them from the benchmark. An output that is not listed can be answered from any input. |
| **`outputs`** | The channels a session requests, such as `['neural:IT']` or `['behavior']`: `stimulus_session(stimuli, outputs, task)`. A session rejects a response on a channel it did not request. |
| **Participant** | A person whose data a benchmark holds. Data coordinates keep the name the data has (for example the `subject` coordinate of an fMRI assembly). |
| **`participate(session)`** | The one evaluation method every subject implements. |
| **`Perturbation`** | How an intervention changes selected units: `zero`, `scale` or `drive` for activity; `replace` only for weights and attention heads. |
| **`policy`** | A `BrainScoreModel` part, in one form: `row -> {output channel: payload}`. `policy_in_channels` (default `('observation',)`) names the row keys it reads; `policy_out_channels` (default `('motor',)`) the outputs it answers on. A policy that keeps history clears it in `reset()`. |
| **`process(row)`** | The one-row call of `BrainScoreModel`: a loop that a simulator owns calls the model one step at a time, and the model may record its regions during that step. Not part of the contract. |
| **Prompt** | One trial as a replying model sees it, in order: the instruction, the stimulus parts (in `present` order), then the response's directive. |
| **Protocol** | The procedure of an experiment: which conditions, how many repetitions, where sessions begin and end, and when the subject is reset. |
| **Readout** | A classifier fitted on one layer's activity (`behavioral_readout_layer`). It gives probabilities; its choice is the most probable option, which the registration names with `answer_owners={'behavior:choice': ('readout', 'argmax')}`. |
| **`record`** | To read activity from a subject; a saved run is a `RunRecord`. What to record is a requested output (`neural:<region>`), never a separate argument. |
| **Region** | A brain area a benchmark names, such as `IT`. A benchmark requests it as the channel `neural:<region>`. |
| **`region_layer_map`** | Which layer (or units) stands in for each brain region. A hypothesis that benchmarks test. |
| **Repetition** | One independent run of a condition: one session, with the subject reset before and after it. |
| **`reply`** | A `BrainScoreModel` part: `prompt -> text` (or a `StreamEvent` with extra data). The model's raw reply to one trial; the task's response parses it. |
| **`required_channels`** | The inputs a subject cannot run without; a subset of `in_channels`. A benchmark that does not provide all of them is N/A. On a `BrainScoreModel`, leave it out; state it only for one network that cannot run unless it gets all of the listed inputs together (for example a joint extractor). A model made of separate extractors requires nothing, because each one runs alone. The constructor checks only that the channels are declared, not this claim. |
| **`required_input_channels`, `requested_output_channels`** | What a benchmark, a protocol or a tool declares: the inputs its stimuli give and the outputs it needs. A benchmark that keeps its `stimulus_set` (and `task`) need not declare its inputs: pre-flight reads them from the table's columns and the task's instruction. |
| **`reset()`** | Clears run state between independent runs. Does not retrain the model. |
| **Response** | An event a subject sends with `session.respond`. `TaskContext.response` is the form of the answer; `answer` and `reply` are part names. |
| **Response record** | The record on every answer (`attrs['response']`) and copied to every score: format, category, origin (`native` or `derived`), method (`generation`, `likelihood`, `readout`, `link`), conversion (`argmax`, `softmax`, `softmax_argmax`), source format, parameters and `parsed_by`. Built by `brainscore_core.response_formats.response_record`. |
| **Row** | One closed-loop step: a mapping of channel to payload. A policy answers it with a mapping of output channel to payload. |
| **`RunRecord`** | Reads a saved experiment without running the model. |
| **Score** | A metric result with metadata. Raw or ceiled. |
| **`Selection`** | A concrete set of units: layer path, optional indices, axis. Shared by recording and interventions. |
| **Session** | One continuous visit by one subject, between two resets: `next_input()`, `respond(event)`, `collect()`. You build it from a source, the outputs you request and the task. It holds many trials, or one continuous stream. |
| **`SessionProtocol`** | A protocol that creates a fresh session for each repetition and resets the subject around it. |
| **Source** | Where a session's inputs come from: a list or generator of events (`FixedSource`), or an environment (`EnvironmentSource`). |
| **`StateChange`** | The payload of an intervention: apply a change to a resolved selection, or undo one by its handle. Sent with `apply_state_change` or the `Intervene` tool. |
| **Stimuli** | A `StimulusSet` given to a session or a helper: the argument is always `stimuli`. One stimulus is a row. |
| **`StimulusSet`** | A table with one row per stimulus; `stimulus_id` links each row to its response. |
| **Subject** | The thing evaluated: anything that declares channels and implements `participate`. Every function that takes it names the argument `subject`. A person in the data is a participant. |
| **Task** | The `TaskContext` of a session, given as `task`: what the subject is asked to do. |
| **`TaskContext`** | What the subject is asked to do (`instruction`, or `task_type='passive'`), the form of the answer (`response`: `Choice`, `Scale`, `Text`), and labeled `fitting_stimuli` for a readout. Which part answers is a property of the model. |
| **Tool** | Equipment attached to an experiment. It watches the session (inputs and outputs) or the layers (`TorchInstrumentation`), or changes the subject through a declared channel (`Intervene`). No experiment tool wraps a method of the subject; `Witness`, an observer outside experiments, replaces `participate` for its block and restores it. Experiment tools are in `brainscore.experiments`; tools that are fitted, saved and applied (linear probe, LEACE, steering) and the Algonauts prediction export are in `brainscore/tools/`. |
| **Trait `memory:trial_history`** | A subject declares it when it keeps earlier trials and its answers from `begin` until `end`. A `BrainScoreModel` declares it with `memory='trial_history'`. |
| **Trial** | A unit within a session: one stimulus (one `StimulusSet` row, perhaps with several parts) followed by a response. |
| **Trial memory** | How long a subject keeps its trials: `none` (default; each trial is its own episode) or `history` (one episode; the subject keeps what it took in and answered on earlier trials; needs the trait `memory:trial_history`). |
| **Typed helpers** | Shorthands that build one session and collect it: `neural_response(subject, stimuli, outputs)`, `behavioral_response(subject, stimuli, task)`, `apply_state_change(subject, state_change)`, `run_environment(subject, environment)`. They use the words of `stimulus_session`. |
| **`UnitSelection`** | A recipe for choosing units (`IndexSelection`, `RandomSelection`, `FunctionalSelection`) that resolves to a `Selection`. |
| **Verbatim, log probabilities** | `Choice(verbatim=True)` shows the trial text exactly as given, with no added sentence or instruction (a reply part keeps its own chat template if it has one; the record notes that in `parameters['template']`); a part that scores options scores `f"{text} {option}"` as plain text. A session that requests `behavior:log_probabilities` gets each option's summed log probability, not normalized, from a `likelihood` part. `Choice(labels='options')`, with a column name in place of the options, reads options that differ for each row from that stimulus column. |
| **Visual form** | Whether a subject takes pictures (`vision:image`), video (`vision:video`) or both, natively. Benchmarks show each stimulus in its own form when the subject has it, and record any adaptation on the score. |
| **Wrapper** | An extractor shipped with Brain-Score: `PytorchWrapper` (in `brainscore_vision`), `VLMVisionWrapper`, `VLMVideoWrapper`, `VideoWrapper`, `TextWrapper`, `AudioWrapper` (in `brainscore.model_helpers`). |
