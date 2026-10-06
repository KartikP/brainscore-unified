# UMI API reference and cookbook

UMI is a production candidate. See [Getting started](getting_started.md), [Tools and integrations](tool_authoring.md), and the [support matrix](supported_features.md).

## Which model class to use

Import from `brainscore_core.model_interface`:

- `Subject`: implement identity, input/output channels, and `interact(session)` for a native integration.
- `UnifiedModel`: a `Subject` base with `process`, task/recording setup, and layer/modality declarations.
- `BrainScoreModel`: reuse extraction wrappers and capability callables.

See [Concepts](concepts.md#subject) for how these classes fit together.

## Registry entry points

| Function | Purpose |
| --- | --- |
| brainscore.load_model(identifier) | Load a registered UMI model or adapted legacy model |
| brainscore.load_benchmark(identifier) | Load a registered benchmark |
| brainscore.load_metric(identifier, ...) | Load a registered metric |
| brainscore.load_dataset(identifier) | Load a registered DataAssembly |
| brainscore.load_stimulus_set(identifier) | Load a registered StimulusSet |
| brainscore.score(model_identifier, benchmark_identifier) | Run compatibility and memory checks, then score |

Cold loads of shipped multi-gigabyte checkpoints stop before downloading and
report the source, approximate size, cache destination, and free disk. Fully
cached weights load locally. Set `BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK=1` to
allow managed CI/EC2 downloads without that guard; see
[model downloads](model_downloads.md) for scope and disk-budget limitations.

## Subject lifecycle

A native subject consumes inputs with `session.next_input()` and returns events
with `session.emit(event)` inside `interact(session)`. It does not need model
layers, modality properties, or `process()`. Call `reset()`
between independent evaluations when state must be cleared.

`BrainScoreModel` and domain adapters also support this lifecycle:

1. start_recording(region, time_bins=None) for neural output, or
   start_task(TaskContext(...)) for behavioral output.
2. process(input_event) for a StimulusSet, StateChange, EnvironmentStep, or
   Message.
3. reset() to clear recording, task, perturbation, and provider-owned history.

Reset between independent evaluations. Callable providers can expose a `reset()`
method; BrainScoreModel invokes it once per provider, including bound-method
registrations. Both permanent adapters forward reset to supported legacy helpers.
Perturbation benchmarks must also clean up in `finally` when evaluation fails.

## Record a region

~~~python
from brainscore import load_model

model = load_model("clip-vit-b-32")
model.start_recording("IT")
assembly = model.process(stimulus_set)
print(assembly.dims)
model.reset()
~~~

The region must be present in the model's region_layer_map. Use
start_recording("all") only when that mapping is non-empty.

For separate towers with relative layer names, declare `region_modality_map`,
for example `{'IT': 'vision', 'language_system': 'text'}`. The same layer string
can then identify different layers in different towers. Recording regions from
several modalities requires `process(stimuli, multi_modality=True)` and inputs
for every requested tower. Composite regions retain per-layer `unit_index`
coordinates from before subsetting; functional selection uses these original
addresses. Custom subset extractors must retain original unit coordinates.

## Text context and presentation identity

Rows of a text StimulusSet are independent by default. To present ordered parts
of a passage, add a `context_id` column: rows with the same identifier share
preceding parts in their input order, while different identifiers reset context.
Both TextWrapper and LanguageModelAdapter implement this rule. Pereira's unified
benchmarks declare passage groups explicitly (benchmark version 2).

~~~python
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

stimuli = StimulusSet({
    'stimulus_id': ['p1-s1', 'p1-s2', 'p2-s1'],
    'sentence': ['The cat sat.', 'It slept.', 'A dog ran.'],
    'context_id': ['p1', 'p1', 'p2'],
})
~~~

The second presentation sees `The cat sat. It slept.`. Context groups belong to
one `process()` call. Streaming callers must supply complete context within
each call/window; a context ID does not create hidden state across calls.
`per_token` TextWrapper inputs must supply complete text explicitly rather than
use context grouping, so word timestamps cannot be silently applied to added
prefix tokens. Raw legacy `digest_text(list_of_parts)` keeps its passage semantics.

The language adapter checks presentation counts and available legacy row
coordinates, then restores every input presentation column, including
`stimulus_id`. Benchmarks can align outputs by those IDs without adapter-specific
repairs. Context-dependent text caches use a separate content-derived key.

## Run a behavioral task

~~~python
from brainscore_core.model_interface import TaskContext

model.start_task(TaskContext(
    task_type="probabilities",
    label_set=["real", "pseudo"],
    fitting_stimuli=train_stimuli,
))
predictions = model.process(test_stimuli)
model.reset()
~~~

Generation-capable models can provide an instruction and label_set instead of
fitting a readout. EXTENDING.md documents the callable slot signatures.

## Apply and remove a perturbation

~~~python
from brainscore_core.model_interface import Perturbation, Selection, StateChange

applied = model.process(StateChange(
    kind="ablation",
    target=Selection(layer="encoder.layers.10", indices=[0, 1]),
    perturbation=Perturbation(kind="zero"),
))
model.process(StateChange(kind="reset", handle_id=applied.handle_id))
~~~

Use a fresh RESULTCACHING_HOME or disable result caching for mutable-state
experiments. Current activation cache keys do not uniformly fingerprint every
hook or perturbation condition.

For a matched random control of a localized subset, pass
`population=selection.metadata['unit_population']` to `RandomSelection`, with
`n_total=len(population)`. Sampling `range(n_total)` would lesion different units
when the recorded population contains original indices such as `[4, 1]`.

## Run an embodied step

~~~python
from brainscore_core.model_interface import EnvironmentStep

response = model.process(EnvironmentStep(
    observation=observation,
    instruction="move to the goal",
    step_num=0,
    is_first=True,
))
action = response.action
~~~

The environment harness owns the observation and action schemas. Consult its
contract before using the generic run_environment helper.

## Streaming helpers

brainscore_core.streaming_helpers exports:

- neural_response for open-loop neural recording
- behavioral_response for behavioral sessions
- apply_state_change for perturbation sessions
- run_environment for compatible reset/step environments

`brainscore.score` accepts registry identifiers or model/benchmark objects. Use `neural_response` for direct neural measurement without a benchmark.

## Model helper imports

See [getting_started.md](getting_started.md) for the wrapper import table and
the current distinction between vision preprocessing and wrapper-backed
modalities.

## Existing Brain-Score models

Existing vision models use `BrainModel`/`look_at`; language models use `ArtificialSubject`/`digest_text`. UMI loaders adapt these domain interfaces. For a new integration, implement `Subject.interact(session)` or use `BrainScoreModel` for extraction and capability helpers.

## Environment and validation limits

Use the [shared constraints](../install/v2-constraints.txt) for the tested Python 3.11 CPU integration stack, including scikit-learn 1.7.2 and Transformers 4.57.6. The [support matrix](supported_features.md) distinguishes proposed profiles from completed qualification.

Large models and full benchmark runs require suitable compute and staged data. An illustrative notebook or synthetic test does not establish brain-alignment or GPU qualification.

## Experiments and tools

`Experiment(subject=..., protocol=..., tools=..., output_dir=...)` assembles a run without changing the subject contract. Use `SessionProtocol` for session-based subjects or `CallableProtocol` around an existing evaluator. Conditions name settings being compared; trials identify repetitions.

`RecordInputsOutputs`, `RecordActivity`, and `Ablate` record or change the run. Layer targets use model paths and `Selection`; `result.record` returns the shared `RunRecord` reader. See [set up an experiment](experiment_toolbox.md) for the complete API and [shared vocabulary](conventions.md) for terminology.

`TorchInstrumentation` connects internal tools to a PyTorch network, including a local robotics policy. `ObserveCalls` wraps the direct `observe` mechanism; recording tools write through `RunRecorder`. See [component responsibilities](experiment_toolbox.md#how-the-pieces-fit) and [policy instrumentation](policy_instrumentation.md).

**Feature branch:** `RecordReasoning` saves exposed reasoning in `reasoning/`, read with `RunRecord(result.directory / 'reasoning')`. `build_trace_subject(..., streaming=True)` emits intermediate response traces during sessions and parses only the final answer. See [reasoning recording](reasoning_recording.md).
