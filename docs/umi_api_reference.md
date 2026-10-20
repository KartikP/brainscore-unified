# UMI API reference and cookbook

UMI is a v2 source candidate. See [Getting started](getting_started.md), [Tools and integrations](tool_authoring.md), and the [support matrix](supported_features.md).

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

A target can name a mapped region or a raw layer path; unknown region names
produce a warning. Strings and lists use the same rule. `start_recording("all")`
means all mapped regions, not all network layers.

For separate towers, keep routing in the selector:

```python
from brainscore_core.model_interface import LayerSelector

region_layer_map = {
    'IT': LayerSelector(name='encoder.layers.10', modality='vision'),
    'language_system': LayerSelector(name='encoder.layers.10', modality='text'),
}
```

The same layer string
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

Supported extraction wrappers check weights, inputs, and supported hook configuration
before reusing activations. If a custom intervention's state cannot be represented
in `cache_config()`, disable result caching with `RESULTCACHING_DISABLE=1`.
Make `.data` alias or NumPy weight edits between scoring runs. See
[activation caching](caching.md) for wrapper coverage and limits.

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

The environment wrapper declares observations and actions. See
[environment sessions](environment_sessions.md) for specifications, seeded reset,
execution records, and clocks. `run_environment` returns applied actions and
closes the environment; its caller owns subject reset.

## Sessions and convenience functions

Start with `InMemorySession(stimulus_set=..., requested_output_channels=...)` for fixed inputs. Add `task_context` for behavior or `streaming=True` for row-by-row delivery. Run it with `subject.interact(session)`.

| Session | Distinct behavior |
| --- | --- |
| `InMemorySession` | Fixed inputs; collect the responses. |
| `EnvironmentSession` | Each action determines the next observation. |
| `StreamingStimulusSetSession` | Build events lazily from a stimulus table; bounded input-event buffering. |
| `WindowedStreamSession` | Group frames into bounded windows. |
| `RealTimeStreamSession` | Apply a clock and a late-input policy. |

For a single operation, `brainscore_core.streaming_helpers` also provides `neural_response`, `behavioral_response`, `apply_state_change`, and `run_environment`. Always name the target when recording, for example `neural_response(subject=subject, stimulus_set=stimuli, record='V4')`.

`StimulusSetSession`, `BehavioralSession`, `StateChangeSession`, and their factory aliases remain compatibility conveniences. New integrations do not need to learn these alternatives. Their historical default recording target, including `neural_response`'s default, remains `IT`; use an explicit target to avoid an unintended measurement.

## Model helper imports

See [getting_started.md](getting_started.md) for the wrapper import table and
the input preparation and aggregation options for each wrapper.

## Existing Brain-Score models

Existing vision models use `BrainModel`/`look_at`; language models use `ArtificialSubject`/`digest_text`. UMI loaders adapt these domain interfaces. For a new integration, implement `Subject.interact(session)` or use `BrainScoreModel` for extraction and capability helpers.

## Environment and validation limits

Use the [shared constraints](../install/v2-constraints.txt) for the tested Python 3.11 CPU integration stack, including scikit-learn 1.7.2 and Transformers 4.57.6. The [support matrix](supported_features.md) distinguishes proposed profiles from completed qualification.

Large models and full benchmark runs require suitable compute and staged data. An illustrative notebook or synthetic test does not establish brain-alignment or GPU qualification.

## Experiments and tools

`Experiment(subject=..., protocol=..., tools=..., output_dir=...)` assembles a run without changing the subject contract. Use `SessionProtocol` for session-based subjects or `CallableProtocol` around an existing evaluator. Conditions name settings being compared; trials identify repetitions.

`RecordInputsOutputs`, `RecordActivity`, and `Ablate` record or change the run. Layer targets use model paths and `Selection`; `result.record` returns the shared `RunRecord` reader. See [set up an experiment](experiment_toolbox.md) for the complete API and [shared vocabulary](conventions.md) for terminology.

`TorchInstrumentation` connects internal tools to a PyTorch network, including a local robotics policy. `ObserveCalls` wraps the direct `observe` mechanism; recording tools write through `RunRecorder`. See [component responsibilities](experiment_toolbox.md#how-the-pieces-fit) and [policy instrumentation](policy_instrumentation.md).

**Feature branch:** `OpenPIInstrumentation(policy, checkpoint=..., steps=...)` connects the same activity/ablation tools to three named sites in the pinned Pi0/Pi0.5 JAX sampler. `RemoteOpenPIInstrumentation(client)` uses `OpenPIPolicyClient` and `OpenPIToolServer` for request-scoped remote access. See [policy instrumentation](policy_instrumentation.md) for targets and qualification limits.

**Feature branch:** `RecordReasoning` saves exposed reasoning in `reasoning/`, read with `RunRecord(result.directory / 'reasoning')`. `build_trace_subject(..., streaming=True)` emits intermediate response traces during sessions and parses only the final answer. See [reasoning recording](reasoning_recording.md).

## Compatibility forms

Simple `region_layer_map` entries still read back as strings. Entries with unit
subsets or routing read back as selectors, preserving all settings when copied.
Use `model.region_layer_selectors[region].layer_path` when a consumer needs only
one layer path; use `CompositeSelector.layer_paths` for all composite layers.
The separate `region_modality_map` argument is accepted and normalized into selectors.

Existing registrations with wrappers in `preprocessors` remain accepted at the
constructor boundary. New registrations put every extraction wrapper in
`activations_model`, and configure preprocessing in the wrapper itself.

`SessionProtocol` factories receive `Trial(condition, identifier)`, including when
conditions are omitted. An ID-only factory can opt into `factory_argument='identifier'`.
The default condition is `'default'`. Session helper constructors and factory
aliases remain available; new examples use `InMemorySession` for fixed inputs.

## Choose how a behavioral answer is obtained

`TaskContext.prefer_path` determines the procedure before inputs are evaluated:

| Value | Procedure |
| --- | --- |
| `readout` | Fit a classifier on `fitting_stimuli`, then predict from recorded features. |
| `generation` | Use `instruction` and `label_set` with the model's generation function. No classifier is fitted. |
| `auto` or omitted | Prefer generation when available; otherwise fit a readout. Warn if both are available. |

These are different scientific measurements. A benchmark claiming one procedure should select it explicitly. ROAR retains its automatic registered variants; controlled comparisons can construct `Yeatman2021LexicalDecision(modality='text', prefer_path='readout')` or `prefer_path='generation'`. Those explicit choices add `-readout` or `-generation` to the benchmark identifier. Scores record the requested path; the scoring entry point also records the selected behavioral capability.

## Extend event handling

Register a `Capability` to add event handling to `BrainScoreModel`; see [tool authoring](tool_authoring.md). The older `register_input_handler` hook remains callable with a deprecation warning. Tools that only observe or intervene in experiments implement `Tool` and do not need a capability.
