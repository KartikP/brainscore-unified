# UMI API reference and cookbook

This is the shipped task-oriented reference for the public UMI surface. It
supplements source docstrings and the domain-specific legacy documentation.

## Registry entry points

| Function | Purpose |
| --- | --- |
| brainscore.load_model(identifier) | Load a registered UMI model or adapted legacy model |
| brainscore.load_benchmark(identifier) | Load a registered benchmark |
| brainscore.load_metric(identifier, ...) | Load a registered metric |
| brainscore.load_dataset(identifier) | Load a registered DataAssembly |
| brainscore.load_stimulus_set(identifier) | Load a registered StimulusSet |
| brainscore.score(model_identifier, benchmark_identifier) | Run compatibility and memory checks, then score |

## Subject lifecycle

The public subject contract is:

1. start_recording(region, time_bins=None) for neural output, or
   start_task(TaskContext(...)) for behavioral output.
2. process(input_event) for a StimulusSet, StateChange, EnvironmentStep, or
   Message.
3. reset() to clear recording, task, and perturbation state.

BrainScoreModel is the compositional implementation. Subject is the preferred
interface name; UnifiedModel is a compatibility alias.

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

- score_stimuli for open-loop neural recording
- score_behavior for behavioral sessions
- apply_state_change for perturbation sessions
- run_environment for compatible reset/step environments

The current score function accepts registry identifiers. Use score_stimuli for
an ad-hoc in-memory subject.

## Model helper imports

See [getting_started.md](getting_started.md) for the wrapper import table and
the current distinction between vision preprocessing and wrapper-backed
modalities.

## Legacy migration

BrainModel/look_at and ArtificialSubject/digest_text are pre-UMI interfaces.
New cross-domain code should use Subject or BrainScoreModel with process().
The domain packages retain legacy APIs for compatibility and adapt them when
loaded through the unified registry.

## Known distribution boundaries

- Python 3.11, NumPy below 2, xarray 2022.3.0, sklearn 1.5.x, and
  Transformers 4.57.x are the supported shared versions.
- Full video, audio-video, Algonauts, and large VLM scoring are EC2-only.
- Notebook results marked illustrative or demo-only are not brain-alignment
  validation.
