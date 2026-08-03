# Coming from Brain-Score

For existing Brain-Score users — `score(model_identifier, benchmark_identifier)`, model
plugins returning a `ModelCommitment`, an `ArtificialSubject` for language — this page
maps the established API onto the Unified Model Interface (UMI).

## Score through UMI

Scoring is currently performed per domain:

~~~python
from brainscore_vision import score
s = score(model_identifier="alexnet", benchmark_identifier="MajajHong2015.IT-pls")
~~~

UMI checks the unified registry, then the vision and language registries:

~~~python
import brainscore
s = brainscore.score("alexnet", "MajajHong2015.IT-pls")   # same identifiers, one call
~~~

`load_model`, `load_benchmark`, and `score` use these registries. `score` also
takes already-built objects, not just identifiers.

## Concept map

| In Brain-Score today | In UMI |
| --- | --- |
| `ModelCommitment` / `BrainModel` (the concrete scored model) | `BrainScoreModel` (the concrete model that is constructed) |
| `look_at(stimuli)` (vision) · `digest_text(text)` (language) | one method: `process(input_event)` |
| `activations_model` = `PytorchWrapper(...)` | `PytorchWrapper`, plus `TextWrapper` / `VideoWrapper` / `AudioWrapper` / `VLMVisionWrapper` for other modalities |
| `get_layers(...)` + `ModelCommitment(layers=...)` | `region_layer_map` on `BrainScoreModel` (any region → any layer) |
| `model_registry["id"] = lambda: ...` | `model_registry` in `unified/brainscore/models/<name>/__init__.py` |
| `ArtificialSubject` (language ABC) | `BrainScoreModel` with a `TextWrapper`; UMI adapter for the legacy ABC |

## Capabilities

The single `process(input_event)` method lets one model take more than one kind
of input:

- **Cross-domain scoring.** Register a vision-language model once and score it on
  MajajHong (vision) *and* Pereira (language).
- **Multimodal benchmarks.** A model with more than one preprocessor is driven on
  the modality (or modalities) a benchmark provides.
- **New capabilities**, each an optional slot on `BrainScoreModel`:
  - `generation_fn` / `behavioral_readout_layer` — behavioral tasks (e.g. ROAR).
  - `action_fn` — closed-loop embodied evaluation (`process(EnvironmentStep)`).
  - `state_change_fn` — lesion / perturbation studies (`process(StateChange)`).

## Registering a model: before and after

**Brain-Score today** — a vision plugin commits a region→layer mapping and scores
via `look_at`:

~~~python
# models/<name>/__init__.py
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
model_registry["my-cnn"] = lambda: ModelCommitment(
    identifier="my-cnn", activations_model=get_model(), layers=get_layers())
~~~

**UMI** — construct a `BrainScoreModel` with an explicit `region_layer_map` and
`preprocessors`, scored via `process`:

~~~python
# unified/brainscore/models/<name>/__init__.py
from brainscore import model_registry
from brainscore_core.model_interface import BrainScoreModel
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

def load_model():
    # Build the network ONCE and hand the same object to both. Calling backbone()
    # twice loads it into memory twice AND gives the wrapper a different instance
    # from the one the model holds — so a perturbation applied to one would not
    # affect what the other extracts.
    net = backbone()
    activations = PytorchWrapper(identifier="my-cnn", model=net,
                                 preprocessing=preprocess)
    return BrainScoreModel(
        "my-cnn", model=net, activations_model=activations,
        preprocessors={"vision": preprocess},
        region_layer_map={"V1": "layer1", "V4": "layer3", "IT": "layer4"})

model_registry["my-cnn"] = load_model
~~~

Choose a wrapper from the table in [getting_started.md](getting_started.md). To
add a benchmark or a new capability, continue in
[EXTENDING.md](../EXTENDING.md).
