# Coming from Brain-Score? Start here.

If you already use Brain-Score — `score(model_identifier, benchmark_identifier)`,
model plugins that return a `ModelCommitment`, an `ArtificialSubject` for language —
this page maps what you know onto the Unified Model Interface (UMI). The short
version:

- **Nothing you have breaks.** Every existing vision and language model plugin and
  benchmark keeps working, unchanged. UMI wraps them automatically.
- **UMI is opt-in.** You only touch the new API if you want what it adds: one
  registration that scores across the vision *and* language leaderboards,
  multimodal models, and new capabilities (behavior, embodied action, lesions).

## The scoring call is the same shape

Today you score per domain:

~~~python
from brainscore_vision import score
s = score(model_identifier="alexnet", benchmark_identifier="MajajHong2015.IT-pls")
~~~

UMI adds one entry point that reaches both domains — and it falls back to the
vision and language registries, so your existing identifiers still resolve:

~~~python
import brainscore
s = brainscore.score("alexnet", "MajajHong2015.IT-pls")   # same identifiers, one call
~~~

`load_model` / `load_benchmark` / `score` behave as you expect. `score` also takes
already-built objects, not just identifiers.

## Concept map

| In Brain-Score today | In UMI |
| --- | --- |
| `ModelCommitment` / `BrainModel` (the concrete scored model) | `BrainScoreModel` (the concrete model you construct) |
| `look_at(stimuli)` (vision) · `digest_text(text)` (language) | one method: `process(input_event)` |
| `activations_model` = `PytorchWrapper(...)` | the same `PytorchWrapper`, plus `TextWrapper` / `VideoWrapper` / `AudioWrapper` / `VLMVisionWrapper` for other modalities |
| `get_layers(...)` + `ModelCommitment(layers=...)` | `region_layer_map` on `BrainScoreModel` (any region → any layer) |
| `model_registry["id"] = lambda: ...` | the same registry pattern, in `unified/brainscore/models/<name>/__init__.py` |
| `ArtificialSubject` (language ABC) | `BrainScoreModel` with a `TextWrapper`; the legacy ABC still works via the adapter |

The legacy classes and methods are still present and still called — the UMI
adapters delegate to them. You are not being asked to rewrite anything.

## What UMI actually adds

The point of the single `process(input_event)` method is that one model can take
more than one kind of input:

- **Cross-domain scoring.** Register a vision-language model once and score it on
  MajajHong (vision) *and* Pereira (language) — no second plugin.
- **Multimodal benchmarks.** A model with more than one preprocessor is driven on
  the modality (or modalities) a benchmark provides.
- **New capabilities**, each an optional slot on `BrainScoreModel`:
  - `generation_fn` / `behavioral_readout_layer` — behavioral tasks (e.g. ROAR).
  - `action_fn` — closed-loop embodied evaluation (`process(EnvironmentStep)`).
  - `state_change_fn` — lesion / perturbation studies (`process(StateChange)`).

None of these exist in the per-domain API; all are additive.

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

**UMI** — the same backbone and wrapper, but a `BrainScoreModel` with an explicit
`region_layer_map` and `preprocessors`, scored via `process`:

~~~python
# unified/brainscore/models/<name>/__init__.py
from brainscore import model_registry
from brainscore_core.model_interface import BrainScoreModel
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

def load_model():
    activations = PytorchWrapper(identifier="my-cnn", model=backbone(),
                                 preprocessing=preprocess)
    return BrainScoreModel(
        "my-cnn", model=backbone(), activations_model=activations,
        preprocessors={"vision": preprocess},
        region_layer_map={"V1": "layer1", "V4": "layer3", "IT": "layer4"})

model_registry["my-cnn"] = load_model
~~~

You choose the wrapper the same way you do today — see the wrapper table in
[getting_started.md](getting_started.md). To add a benchmark or a new capability,
continue in [EXTENDING.md](../EXTENDING.md).

## Do I have to migrate?

No. If your model only needs the vision (or only the language) leaderboard as it
works today, leave it — the legacy plugin is auto-wrapped and scores exactly as
before. Register through UMI when you want cross-domain scoring, a multimodal
model, or one of the new capabilities.
