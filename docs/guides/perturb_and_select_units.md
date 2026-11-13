# Perturb and select units

## Interventions

An intervention changes the subject while it runs: a lesion, a stimulation or a drug. Three objects describe it:

| Object | Says |
| --- | --- |
| `Selection(layer, indices=None, axis=-1)` | Which units. `indices=None` means the whole layer. |
| `Perturbation(kind='zero' \| 'scale' \| 'drive' \| 'replace', scale=..., amount=...)` | How to change them. |
| `StateChange(kind='ablation', target=..., perturbation=...)` | The intervention. `StateChange(kind='reset', handle_id=...)` undoes one. |

A `BrainScoreModel` applies an intervention with its `intervention` part, which declares the `lesion`, `stimulation` and `pharmacological` inputs. For PyTorch networks, build the part with forward hooks. There is one way to send an intervention: a session on the declared channel. `apply_state_change` builds that session:

```python
from brainscore import BrainScoreModel
from brainscore_core import Perturbation, Selection, StateChange
from brainscore_core.streaming_helpers import apply_state_change
from brainscore.perturbation import build_pytorch_ablation_fn

model = BrainScoreModel(identifier='my-cnn', model=net, extractor={'vision': vision},
                        region_layer_map={'IT': 'layer4'},
                        intervention=build_pytorch_ablation_fn(net))

applied = apply_state_change(model, StateChange(kind='ablation',
                             target=Selection(layer='layer3', indices=[0, 5, 9], axis=1),
                             perturbation=Perturbation(kind='zero')))
# ... record or score the changed subject ...
apply_state_change(model, StateChange(kind='reset', handle_id=applied.handle_id))  # undo this one
model.reset()        # undoes every active intervention
```

Undoing one intervention leaves the others in place. A benchmark undoes every intervention it applies, by its handle; `score()` fails when a benchmark leaves one active.

In an experiment with several conditions, attach the `Intervene` tool: it sends the same `StateChange` with `apply_state_change` at the start of each matching repetition and undoes it at the end ([Experiments](experiments.md)).

**A network that is not a subject.** The `Ablate`, `ScaleActivity` and `PatchActivity` tools change a network through `TorchInstrumentation` hooks, outside any subject. Use them only for a model that is not a subject, for example an OpenPI policy server. For a subject, use `Intervene`.

**Axes.** `axis=-1` indexes the last tensor axis (units of a linear layer). For convolution outputs `(batch, channels, height, width)`, `axis=1` selects channels and `axis=None` indexes the flattened features.

**Caching.** Activation caches check the weights and settings before reusing results. If your intervention changes state the cache cannot see, run with `RESULTCACHING_DISABLE=1`.

## Choose units by function

A `UnitSelection` is a recipe that `resolve(subject)` turns into a `Selection`:

| Recipe | Chooses |
| --- | --- |
| `IndexSelection(layer, indices)` | The units you list. |
| `FunctionalSelection(region, localizer_stimuli, contrast=(['face'], ['object']), n_units=50)` | The units that respond most to one category over another, like an fMRI localizer. |
| `RandomSelection(layer, n_units, n_total, population=..., seed=...)` | Random units: the control. |

`FunctionalSelection` records the region through a session: it works for any subject that records the region, and it takes the labels from the `contrast_column` of the localizer stimuli. For a region that spans several layers, `resolve_layers(subject)` gives one `Selection` per layer. Resolve every recipe before the intervention: the target of a `StateChange` is a resolved `Selection`. An unresolved recipe is refused with an error that names this fix. Two reasons: the localizer is its own session, and a subject takes part in one session at a time; and a resolved target is the exact set of units, so a recorded run replays without the localizer. Always compare a localized lesion with random units of the same number, drawn from the same population:

```python
selected = FunctionalSelection('IT', localizer_stimuli, contrast=(['face'], ['object']), n_units=50).resolve(model)
population = selected.metadata['unit_population']
control = RandomSelection(selected.layer, n_units=50, n_total=len(population),
                          population=population, seed=0).resolve(model)
```

Use held-out stimuli for the localizer and for the test, and several random seeds.

## Regions from several layers

```python
from brainscore import CompositeSelector

region_layer_map = {'IT': CompositeSelector(layers=(('layer3', (0, 1, 2)), ('layer4', None)))}
```

See [Lesson 7](../../notebooks/lessons/07_perturb_a_model.ipynb) and [Lesson 8](../../notebooks/lessons/08_choose_units.ipynb).
