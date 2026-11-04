# Perturb and select units

## Perturb

Three objects describe a perturbation:

| Object | Says |
| --- | --- |
| `Selection(layer, indices=None, axis=-1)` | Which units. `indices=None` means the whole layer. |
| `Perturbation(kind='zero' \| 'scale' \| 'replace', scale=...)` | How to change them. |
| `StateChange(kind='ablation', target=..., perturbation=...)` | Apply it. `StateChange(kind='reset', handle_id=...)` undoes one. |

A `BrainScoreModel` applies them through its `intervention` part. For PyTorch networks, build one with forward hooks:

```python
from brainscore_core import BrainScoreModel, Perturbation, Selection, StateChange
from brainscore.perturbation import build_pytorch_ablation_fn

model = BrainScoreModel(identifier='my-cnn', model=net, extractor={'vision': vision},
                        region_layer_map={'IT': 'layer4'},
                        intervention=build_pytorch_ablation_fn(net))

applied = model.process(StateChange(kind='ablation',
                                    target=Selection(layer='layer3', indices=[0, 5, 9], axis=1),
                                    perturbation=Perturbation(kind='zero')))
# ... record or score the perturbed model ...
model.reset()        # undoes every active perturbation
```

In a benchmark, deliver the change through a session with `apply_state_change(subject, state_change)` from `brainscore_core.streaming_helpers`. In a study with several conditions, use the `Ablate` or `ScaleActivity` tools instead ([Experiments](experiments.md)).

**Axes.** `axis=-1` indexes the last tensor axis (units of a linear layer). For convolution outputs `(batch, channels, height, width)`, `axis=1` selects channels and `axis=None` indexes the flattened features.

**Caching.** Activation caches check the weights and settings before reusing results. If your intervention changes state the cache cannot see, run with `RESULTCACHING_DISABLE=1`.

## Choose units by function

A `UnitSelection` is a recipe that `resolve(model)` turns into a `Selection`:

| Recipe | Chooses |
| --- | --- |
| `IndexSelection(layer, indices)` | The units you list. |
| `FunctionalSelection(recording_target, localizer_stimuli, contrast=(['face'], ['object']), n_units=50)` | The units that respond most to one category over another, like an fMRI localizer. |
| `RandomSelection(layer, n_units, n_total, population=..., seed=...)` | Random units: the control. |

A recipe can be passed directly as a `StateChange` target. Always compare a localized lesion with random units of the same number, drawn from the same population:

```python
selected = FunctionalSelection('IT', localizer_stimuli, contrast=(['face'], ['object']), n_units=50).resolve(model)
population = selected.metadata['unit_population']
control = RandomSelection(selected.layer, n_units=50, n_total=len(population),
                          population=population, seed=0).resolve(model)
```

Use held-out stimuli for the localizer and for the test, and several random seeds.

## Regions from several layers

```python
from brainscore_core import CompositeSelector

region_layer_map = {'IT': CompositeSelector(layers=(('layer3', (0, 1, 2)), ('layer4', None)))}
```

See [Lesson 7](../../notebooks/lessons/07_perturb_a_model.ipynb), [Lesson 8](../../notebooks/lessons/08_choose_units.ipynb) and [Experiment 5](../../notebooks/experiments/05_how_much_ablation_matters.ipynb).
