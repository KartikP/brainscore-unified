# Perturb and select units

The code on this page runs as written. A small network, its image preparation and
twelve images labeled `face` or `object` stand in for your network and your localizer:

```python
import os
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


# Stand-in network, its extractor, and twelve localizer images
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer3 = torch.nn.Conv2d(3, 12, 3)
        self.layer4 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(2),
                                          torch.nn.Flatten(), torch.nn.Linear(48, 64))

    def forward(self, x):
        return self.layer4(self.layer3(x))


def preprocess(paths):
    images = [np.asarray(Image.open(path).convert('RGB'), np.float32) / 255 for path in paths]
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)


torch.manual_seed(0)
net = Net().eval()
vision = PytorchWrapper(identifier='my-cnn', model=net, preprocessing=preprocess)

workspace = tempfile.TemporaryDirectory()      # removed when the script ends
folder = workspace.name
rows = []
for index in range(12):
    label = 'face' if index % 2 == 0 else 'object'
    pixels = np.random.RandomState(index).randint(0, 128, (16, 16, 3), dtype=np.uint8)
    pixels[..., 0] += 127 if label == 'face' else 0      # the 'face' images are redder
    path = os.path.join(folder, f'{index}.png')
    Image.fromarray(pixels).save(path)
    rows.append({'stimulus_id': f'image{index}', 'image_path': path, 'label': label})
localizer_stimuli = StimulusSet(pd.DataFrame(rows))
localizer_stimuli.identifier = 'my-localizer'
```

## Interventions

An intervention changes the subject while it runs: a lesion, a stimulation or a drug. Three objects describe it:

| Object | Says |
| --- | --- |
| `Selection(layer, indices=None, axis=-1)` | Which units. `indices=None` means the whole layer. |
| `Perturbation(kind='zero' \| 'scale' \| 'drive' \| 'replace', scale=..., amount=...)` | How to change them. An ablation of activity takes `zero`, `scale` or `drive`; `replace` changes weights or attention heads only, and an activity ablation with it fails at the first forward pass. |
| `StateChange(kind='ablation', target=..., perturbation=...)` | The intervention. `StateChange(kind='reset', handle_id=...)` undoes one. |

A `BrainScoreModel` applies an intervention with its `intervention` part, which declares the `lesion`, `stimulation` and `pharmacological` inputs. For PyTorch networks, build the part with forward hooks. There is one way to send an intervention: a session on the declared channel. `apply_state_change` builds that session:

```python
from brainscore import BrainScoreModel
from brainscore_core import Perturbation, Selection, StateChange
from brainscore_core.streaming_helpers import apply_state_change
from brainscore.model_helpers.intervention import pytorch_intervention

# Model with an intervention part (forward hooks)
model = BrainScoreModel(identifier='my-cnn', model=net, extractor={'vision': vision},
                        region_layer_map={'IT': 'layer4'},
                        intervention=pytorch_intervention(net))

# Silence channels 0, 5 and 9 of layer3
applied = apply_state_change(model, StateChange(kind='ablation',
                             target=Selection(layer='layer3', indices=[0, 5, 9], axis=1),  # axis 1: channels
                             perturbation=Perturbation(kind='zero')))
# ... record or score the changed subject ...
apply_state_change(model, StateChange(kind='reset', handle_id=applied.handle_id))  # undo this one
model.reset()        # undoes every active intervention
```

Undoing one intervention leaves the others in place. A benchmark undoes every intervention it applies, by its handle; `score()` fails when a benchmark leaves one active.

In an experiment with several conditions, attach the `Intervene` tool: it sends the same `StateChange` with `apply_state_change` at the start of each matching repetition and undoes it at the end ([Experiments](experiments.md)).

**Two ways to change activity.** `Intervene` sends a `StateChange` to the subject's `intervention` part. The `Ablate`, `ScaleActivity` and `PatchActivity` tools put hooks on a network through `TorchInstrumentation`, outside the subject: on a model that is not a subject (an OpenPI policy server), or on the network inside a subject when no `StateChange` says the change (patching activity from another input has no `StateChange` form). Use `Intervene` for a subject whenever a `StateChange` says the change. The two ways differ in who owns each consequence:

| | `Intervene` on a subject | A network tool on a network inside a subject |
| --- | --- | --- |
| Undo | the subject, by the handle, at the end of the repetition; `reset()` of the subject also undoes it | the tool, when its repetition ends, also after an error; `reset()` of the subject does not undo it |
| Readout | fits again after each apply and undo; `readout_fit='intact'` fits with the change switched off | not told; the experiment resets the subject at each repetition, so the readout fits on the changed network; `readout_fit='intact'` with a network tool is refused before any compute |
| Stored activations | not read or stored while the change is active | the key holds the network's forward hooks, so a hooked pass gets its own entry; `PatchActivity` refuses a pass served from storage |
| Record of the change | the subject answers on `perturbation`; `Intervene` adds an `intervention` event with the handle | the tool adds `intervention_start` and `intervention_end` events with its description; `PatchActivity` also saves its reference |

**Axes.** `axis=-1` indexes the last tensor axis (units of a linear layer). For convolution outputs `(batch, channels, height, width)`, `axis=1` selects channels and `axis=None` indexes the flattened features.

**A custom intervention part.** `intervention=` takes any function `StateChange -> (PerturbationApplied, cleanup)`. The model does the following around it:

- It keeps the cleanup under the returned `handle_id`, which must be new for each call. Another return is refused: the model runs the returned cleanup, if there is one, and raises.
- Each apply and each undo makes the behavioral readout fit again at the next task; a readout fitted with `TaskContext(readout_fit='intact')` stays. That fit can switch off only changes made by `pytorch_intervention`; with a custom change active, that fit stops with `ExecutionNotFeasibleError`.
- While any intervention is active, the model reads no stored activations and stores none. After the last undo it uses them again. Their key holds the extractor's network: its weights, buffers, structure and forward hooks. A change that is still there after the undo and that the key cannot see (for example a Python attribute a module reads) needs `RESULTCACHING_DISABLE=1`.
- The other parts (`answer`, `reply`, `likelihood`, `policy`) are not told of a change. A part that keeps results of its own must not keep them across a change; `reset()` of the model calls the `reset()` of each part that has one.
- A cleanup that raises keeps its handle, and `reset()` of the model raises and names it, so you can try again.

## Choose units by function

A `UnitSelection` is a recipe that `resolve(subject)` turns into a `Selection`:

| Recipe | Chooses |
| --- | --- |
| `IndexSelection(layer, indices)` | The units you list. |
| `FunctionalSelection(region, localizer_stimuli, contrast=(['face'], ['object']), n_units=50)` | The units that respond most to one category over another, like an fMRI localizer. |
| `RandomSelection(layer, n_units, n_total, population=..., seed=...)` | Random units: the control. |

`FunctionalSelection` records the region through a session: it works for any subject that records the region, and it takes the labels from the `contrast_column` of the localizer stimuli. For a region that spans several layers, `resolve_layers(subject)` gives one `Selection` per layer. Resolve every recipe before the intervention: the target of a `StateChange` is a resolved `Selection`. An unresolved recipe is refused with an error that names this fix. Two reasons: the localizer is its own session, and a subject takes part in one session at a time; and a resolved target is the exact set of units, so a recorded run replays without the localizer. Always compare a localized lesion with random units of the same number, drawn from the same population:

```python
from brainscore import FunctionalSelection, RandomSelection

# 10 IT units most selective for faces
selected = FunctionalSelection(region='IT', localizer_stimuli=localizer_stimuli,
                               contrast=(['face'], ['object']), n_units=10).resolve(model)
# Control: 10 random units, same layer
control = RandomSelection.matched_to(selected, seed=0).resolve(model)
print(selected.layer, len(selected.indices), control.layer, len(control.indices))   # layer4 10 layer4 10
```

`RandomSelection.matched_to` takes the layer, the number of units, the population and the
axis from the selection, so the control draws from the same units in the same way.

Use held-out stimuli for the localizer and for the test, and several random seeds.

## Regions from several layers

```python
from brainscore import CompositeSelector

# IT: units 0-2 of layer3, all of layer4
region_layer_map = {'IT': CompositeSelector(layers=(('layer3', (0, 1, 2)), ('layer4', None)))}
```

See [Lesson 7](../../notebooks/lessons/07_perturb_a_model.ipynb) and [Lesson 8](../../notebooks/lessons/08_choose_units.ipynb).
