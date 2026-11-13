# Wrap your model

Choose one of two routes.

| Your system | Route |
| --- | --- |
| A neural network whose layers you can access | `BrainScoreModel`, assembled from parts (below) |
| Anything with its own loop: a remote API, a simulator agent, a rule | Subclass `Subject` and write `participate` ([Lesson 1](../../notebooks/lessons/01_your_first_subject.ipynb)) |

## BrainScoreModel in four steps

The code on this page runs as written. A small network stands in for yours; replace
`net`, `preprocess` and the image files with your own:

```python
import os
import tempfile

import numpy as np
import torch
from PIL import Image


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(3, 12, 3)
        self.layer3 = torch.nn.Conv2d(12, 12, 3)
        self.layer4 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(2),
                                          torch.nn.Flatten(), torch.nn.Linear(48, 8))

    def forward(self, x):
        return self.layer4(self.layer3(self.layer1(x)))


torch.manual_seed(0)
net = Net().eval()


def preprocess(paths):
    """Image files -> the batch the network expects."""
    images = [np.asarray(Image.open(path).convert('RGB').resize((16, 16)), np.float32) / 255
              for path in paths]
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)


folder = tempfile.mkdtemp()
image_paths = {}
for seed, name in enumerate(['cat', 'dog']):
    image_paths[name] = os.path.join(folder, f'{name}.png')
    pixels = np.random.RandomState(seed).randint(0, 256, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(image_paths[name])
```

**1. Find the layer paths.**

```python
print([name for name, _ in net.named_modules() if name])
# ['layer1', 'layer3', 'layer4', 'layer4.0', 'layer4.1', 'layer4.2', 'layer4.3']
```

**2. Build one extractor per input type.** Pick the wrapper from the [class map](../03_class_map.md#extractor-one-per-input-type).

```python
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

vision = PytorchWrapper(identifier='my-cnn', model=net, preprocessing=preprocess)
```

`preprocess` takes a list of image paths and returns the batch the network expects. Use the same preprocessing the model was trained with.

**3. Propose a region-to-layer map.** Each region becomes a `neural:<region>` output.

```python
region_layer_map = {'V1': 'layer1', 'V4': 'layer3', 'IT': 'layer4'}
```

This map is a hypothesis. To choose layers from data, run the layer mapping explorer (`brainscore_vision.tools.layer_mapping_explorer`), which scores each layer on standard region benchmarks. The layer it picks scores best on those benchmarks because it was picked there, so report the model's score on data that did not take part in the choice.

**4. Assemble.**

```python
from brainscore import BrainScoreModel

model = BrainScoreModel(
    identifier='my-cnn',
    model=net,                                   # the module interventions act on
    extractor={'vision': vision},
    region_layer_map=region_layer_map,
    behavioral_readout_layer='layer4',           # optional: behavior by readout
)
print(sorted(model.out_channels))
# ['behavior', 'behavior:choice', 'behavior:probabilities', 'behavior:rating', 'behavior:rt',
#  'neural:IT', 'neural:V1', 'neural:V4']
```

Pass the same `net` object to the extractor and to `model=`, so recording and interventions act on the same network.

Layer paths are relative to the module the extractor wraps (`extractor.layer_names()` lists them). A path the extractor does not have fails when you build the model, and `start_recording` rejects a name that is neither a mapped region nor a layer; both errors list close matches.

A session that requests several regions gives each `neural:<region>` channel the same payload as a session for that region alone. When the regions are recorded by one extractor that declares `independent_layers = True`, the extractor runs once for all of them (once per row in a streamed session), and each channel gets its own layers. Every wrapper in `brainscore.model_helpers` declares it (`TextWrapper`, `VLMVisionWrapper`, `VLMVideoWrapper`, `VideoWrapper`, `AudioWrapper`). Any other extractor runs once per region. Declare it on your own extractor only if its output for some layers, cut to one layer, is exactly its output for that layer alone (values, coordinates, attributes): for example, number the units of each layer from zero (`neuroid_num`). The environment variable `BRAINSCORE_REGIONS_PER_PASS=one` gives every region its own pass (see `EXTENDING.md`).

A region is also recorded while the model answers a task or acts in a closed loop. The extractor's `record(layers, stimuli)` records the forward pass that another part (`reply`, `likelihood`, `answer`, `policy`) runs on `stimuli` (None for a policy step). Its `take(row)` gives one assembly per layer, made by the same functions that make `__call__`'s output, so each layer keeps its dtype. Every wrapper in `brainscore.model_helpers` has it, and each wrapper's `__call__` runs its own forward pass through the same `LiveRecording`. Your own extractor should have `record` too; without it, a region is recorded during a task only through hooks on the network passed as `model=` (every layer of the region must be a module of that network), and otherwise the pair is N/A before any compute.

A recording during a task equals the passive recording of the same stimuli, or it is refused with a clear error. What `stimuli` decide is refused before any forward pass: `AudioWrapper` clips of different lengths (the passive call pads them to the longest) or longer than `max_duration_sec` (the passive call cuts them into chunks); `VideoWrapper` clips that `context_window_ms` tiles into several windows, clips over `max_clip_ms`, clips of different clocks or numbers of frames; `VLMVideoWrapper` clips of different numbers of time steps; a video recording during a policy step (no clip, so no clock). What the pass decides is refused in its forward pre-hook, before it computes: a pass with several prompts or clips, and a later pass of the same presentation with other input (a second chunk or window). A generating model is recorded at its first pass, the prompt; the later passes see only generated tokens.

## Try it on your own images

A stimulus set is a table with one row per stimulus: a `stimulus_id` and an `image_path` column holding each file's path.

```python
import pandas as pd
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.streaming_helpers import neural_response

stimuli = StimulusSet(pd.DataFrame({
    'stimulus_id': ['cat', 'dog'],
    'image_path': [image_paths['cat'], image_paths['dog']],   # the paths of your files
}))
stimuli.identifier = 'my-images'          # names the activation cache entry

activations = neural_response(model, stimuli, ['neural:IT'])
print(activations.dims, activations.shape)   # ('presentation', 'neuroid') (2, 8)
```

`neural_response` runs a session through `model.participate`, as a benchmark does. In a
notebook, `model.start_recording('IT')` followed by `model.process(stimuli)` gives the same
assembly; `process` is a convenience of `BrainScoreModel`, not part of the contract.

The column name tells the model what kind of input each row holds (`sentence` for text, `audio_path` for audio). A `stimulus_paths` dictionary alone is not enough.

## Models with several inputs

Give each input type its own extractor. If two extractors use the same layer names, route regions with `LayerSelector`. This example needs a network with an image part and a text part, such as CLIP (`clip`, with one wrapper for each part):

```python
from brainscore import LayerSelector

model = BrainScoreModel(
    identifier='my-clip',
    model=clip,
    extractor={'vision': image_wrapper, 'text': text_wrapper},
    region_layer_map={
        'IT': LayerSelector(name='encoder.layers.10', modality='vision'),
        'language_system': LayerSelector(name='encoder.layers.10', modality='text'),
    },
)
```

The [CLIP registration](../../brainscore/models/clip_vit_b_32/model.py) is a complete example.

**Stills and clips.** The `extractor` keys declare the visual forms a model takes natively: a `'vision'` key takes pictures (`vision:image`) and a `'video'` key takes clips (`vision:video`). A model that takes both, such as a vision-language model, registers one of each; each stimulus then goes to the matching one. Never wrap a picture as a one-frame video for a model that takes pictures. The [Qwen3.6 registration](../../brainscore/models/qwen36_27b/model.py) registers both.

## Models without accessible layers

A hosted model or any system that only returns answers has no extractor. Declare its input and give it the parts that answer: a `reply` for tasks with a response (choices, ratings, written answers) and, for older `task_type`/`label_set` tasks, an `answer`:

```python
def my_reply(prompt):                                      # calls your API in a real model
    return 'real'


def my_answer(*, stimulus_row, instruction, label_set):   # likewise
    return label_set[0]


model = BrainScoreModel(identifier='my-api-model', in_channels={'text'},
                        reply=my_reply, answer=my_answer)
print(sorted(model.out_channels))
# ['behavior', 'behavior:choice', 'behavior:confidence', 'behavior:rating', 'behavior:text']
```

For a hosted API, `build_api_reply(provider, model_id)` (a `reply`) and `build_api_generation_fn(provider, model_id)` (an `answer`) build both from one client per provider ([Behavior](behavior.md)).

It can take part in behavioral benchmarks but not neural ones; neural benchmarks will report N/A.

## Register it (optional)

You can always pass the object directly to `brainscore.score`. To load it by name, copy [templates/new_model](../../templates/new_model), implement the factory and its test, and register it in `brainscore/models`.
