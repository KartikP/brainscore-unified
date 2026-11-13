# Wrap your model

Choose one of two routes.

| Your system | Route |
| --- | --- |
| A neural network whose layers you can access | `BrainScoreModel`, assembled from parts (below) |
| Anything with its own loop: a remote API, a simulator agent, a rule | Subclass `Subject` and write `participate` ([Lesson 1](../../notebooks/lessons/01_your_first_subject.ipynb)) |

## BrainScoreModel in four steps

**1. Find the layer paths.**

```python
[name for name, _ in net.named_modules() if name]
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

This map is a hypothesis. To choose layers from data, run the layer mapping explorer (`brainscore_vision.tools.layer_mapping_explorer`), which scores each layer on standard region benchmarks.

**4. Assemble.**

```python
from brainscore import BrainScoreModel

model = BrainScoreModel(
    identifier='my-cnn',
    model=net,                                   # the module perturbations act on
    extractor={'vision': vision},
    region_layer_map=region_layer_map,
    behavioral_readout_layer='layer4',           # optional: behavior by readout
)
```

Pass the same `net` object to the extractor and to `model=`, so recording and perturbation act on the same network.

Layer paths are relative to the module the extractor wraps (`extractor.layer_names()` lists them). A path the extractor does not have fails when you build the model, and `start_recording` rejects a name that is neither a mapped region nor a layer; both errors list close matches.

## Try it on your own images

A stimulus set is a table with one row per stimulus: a `stimulus_id` and an `image_path` column holding each file's path.

```python
import pandas as pd
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

stimuli = StimulusSet(pd.DataFrame({
    'stimulus_id': ['cat', 'dog'],
    'image_path': ['/path/to/cat.png', '/path/to/dog.png'],
}))
stimuli.identifier = 'my-images'          # names the activation cache entry

model.start_recording('IT')
activations = model.process(stimuli)      # (presentation, neuroid)
```

The column name tells the model what kind of input each row holds (`sentence` for text, `audio_path` for audio). A `stimulus_paths` dictionary alone is not enough.

## Models with several inputs

Give each tower its own extractor. If two towers use the same layer names, route regions with `LayerSelector`:

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

A hosted model or any system that only returns answers has no extractor. Declare its input and give it an `answer`:

```python
model = BrainScoreModel(identifier='my-api-model', supported_modalities={'text'}, answer=my_answer)
```

It can take part in behavioral benchmarks but not neural ones; neural benchmarks will report N/A.

## Register it (optional)

You can always pass the object directly to `brainscore.score`. To load it by name, copy [templates/new_model](../../templates/new_model), implement the factory and its test, and register it in `brainscore/models`.
