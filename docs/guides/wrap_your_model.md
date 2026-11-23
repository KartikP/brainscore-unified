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


# Stand-in network: three named layers
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(3, 12, 3)  # 3 channels in, 12 out, 3 x 3 kernel
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


# Two random 16 x 16 images: cat.png, dog.png
workspace = tempfile.TemporaryDirectory()      # removed when the script ends
folder = workspace.name
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

Limit: `PytorchWrapper` keeps one value for each module, so for a module that runs more than once for one input it keeps only the last call, without an error, and it refuses a `<layer>-t<n>` or `:input` name when you build the model. For such a module, use a wrapper of `brainscore.model_helpers`, or name a module that runs once.

For a Hugging Face vision model, use `VLMVisionWrapper(model=<hf vision model>, processor=<image processor>)` from `brainscore.model_helpers.vlm_vision_wrapper`. `PytorchWrapper` reads float32 outputs only (a bfloat16 model fails) and keeps the last call of a repeated layer; `VLMVisionWrapper` reads bfloat16 and float16 outputs, and for a repeated layer it records the call you name as `<layer>-t<n>`. Load the image processor with `use_fast=False`; for a processor that holds an image processor, `pin_image_processor(processor, model_id)` (from `brainscore_core.hf_compat`) does this, so the preprocessed pixels do not change with the transformers version.

**3. Propose a region-to-layer map.** Each region becomes a `neural:<region>` output.

```python
region_layer_map = {'V1': 'layer1', 'V4': 'layer3', 'IT': 'layer4'}
```

This map is a hypothesis. To choose layers from data, score each layer against measured responses. For a model in the vision registry, the vision layer mapping explorer (`brainscore_vision.tools.layer_mapping_explorer`) loads the model by its registry identifier and scores each layer on standard region benchmarks. For a model wrapped here and not registered, `brainscore.tools.layer_mapping.sweep_model(wrapper, stimuli, target, target_stimulus_ids, layers)` scores each layer against a response array that you give; it runs no benchmark. The layer either tool picks scores best on the data it was picked on, so report the model's score on data that did not take part in the choice.

**4. Assemble.**

```python
from brainscore import BrainScoreModel

model = BrainScoreModel(
    identifier='my-cnn',
    model=net,                                   # the module interventions act on
    extractor={'vision': vision},
    region_layer_map=region_layer_map,
    behavioral_readout_layer='layer4',           # optional: behavior by readout
    # The readout gives probabilities; its choice is the most probable option.
    # State that conversion for a model that must answer a choice task:
    answer_owners={'behavior:choice': ('readout', 'argmax')},
)
print(sorted(model.out_channels))                # the declared outputs
# ['behavior', 'behavior:choice', 'behavior:probabilities', 'behavior:rating', 'behavior:rt',
#  'neural:IT', 'neural:V1', 'neural:V4']

# Without that line, 'behavior:choice' is not in the list.
```

Pass the same `net` object to the extractor and to `model=`, so recording and interventions act on the same network.

Layer paths are relative to the module the extractor wraps (`extractor.layer_names()` lists them). A path the extractor does not have fails when you build the model, with close matches; a session that requests a region the map does not name is refused before any compute.

When a module runs more than once for one input (a looped block), name one of its calls as `<layer>-t<n>`, counted from 0, as the old vision time steps: `'layers.6-t1'` is the second call of `layers.6`. The wrappers in `brainscore.model_helpers` read this form; they refuse the plain name of such a module and a call number at or above the count.

A session that requests several regions gives each `neural:<region>` channel the same payload as a session for that region alone. When the regions are recorded by one extractor that declares `independent_layers = True`, the extractor runs once for all of them (once per row in a streamed session), and each channel gets its own layers. Every wrapper in `brainscore.model_helpers` declares it (`TextWrapper`, `VLMVisionWrapper`, `VLMVideoWrapper`, `VideoWrapper`, `AudioWrapper`). Any other extractor runs once per region. Declare it on your own extractor only if its output for some layers, cut to one layer, is exactly its output for that layer alone (values, coordinates, attributes): for example, number the units of each layer from zero (`neuroid_num`). The environment variable `BRAINSCORE_REGIONS_PER_PASS=one` gives every region its own pass (see `EXTENDING.md`).

A region is also recorded while the model answers a task or acts in a closed loop. The extractor's `record(layers, stimuli)` records the forward pass that another part (`reply`, `likelihood`, `answer`, `policy`) runs on `stimuli` (None for a policy step). Its `take(row)` gives one assembly per layer, made by the same functions that make `__call__`'s output, so each layer keeps its dtype. Every wrapper in `brainscore.model_helpers` has it, and each wrapper's `__call__` runs its own forward pass through the same `LiveRecording`. Your own extractor should have `record` too; without it, a region is recorded during a task only through hooks on the network passed as `model=` (every layer of the region must be a module of that network), and otherwise the session is refused before any compute: `score` raises `ExecutionNotFeasibleError`, because the declarations do not express this limit.

A recording during a task uses the same reduction and packaging as the passive recording, so the two are equal when the network receives the same input; otherwise it is refused with a clear error. A model whose answering part prepares the stimulus in its own way (a chat processor) records what the network received during the task, which can differ from what the extractor prepares for the passive recording. What `stimuli` decide is refused before any forward pass: `AudioWrapper` clips of different lengths (the passive call pads them to the longest) or longer than `max_duration_sec` (the passive call cuts them into chunks); `VideoWrapper` clips that `context_window_ms` tiles into several windows, clips over `max_clip_ms`, clips of different clocks or numbers of frames; `VLMVideoWrapper` clips of different numbers of time steps; a video recording during a policy step (no clip, so no clock). What the pass decides is refused in its forward pre-hook, before it computes: a pass with several prompts or clips, and a later pass of the same presentation with other input (a second chunk or window). A generating model is recorded at its first pass, the prompt; the later passes see only generated tokens.

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

# Show the images, record IT
activations = neural_response(model, stimuli, ['neural:IT'])
print(activations.dims, activations.shape)   # ('presentation', 'neuroid') (2, 8)
```

`neural_response` runs a session through `model.participate`, as a benchmark does.

The column name tells the model what kind of input each row holds (`sentence` for text, `audio_path` for audio). A `stimulus_paths` dictionary alone is not enough. A wrapper takes only the column names that the core catalog binds to its input (`brainscore_core.io_catalog.stimulus_columns`), tried in the catalog's order: `audio_path`, `audio_file_name`, `audio_file`, `audio` for sound; `video_path`, `video_file_name`, `video` for clips; `image_file_name`, `image_path`, `filename`, `image`, `vision` for images.

## Models with several inputs

Give each input type its own extractor. An extractor that cannot list its layers (a plain callable extractor(stimuli, layers)) counts as having every layer, so next to another extractor every region names its input with LayerSelector(name=..., modality=...). If two extractors use the same layer names, route regions with `LayerSelector`. This example needs a network with an image part and a text part, such as CLIP (`clip`, with one wrapper for each part):

```python
from brainscore import LayerSelector

model = BrainScoreModel(
    identifier='my-clip',
    model=clip,
    extractor={'vision': image_wrapper, 'text': text_wrapper},
    region_layer_map={
        # Same layer name in both parts: name the part
        'IT': LayerSelector(name='encoder.layers.10', modality='vision'),
        'language_system': LayerSelector(name='encoder.layers.10', modality='text'),
    },
)
```

The [CLIP registration](../../brainscore/models/clip_vit_b_32/model.py) is a complete example.

In a network where a picture also reaches a text part's layer (the picture tokens of Qwen-VL enter its language model), name the recording extractor first and the other inputs after it, `LayerSelector(name='layers.28', modality=('text', 'vision'))`: the region is then recorded from a picture during an answer, and a passive picture session is refused before compute. This is your claim about the wiring, and it is not checked before a pass: a region stated as reached from an input that never reaches it fails at the first pass with `No forward pass reached ...`.

A text part is wrapped with `TextWrapper(model, tokenizer)`. The tokenizer is a Hugging Face
tokenizer, or any callable with the same call: it takes a list of texts and `padding`,
`truncation`, `max_length` and `return_tensors='pt'`, and returns a mapping whose keys are the
network's forward arguments (`input_ids`, and `attention_mask` to mark padding). The
wrapper calls the network with that mapping. A text longer than the limit keeps its newest
tokens. The limit is the model's own window: the smaller of the tokenizer's `model_max_length`
and the `max_position_embeddings` of the network's configuration. A plain callable states
neither, so state `max_length` on the wrapper. Here a tiny network with an image part and a
text part, next to the image network above:

```python
from brainscore import LayerSelector
from brainscore.model_helpers.text_wrapper import TextWrapper


# Stand-in text network
class TextNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(128, 8)
        self.encoder = torch.nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask=None):
        return self.encoder(self.embed(input_ids))


# Stand-in tokenizer: characters as token ids
def tokenizer(texts, max_length, padding=True, truncation=True, return_tensors='pt', **kwargs):
    """Characters as token ids, padded with 0 to the longest text; a longer text keeps its newest
    max_length characters, as the wrapper asks of a real tokenizer."""
    ids = [[ord(character) % 128 for character in text][-max_length:] for text in texts]
    width = max(len(row) for row in ids)
    return {'input_ids': torch.tensor([row + [0] * (width - len(row)) for row in ids]),
            'attention_mask': torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in ids])}


# One model, one extractor per input type
text_net = TextNet().eval()
both = BrainScoreModel(
    identifier='my-cnn-and-text',
    extractor={'vision': vision,
               'text': TextWrapper(model=text_net, tokenizer=tokenizer, layer_aggregation='mean',
                                   max_length=512)},   # the limit, stated once
    region_layer_map={'IT': LayerSelector(name='layer4', modality='vision'),
                      'language_system': LayerSelector(name='encoder', modality='text')},
)
print(sorted(both.in_channels), sorted(c for c in both.out_channels if c.startswith('neural')))
# ['control', 'text', 'vision:image'] ['neural:IT', 'neural:language_system']
```

**Stills and clips.** The `extractor` keys declare the visual forms a model takes natively: a `'vision'` key takes pictures (`vision:image`) and a `'video'` key takes clips (`vision:video`). A model that takes both, such as a vision-language model, registers one of each; each stimulus then goes to the matching one. Never wrap a picture as a one-frame video for a model that takes pictures. The [Qwen3.6 registration](../../brainscore/models/qwen36_27b/model.py) registers both. It needs transformers 5 or later and is not loadable in the reference environment (transformers 4.57.6).

## Models without accessible layers

A hosted model or any system that only returns answers has no extractor. Declare its input and give it the parts that answer: a `reply` for tasks with a response (choices, ratings, written answers) and an `answer` for a plain choice. Both return the text the model generated, exactly; Brain-Score reads the choice from it with core's `Choice.parse` and stores the text as the raw answer. A function that returns a label also works, because a label is a text that reads as itself:

```python
def my_reply(prompt):                                      # calls your API in a real model
    return 'real'


def my_answer(*, stimulus_row, instruction, label_set):   # likewise
    return label_set[0]


# No extractor: inputs declared directly
model = BrainScoreModel(identifier='my-api-model', in_channels={'text'},
                        reply=my_reply, answer=my_answer)
print(sorted(model.out_channels))
# ['behavior', 'behavior:choice', 'behavior:rating', 'behavior:text']
```

For a hosted API, `build_api_reply(provider, model_id)` (a `reply`) and `api_answer(provider, model_id)` (an `answer`) build both from one client per provider ([Behavior](behavior.md)).

It can take part in behavioral benchmarks but not neural ones; neural benchmarks will report N/A.

## A subject of your own

A system with its own loop subclasses `Subject` ([Lesson 1](../../notebooks/lessons/01_your_first_subject.ipynb)). Declare `identifier`, `in_channels` and `out_channels` with class attributes, or with properties when a declaration depends on a constructor argument. An assignment such as `self.out_channels = ...` in `__init__` does not work: these names are abstract properties of `Subject`, so the class must define them.

A subject that answers one trial with one cell (a string, a 1-D array or one number on `behavior:<format>`) names two keys in the event's `meta`: `stimulus_id`, the row it answers, in a table session; and `units` (`'ms'` or `'monotone'`) for a `behavior:rt` cell.

```python
import pandas as pd

from brainscore_core import Subject
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


# Subject: first option, fixed reaction time
class FirstOption(Subject):
    """Answers each word with the first option; reports a fixed time when asked."""
    identifier = 'first-option'
    in_channels = {'text'}                           # a class attribute

    def __init__(self, outputs=('behavior:choice',)):
        self._outputs = set(outputs)

    @property
    def out_channels(self):                          # a property: it depends on the argument
        return self._outputs

    def participate(self, session):
        options = session.task.response.labels
        wanted = session.outputs or ()
        while (event := session.next_input()) is not None:
            trial = {'stimulus_id': event.meta['stimulus_id']}  # the row it answers
            if 'behavior:choice' in wanted:
                session.respond(StreamEvent('behavior:choice', options[0], t_ms=None, meta=trial))
            if 'behavior:rt' in wanted:
                session.respond(StreamEvent('behavior:rt', 450.0, t_ms=None,
                                            meta={**trial, 'units': 'ms'}))


# Two words, a choice task, one session
words = StimulusSet(pd.DataFrame({'stimulus_id': ['w0', 'w1'], 'text': ['cat', 'blick']}))
words.identifier = 'two-words'
task = TaskContext(instruction='Is this a word?',
                   response=Choice(['real', 'pseudo'], reaction_time='surprisal'))
session = stimulus_session(words, ['behavior:choice', 'behavior:rt'], task)
FirstOption(outputs=['behavior:choice', 'behavior:rt']).participate(session)
print(session.collect('behavior:choice').values.tolist())   # [[1.0, 0.0], [1.0, 0.0]]
print(session.collect('behavior:rt').values.tolist())       # [450.0, 450.0]
```

## Which benchmarks fit

`check_compatibility(model, benchmark)` (from `brainscore_core`) compares the declarations only and calls no method of the model. It returns the match (its `kind` is `'full'`, or `'partial'` when the model does not take some of the inputs) or raises `CompatibilityError` with the reason. `brainscore.list_benchmarks()` lists the identifiers; `brainscore.load_benchmark(identifier)` builds one, and building a benchmark can download its data. `brainscore.score` makes the same check first and returns N/A for a pair that does not fit.

## Register it (optional)

You can always pass the object directly to `brainscore.score`. To load it by name, register a factory. In the Brain-Score repository: copy [templates/new_model](../../templates/new_model) into `brainscore/models`, and implement the factory and its test. In your own package: add the factory to `brainscore.model_registry` when the package is imported, and import the package before you load the model ([Your own package](your_own_package.md)).

A model plugin is one folder in `brainscore/models`: `__init__.py` registers each identifier (`model_registry['my-model'] = ...`), `model.py` builds the model, and `test.py` loads each identifier with `load_model` and compares the identifier. The plugin checks also run a generic suite, `brainscore/model_helpers/generic_plugin_tests.py`, once for each identifier registered in `__init__.py`, after a real load. It loads the model once and checks three things: the identifier is the registered one; the input and output channels are not empty; the whole declaration passes core's pre-flight (`brainscore_core.compatibility.check_compatibility`) for an experiment that sends the model's own inputs and requests its own outputs. The pre-flight reads the channels, the required channels, the inputs each output answers from, and the traits. It sends no input to the model, so it does not show that a declared output works: put that test in the plugin's own `test.py`. You write no extra file for the generic suite.

## A person as the subject

A person is a subject too. `HumanSubject` (`brainscore.harnesses.human`) declares the behavior formats that its rig gives, for example `out_channels={'behavior:choice'}`, and calls your `respond_fn(task, stimuli)` for each session. What it returns is the answer on that channel. A benchmark that requests a format the rig did not state gets N/A before any stimulus is shown. The class does not drive hardware: `present_fn`, `record_fn` and `respond_fn` are yours.
