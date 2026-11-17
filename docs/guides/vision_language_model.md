# Register a vision-language model with the builder

`build_vlm_subject` loads a Hugging Face model with a vision part, a projector and a
causal language model, and gives a subject: pictures (and clips) recorded from the
vision part, text from the language model, written replies, option likelihoods and a
fitted readout. The regions come from the stated depth hypothesis
(`DEPTH_TABLE` in `brainscore/tools/auto_register.py`, asked for by the builder), not from a layer search;
each score's provenance says so.

What a model from the builder gives (the parts are wired by default; the registration does not list them):

| Output | Part | Needs |
|---|---|---|
| `neural:V1`, `V2`, `V4`, `IT`, `VWFA` | the vision part (VWFA at the IT layer) | pictures; clips with `video` |
| `neural:language_system` | the language model | text; pictures and clips in an answering pass |
| `neural:A1` | the audio encoder | `audio` |
| `behavior:choice`, `rating`, `text` | the reply (writing) | |
| `behavior:log_probabilities` | the likelihood of each option | a verbatim choice on text |
| `behavior:probabilities`, `rt` | the readout on the IT layer | the task's fitting stimuli |
| `perturbation` (inputs `lesion`, `stimulation`, `pharmacological`) | `pytorch_intervention` over both parts | |

A family that cannot take a part says so with one argument: `intervention=False` when its
vision and language block paths start with the same name (the builder refuses with this advice).
`motor` (a policy for an environment) is not given.

A model plugin has the usual three files in `brainscore/models/<name>/`. The builder
is only a helper that `get_model` calls: a `get_model` that builds a `BrainScoreModel`
by hand registers the same way ([Wrap your model](wrap_your_model.md)).

`__init__.py`: the registry line, with a lazy import so that `import brainscore` stays light.

```python
from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('qwen3-vl-4b')


model_registry['qwen3-vl-4b'] = _load
```

`model.py`: `get_model(identifier)` builds the subject. Fill the values:

```python
from brainscore.model_helpers.vlm import build_vlm_subject


def get_model(identifier: str):
    assert identifier == 'qwen3-vl-4b'
    return build_vlm_subject(
        identifier, 'Qwen/Qwen3-VL-4B-Instruct',         # identifier, Hub identifier
        'Qwen3VLForConditionalGeneration',               # the transformers class
        vision=('model.visual', 'blocks'),               # module, then its block list
        language=('model.language_model', 'layers'),
        dtype='bfloat16',                                # the precision of every new model
        download_gb=8.9,                                 # the size of the Hub files
        video=True,                                      # the processor takes clips natively
        probabilities='readout',                         # or ('likelihood', 'softmax')
    )
```

`test.py`: the plugin's own test with the real weights, marked `slow`, as in
`brainscore/models/qwen3_vl_2b/test.py` (declarations, pictures, text, a choice, the readout).

Several sizes of one family that share their code go in one folder, with one registry
line each and `get_model(identifier)` choosing the values, as `qwen25_vl_3b` does for
its 3B and 7B models; a model with other code gets its own folder.

## Find each value

1. **Hub identifier**: the instruction-tuned checkpoint (`-Instruct`, `-it`; `-hf` for a
   repository converted to the transformers classes). Pass `revision=` (the commit on the
   Hub page) to score one fixed set of weights.
2. **Class**: `architectures` in the Hub `config.json`. Check that the pinned transformers
   has it: `python -c "import transformers; transformers.Qwen3VLForConditionalGeneration"`.
   A mixture-of-experts model has its own class; its paths are the same.
3. **`vision`, `language`**: build the class from its config on the meta device (no weights)
   and check that each path ends in the list of repeated blocks:

```python
import torch
import transformers

config = transformers.AutoConfig.from_pretrained('Qwen/Qwen3-VL-4B-Instruct')   # config.json only
with torch.device('meta'):
    network = transformers.Qwen3VLForConditionalGeneration(config)
print(network.get_submodule('model.visual.blocks')[0])          # one vision block
print(len(network.get_submodule('model.language_model.layers')))  # the number of decoder layers
```

4. **`download_gb`**: the sum of the `*.safetensors` files on the Hub page (the download,
   in the precision of the files; `dtype` is the precision in memory, `bfloat16`).
5. **`video`**: `True` when the processor has a `video_processor` that gives
   `pixel_values_videos` and `video_grid_thw`, and the vision part takes `grid_thw` (Qwen-VL,
   GLM-4.1V). Clips then also reach the language layer.
6. **`image_processor_kwargs`**: only for a vision part that takes `(B, C, H, W)` (its processor
   gives no `image_grid_thw`): the settings that give one whole image per picture, for
   example `{'do_image_splitting': False}` (SmolVLM), `{'do_pan_and_scan': False}` (Gemma 3),
   `{'crop_to_patches': False}` (InternVL). The builder refuses a processor that still gives
   tiles. Each layer is reduced to 1000 components (`LayerPCA`), fitted once on ImageNet
   validation images (`MT_IMAGENET_PATH`).
7. **`device_map`**: `'auto'` when the weights (2 bytes per parameter) take more than about
   two thirds of one accelerator. Each extractor keeps the loaded placement; a vision part
   given `image_processor_kwargs` must stay on one device.
8. **`probabilities`**: `'readout'` for new models (a classifier on the IT layer, fitted on the
   task's fitting stimuli); `('likelihood', 'softmax')` takes them from the log probabilities
   of the options, with no fitting.
9. **`audio`**: no extractor for an audio encoder inside a vision-language model exists yet;
   leave it out.

## Checks

Offline, with a stand-in model (no download):

```bash
pytest tests/test_vlm_builder.py -m "not slow"
```

Its loop over the registry builds every registration made with the builder on a stand-in.
On a GPU machine, run the plugin's own test, then one neural and one behavioral
benchmark before a full run:

```bash
pytest brainscore/models/<name>/test.py -m slow
```
