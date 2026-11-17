# Register a vision-language model with the builder

`build_vlm_subject` loads a Hugging Face model with a vision part, a projector and a causal
language model, and gives a subject with every part the model can carry: regions V1 to IT,
VWFA and language_system from the stated depth hypothesis (`DEPTH_TABLE`, no layer search),
the six behavior formats, lesions, and a policy for the grid game. The builder is only a
helper that `get_model` calls; a `get_model` that builds a `BrainScoreModel` by hand
registers the same way. What each part gives, and how to find each value:
[Wrap your model](wrap_your_model.md#vision-language-models-from-the-builder).

A plugin has three files in `brainscore/models/<name>/`. `__init__.py`, the registry line:

```python
from brainscore import model_registry


def _load():  # defer heavy imports to load time (keep `import brainscore` light)
    from .model import get_model
    return get_model('qwen3-vl-4b')


model_registry['qwen3-vl-4b'] = _load
```

`model.py`, `get_model(identifier)` with the values:

```python
from brainscore.model_helpers.vlm import build_vlm_subject


def get_model(identifier: str):
    assert identifier == 'qwen3-vl-4b'
    return build_vlm_subject(
        identifier, 'Qwen/Qwen3-VL-4B-Instruct', 'Qwen3VLForConditionalGeneration',
        vision=('model.visual', 'blocks'),               # module, then its block list
        language=('model.language_model', 'layers'),
        dtype='bfloat16',                                # every new model
        download_gb=8.9,                                 # the Hub safetensors files
        video=True,                                      # the processor takes clips natively
        probabilities='readout',                         # or ('likelihood', 'softmax')
    )
```

A vision part that takes `(B, C, H, W)` states its whole-image settings instead of `video`,
for example `image_processor_kwargs={'do_image_splitting': False}` (SmolVLM). A large model
adds `device_map='auto'`. `intervention=False` and `policy=False` leave a part out.

`test.py`, the plugin's own test with the real weights, marked `slow`: copy
`brainscore/models/qwen3_vl_2b/test.py`. Sizes of one family that share their code go in one
folder with one registry line each, as `qwen25_vl_3b` does; other code gets its own folder.

Checks: offline on a stand-in (no download), then on a GPU machine with the weights:

```bash
pytest tests/test_vlm_builder.py -m "not slow"
pytest brainscore/models/<name>/test.py -m slow
```
