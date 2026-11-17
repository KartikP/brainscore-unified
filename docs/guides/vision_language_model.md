# Register a vision-language model with the builder

`build_vlm_subject` loads a Hugging Face model with a vision part, a projector and a
causal language model, and gives a subject: pictures (and clips) recorded from the
vision part, text from the language model, written replies, option likelihoods and a
fitted readout. The regions come from the stated depth hypothesis
(`DEPTH_TABLE` in `brainscore/tools/auto_register.py`, asked for by the builder), not from a layer search;
each score's provenance says so.

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
`brainscore/models/qwen3_vl_2b/test.py` (declarations, one neural session, one choice).

Several sizes of one family that share their code go in one folder, with one registry
line each and `get_model(identifier)` choosing the values, as `qwen25_vl_3b` does for
its 3B and 7B models; a model with other code gets its own folder.

Find each value:

| Value | Where |
|---|---|
| class | `architectures` in the Hub `config.json` (a mixture-of-experts model has its own class; its paths stay the same) |
| `vision`, `language` | `print(model)` of the class built from its config on the meta device; the module path, then the path of the repeated blocks inside it |
| `video` | `True` only when the vision part takes the processor's flattened patches (Qwen-VL, Qwen2.5-Omni, GLM-4.1V) |
| `pixels` | for a vision part that takes `(B, C, H, W)`: the image-processor settings that give each picture whole, for example `{'do_image_splitting': False}` (SmolVLM), `{}` (Gemma 3) |
| `audio` | `(module path, block list path, extractor)` of an audio encoder |
| `device_map` | `'auto'` for a model that does not fit one accelerator |

Then run the offline checks (a stand-in model, no download):

```bash
pytest tests/test_vlm_builder.py -m "not slow"
```

`test_every_builder_registration_declares_what_its_parts_can_do` loads every
registration made with the builder on a stand-in and checks its declarations. On a
GPU machine, load the model once and run one neural and one behavioral benchmark
before you score it on all benchmarks.
