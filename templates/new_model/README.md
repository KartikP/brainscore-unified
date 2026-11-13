# Template: a new model

A model is one `BrainScoreModel` construction. See `EXTENDING.md` (Seam 1) for the contract and
the architecture map at https://brain-score.github.io/public/UMI/architecture.html for which channels each optional part adds.

## Use it

1. Copy this folder to `brainscore/models/<your_name>/`.
2. Edit `model.py`: load your backbone, configure its wrapper, and pass `extractor={'vision': wrapper}` (or `text`/`audio`). Set `region_layer_map` to the layers you will record.
3. Edit `__init__.py`. Pick your identifier in `model_registry['<your-id>']`.
4. Add `from . import <your_name>` to `brainscore/models/__init__.py`.
5. Adapt `test_model.py` and run it.

Before you copy anything, try the template from the `templates/` folder. Importing a
template registers it, and the benchmark template scores the model template offline:

```python
import brainscore
import new_model, new_benchmark      # registers 'your-model' and 'your-benchmark'

model = brainscore.load_model('your-model')
score = brainscore.score('your-model', 'your-benchmark')
print(model.identifier, sorted(model.out_channels))   # your-model ['neural:IT']
```

The score is a number near zero, because the template network has random weights.
After the copy, use your own identifier and a registered benchmark, for example
`brainscore.score('<your-id>', 'MajajHong2015public.IT-pls')` (which downloads its data).

## Shortcut

For most standard models, the `auto_register` tool infers the wrapper, layers, and a provisional
`region_layer_map` for you. Start there, then refine the layer map with the layer-mapping explorer.

## Worked references (read these)

- `brainscore/models/clip_vit_b_32/`: vision + text VLM, behavioral readout.
- `brainscore/models/vjepa_v1/`: native-temporal video (VideoWrapper, vendored backbone).
- `brainscore/models/api_closed/`: output-only API model (`answer` and `in_channels`; no extraction wrapper).
- `brainscore/models/multimodal_av_vjepa_wav2vec2/`: two extraction wrappers with modality routing in their region selectors.
