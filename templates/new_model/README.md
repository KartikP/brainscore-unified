# Template: a new model

A model is one `BrainScoreModel` construction. See `EXTENDING.md` (Seam 1) for the contract and
the architecture map at https://brain-score.github.io/public/UMI/architecture.html for which dispatch branches each optional slot unlocks.

## Use it

1. Copy this folder to `brainscore/models/<your_name>/`.
2. Edit `model.py` — load your backbone, pick the matching wrapper, define `preprocessors`
   (their keys ARE `supported_modalities`), and set `region_layer_map`.
3. Edit `__init__.py` — pick your identifier in `model_registry['<your-id>']`.
4. Add `from . import <your_name>` to `brainscore/models/__init__.py`.
5. Adapt `test_model.py` and run it.

```python
import brainscore
model = brainscore.load_model('<your-id>')
score = brainscore.score('<your-id>', '<some-benchmark>')
```

## Shortcut

For most standard models, the `auto_register` tool infers the wrapper, layers, and a provisional
`region_layer_map` for you — start there, then refine the layer map with the layer-mapping explorer.

## Worked references (read these)

- `brainscore/models/clip_vit_b_32/` — vision + text VLM, behavioral readout.
- `brainscore/models/vjepa_v1/` — native-temporal video (VideoWrapper, vendored backbone).
- `brainscore/models/api_closed/` — output-only API model (`generation_fn`, `activations_model=None`).
- `brainscore/models/multimodal_av_vjepa_wav2vec2/` — two preprocessors + `region_modality_map`.
