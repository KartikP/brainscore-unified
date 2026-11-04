# Existing Brain-Score models

Models and benchmarks written for classic Brain-Score keep working. Nothing needs to be rewritten.

```python
import brainscore

model = brainscore.load_model('alexnet')                       # a classic vision plugin
score = brainscore.score('alexnet', 'MajajHong2015.IT-pls')    # a classic benchmark
```

## How this works

- `load_model` looks in the unified registry first, then in the vision and language registries. A classic vision `BrainModel` is wrapped in `VisionModelAdapter`; a classic language `ArtificialSubject` in `LanguageModelAdapter`. Both present the model as a subject with the right channels.
- A classic benchmark calls methods such as `look_at` or `digest_text`. When it is scored, it receives a facade that turns those calls into sessions for the subject's `participate`. Any subject can therefore run a classic benchmark, including a `BrainScoreModel` or a hand-written `Subject`. A classic vision benchmark also needs `visual_degrees()` from the subject; `BrainScoreModel` provides it.

Adapters and the facade are permanent parts of Brain-Score, not a temporary migration path.

## Translating a registration

Translate only if you want features the classic interface lacks (several input types, perturbations, closed-loop control). The parts map one to one:

| Classic | `BrainScoreModel` |
| --- | --- |
| `PytorchWrapper(identifier, model, preprocessing)` | `extractor={'vision': VisionWrapper(identifier, model, preprocessing)}` |
| `ModelCommitment(..., region_layer_map={'IT': 'layer4'})` | `region_layer_map={'IT': 'layer4'}` |
| `behavioral_readout_layer='fc'` | `behavioral_readout_layer='fc'` |
| `visual_degrees=8` | `visual_degrees=8` |
| `HuggingfaceSubject(model_id, region_layer_mapping)` | `extractor={'text': TextWrapper(model, tokenizer)}` plus `region_layer_map` |
| `look_at(stimuli)`, `digest_text(text)` | `participate(session)` |

[Lesson 4](../../notebooks/lessons/04_bring_an_existing_model.ipynb) does this translation and checks that both versions record the same activity.
