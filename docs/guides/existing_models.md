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

## When a model will not load

A listed model may still fail to load in your environment. `load_model` then raises `brainscore.LoadError` with the reason and a next step:

| Reason | What to do |
| --- | --- |
| Its weights (or a benchmark's data) are no longer at their recorded address | Nothing on your side; report it to a maintainer with the message. |
| Its requirements conflict with the packages this environment is pinned to (numpy, torch, transformers, ...) | Nothing is installed. Clone the environment and install the plugin's `requirements.txt` there, as the message shows. |
| A package it needs has no release for your platform | Try it on Linux. |
| A package it needs is not installed | Load it with `BS_INSTALL_DEPENDENCIES` unset, so its requirements are installed, or install the package yourself. |

Requirements files sometimes list Python's own modules (`imp`, `collections`); those lines are skipped.

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
