# Coming from Brain-Score

Existing vision and language models can be loaded through UMI's adapters. You do not need to rewrite a model to try the unified scoring entry point.

## Score an existing model

```python
import brainscore

score = brainscore.score('alexnet', 'MajajHong2015.IT-pls')
```

The loaders check unified, then the vision and language registries. `score` also accepts already-built model and benchmark objects.

Compatibility applies to tested model, benchmark, and environment combinations. See the [support matrix](supported_features.md) and [numerical policy](numerical_policy.md).

## Choose the interface

| Your situation | Use |
| --- | --- |
| Existing vision `BrainModel` / `ModelCommitment` | Load through UMI; the vision adapter bridges the existing methods. |
| Existing language `ArtificialSubject` | Load through UMI; the language adapter bridges `digest_text`. |
| New model or policy with its own session loop | `Subject`: declare identity/channels and implement `interact(session)`. |
| Model using shared extraction and capability helpers | `BrainScoreModel`, which extends `UnifiedModel`. |

`UnifiedModel` extends `Subject` with `process`, `start_task`, `start_recording`, and layer/modality declarations. Implement `Subject` directly when your integration only needs session interaction.

## Use shared extraction helpers

For a `BrainScoreModel`, configure a backbone, wrapper, and region mapping. Build the network once so extraction and interventions act on the same instance.

```python
from brainscore_core.model_interface import BrainScoreModel
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

# Supply your network and preprocessing function.
net = backbone()
activations = PytorchWrapper(
    identifier='my-cnn', model=net, preprocessing=preprocess,
)
model = BrainScoreModel(
    'my-cnn', model=net, activations_model=activations,
    preprocessors={'vision': preprocess},
    region_layer_map={'V4': 'layer3', 'IT': 'layer4'},
)
```

The layer names above are examples; select a mapping for your architecture and benchmark. Pass the model directly to `brainscore.score`, or register a factory if you want to load it by name.

## Add capabilities and tools

`BrainScoreModel` supports behavioral readout/generation, action, and state-change callables. Native subjects can handle session events directly. Closed-loop games remain experimental; trained robotics qualification is separate from a working action callback.

- [Getting started](getting_started.md): wrapper imports and examples.
- [Build tools and integrations](tool_authoring.md): extend UMI from your own package.
- [API reference](umi_api_reference.md): recording, context, reset, and interventions.
