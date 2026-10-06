# Getting started

UMI is a production candidate. Check the [feature reference](supported_features.md) for features, environments, and validation limits.

## Install

Use Python 3.11 and install the four repositories together. Follow the [pinned-revision installation instructions](../install/README.md#reproduce-the-reviewed-peer-revisions); the candidate packages are not published on PyPI.

## Run a first example

From the workspace containing the four repositories, with your environment activated:

```sh
python -m brainscore.doctor
python unified/examples/partner_integration.py --out /tmp/umi-first-experiment
```

Use a new output directory each time. This small synthetic example demonstrates external channels, recording, and replay without model downloads. It is an integration check, not a scientific result.

For registration and scoring, open [the layer-mapping quickstart](../notebooks/01_quickstart_layer_mapping.ipynb). To combine recording and interventions on a pretrained model, open [the ResNet-18 experiment](../notebooks/18_resnet_experiment_toolbox.ipynb). Both follow the [shared vocabulary](conventions.md).

## Score a registered model

This example can download CLIP weights and public benchmark data on its first run. See [model downloads](model_downloads.md) for the download guard.

```python
import brainscore

score = brainscore.score('clip-vit-b-32', 'MajajHong2015public.IT-pls-unified')
print(float(score))
```

You can also pass a model object: `brainscore.score(my_model, benchmark_id)`. Registration is optional for your own experiments.

## Choose your integration path

| Goal | Start here |
| --- | --- |
| Use an existing vision or language model | [Coming from Brain-Score](from_brain_score.md) |
| Build a model or policy that handles sessions directly | Implement `Subject` from `brainscore_core.model_interface`: identity, input/output channels, and `interact(session)`. See [Subject lifecycle](umi_api_reference.md#subject-lifecycle). |
| Reuse neural extraction and capability helpers | Construct `BrainScoreModel` with the wrappers below. |
| Assemble a session with recording and interventions | [Experiment toolbox](experiment_toolbox.md) |
| Add a tool, input/output type, or experiment | Use an external package and the [tool-authoring guide](tool_authoring.md). |
| Connect a DROID policy | Follow the [DROID guide](droid_integration.md). |
| Connect a robotics benchmark | Follow the [LIBERO integration walkthrough](robotics_benchmark_integration.md). |

A native `Subject` does not need `process()` or a layer map. `BrainScoreModel` provides them for model extraction and recording workflows.

## Choose an extraction wrapper

Use these when your integration needs model activity extraction:

| Input | Wrapper import |
| --- | --- |
| Images, visual towers, or video | `from brainscore.model_helpers.vision_wrapper import VisionWrapper` |
| Standard image CNN or ViT | `from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper` |
| Text | `from brainscore.model_helpers.text_wrapper import TextWrapper` |
| Flattened-patch visual tower | `from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper` |
| Temporal video | `from brainscore.model_helpers.video_wrapper import VideoWrapper` |
| Audio | `from brainscore.model_helpers.audio_wrapper import AudioWrapper` |

`VisionWrapper` selects an image, VLM, or video wrapper from the architecture. Check its choice and choose recorded layers using benchmark evidence.

For image models, `activations_model` usually holds the extraction wrapper. Text paths put a complete wrapper in `preprocessors['text']`. See the [CLIP registration](../brainscore/models/clip_vit_b_32/model.py) for a working multimodal configuration.

## Share a model or tool

For an external integration, start with [examples/partner_tool](../examples/partner_tool/README.md). Import its registration module before constructing models; no Brain-Score repository edit is required.

To contribute a model to this repository's catalog, copy [templates/new_model](../templates/new_model), implement its factory and test, and register it in `brainscore/models`. That template uses `BrainScoreModel`; its layer mappings are for neural recording. See [EXTENDING.md](../EXTENDING.md) for catalog contribution details.

## Next steps

- [Concepts](concepts.md): the main terms.
- [API reference](umi_api_reference.md): recording, sessions, and scoring.
- [Notebook guide](../notebooks/README.md): examples and their data/compute requirements.
