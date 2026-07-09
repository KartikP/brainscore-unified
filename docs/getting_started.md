# Getting started with Brain-Score UMI

This guide assumes the four-repository distribution was installed with the
root setup.sh and that the brainscore-unified environment is active.

For a download-free first result, run
notebooks/01_quickstart_layer_mapping.ipynb. The registered-model example below
can download CLIP weights and public benchmark data on its first run.

## First registered score

The public run loop is load_model, load_benchmark, then score:

~~~python
import brainscore
model_id = "clip-vit-b-32"
benchmark_id = "MajajHong2015public.IT-pls-unified"
model = brainscore.load_model(model_id)
benchmark = brainscore.load_benchmark(benchmark_id)
print(model.identifier)
print(benchmark.identifier)
score = brainscore.score(model_id, benchmark_id)
print(float(score))
~~~

The explicit load calls make missing registrations fail before the potentially
long score. brainscore.score currently accepts identifiers, not ad-hoc objects.

## Choose the extraction wrapper

There is no universally best recording layer. The last block is a reasonable
smoke-test target, but a scientific registration should compare candidate
layers with the layer-mapping tools and commit the selected region mapping.

| Model input | Wrapper | Import | Provisional layer guidance |
| --- | --- | --- | --- |
| Standard image CNN or ViT | PytorchWrapper | from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper | Start with named late blocks; map V1/V2/V4/IT empirically |
| Encoder or causal text model | TextWrapper | from brainscore.model_helpers.text_wrapper import TextWrapper | Final transformer block; use mean_tokens for encoders and last_token for causal models |
| Flattened-patch VLM vision tower | VLMVisionWrapper | from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper | A late visual block after confirming patch aggregation |
| Native temporal video model | VideoWrapper | from brainscore.model_helpers.video_wrapper import VideoWrapper | A late temporal block with its output time axis verified |
| Audio encoder | AudioWrapper | from brainscore.model_helpers.audio_wrapper import AudioWrapper | A late encoder block; choose mean_time or time_series for the benchmark |
| Closed-loop policy | PolicyWrapper | from brainscore.model_helpers.policy_wrapper import PolicyWrapper | No neural layer unless activation capture is configured separately |

## Understand preprocessors

The current model constructor has two extraction patterns:

- Vision commonly uses a bare image preprocessing callable in
  preprocessors["vision"] and a PytorchWrapper in activations_model.
- Text, audio, video, and flattened-patch VLM paths place a complete wrapper
  object in preprocessors. A tokenizer or feature extractor alone is not a
  UMI preprocessor.

Use the complete CLIP example in the distribution root README or
brainscore/models/clip_vit_b_32/model.py as the reference multimodal pattern.

## Register a model

Copy templates/new_model into brainscore/models/<your_name>, then:

1. Load the backbone and processor lazily inside get_model.
2. Choose the wrapper from the table above.
3. Declare region_layer_map and preprocessors.
4. Register the factory in the model package __init__.py.
5. Import the new package from brainscore/models/__init__.py.
6. Adapt and run the template test.

Continue in [EXTENDING.md](../EXTENDING.md).

## Notebook path

The [notebook manifest](../notebooks/README.md) identifies the recommended
order, runtime, prerequisites, and whether each result is local, illustrative,
structural, or EC2-only.
