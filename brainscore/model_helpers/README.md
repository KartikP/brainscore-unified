# Model helpers

This folder holds the general parts that a model plugin composes into a `BrainScoreModel`.
The reader is a person who registers a model.

## What do you want to do?

| Job | Helper | What it gives |
| --- | --- | --- |
| Record a layer from text | `text_wrapper.py`: `TextWrapper` | An extractor for an encoder or causal text model, with the layer aggregation you choose. |
| Record a layer from pictures (flattened-patch form) | `vlm_vision_wrapper.py`: `VLMVisionWrapper` | An extractor for a vision encoder whose input is a patch list plus grid sizes. |
| Record a layer from clips | `video_wrapper.py`: `VideoWrapper` | An extractor for a model that takes a `(batch, T, C, H, W)` clip and gives activity over time. |
| Record a layer from clips, with a vision-language processor | `vlm_video_wrapper.py`: `VLMVideoWrapper` | An extractor that hands the whole clip to the processor and gives one time bin per temporal patch. |
| Record a layer from sound | `audio_wrapper.py`: `AudioWrapper` | An extractor for an audio backbone; `mean` or `sequence` aggregation. |
| Record with a plain PyTorch vision module | `PytorchWrapper`, in `brainscore_vision.model_helpers.activations.pytorch` | An extractor for still images; it lives in the vision package, not here. |
| Record one pass over a picture, a clip and a sentence together (the fused form) | `fused.py`: `fused_chat_extractor` | One forward pass of a chat model over image, video and text; one step per kind plus `final`. |
| Let a chat model write an answer and give likelihoods | `replies.py`: `chat_parts` | A `reply` part and a `likelihood` part for a Hugging Face chat model. |
| Build a text-only causal language model in one call | `causal_lm.py`: `build_causal_lm_subject` | A `BrainScoreModel` that records, takes interventions, replies and scores options. |
| Use a model behind a web service | `api_behavioral.py`: `api_answer`, `build_api_reply`, `api_policy` | An `answer` part, a `reply` part or a `policy` that calls a provider API, with a response cache. |
| Lesion or stimulate units | `intervention.py`: `pytorch_intervention` | An `intervention` part that installs and removes forward hooks on a torch model. |
| Choose units or a layer for a region | `brainscore/selection.py` (outside this folder): `UnitSelector`, `LayerSelector`, `CompositeSelector` | Unit selection for a region; the `region_layer_map` of the model names the layers. |
| Add facts to a score's provenance | `provenance.py`: `ProvenanceSubject` | A wrapper that puts your registration facts on every response of a subject. |
| Choose the device | `device.py`: `select_device` | A `torch.device`: the one you name, else CUDA, MPS or CPU. |

## The files

- `__init__.py`: the module note on the extractors; it holds no code.
- `device.py`: the device choice that the wrappers share.
- `extractors.py`: the code that every recording shares: layer lookup, hooks (`LiveRecording`), batching, the cache key (`CachingExtractor`), and the check that a recording reads only inputs it knows.
- `text_wrapper.py`: the extractor for text models.
- `vlm_vision_wrapper.py`: the extractor for pictures in a vision-language model.
- `video_wrapper.py`: the extractor for video models.
- `vlm_video_wrapper.py`: the extractor for clips in a vision-language model.
- `audio_wrapper.py`: the extractor for audio models.
- `fused.py`: the recording of one pass over image, video and text together.
- `replies.py`: the reply and likelihood parts of a chat model, and the prompt-to-input step they use.
- `causal_lm.py`: the one-call registration of a causal language model.
- `api_behavioral.py`: the parts that use a closed-weight model behind a web service.
- `intervention.py`: the intervention part for a PyTorch model.
- `selection.py`: the tensor coordinates and layer-site names that recording and intervention share.
- `provenance.py`: the response wrapper for provenance and the per-session note of the parts used.

## What belongs here, and what does not

- Code here is general and serves many kinds of models.
- Nothing here names a model, a family of models or a study.
- Code for one model or one group of models lives in that plugin's folder under `brainscore/models/`.
- A new general part is added here only when a model author outside the current work gains from it, and its file name says what it does.

## Guides

To register a model, read [wrap your model](../../docs/guides/wrap_your_model.md), [perturb and select units](../../docs/guides/perturb_and_select_units.md), [behavior](../../docs/guides/behavior.md) and [new output](../../docs/guides/new_output.md).
