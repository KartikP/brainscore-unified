# UMI development study: vision-language models

Fifteen Hugging Face vision-language models (Qwen3-VL, Qwen2.5-VL, SmolVLM2, Gemma 3, InternVL3), registered
by one set of rules so that they compare with each other. Each identifier ends in `_UMI-dev`.
Everything of the study is in this folder: `model_configs.json` holds the values of each model,
`model.py` builds every one of them, `test.py` tests them, `requirements.txt` states the packages
they need beyond the reference environment.

Qwen2.5-VL 3B and 7B also exist outside the study (`qwen2.5-vl-3b`, `qwen2.5-vl-7b`), with layers
chosen by hand. A score of the study version and a score of that registration are different
measurements of the same weights.

## The rules every model follows

- **Layers: a stated depth hypothesis, no search.** A part with N blocks gives a region the
  block at index f * (N - 1), rounded half up: the vision part V1 0.10, V2 0.25, V4 0.50,
  IT 0.90, VWFA 0.90 (the IT layer); the language model language_system 0.67. The version word
  is `depth hypothesis v1` (`DEPTH_TABLE` in `model.py`); each score's provenance names it. No
  data chose a layer, so a score is a lower bound for the model, and a difference between two
  models can come from the layer and not from the model.
- **Precision:** bfloat16. Activity is exported as float32.
- **Answers:** written replies give choices, ratings and text; the likelihood of each option gives
  log probabilities; a readout fitted on the IT layer gives probabilities and response times.
  The readout reads the vision part only: it needs pictures and the task's fitting stimuli. The
  provenance names the readout layer.
- **Pictures:** each picture is one whole image; a processor that gives tiles or crops is refused.
- **Placement:** each part on one accelerator; a device map that offloads weights or splits a
  part is refused. The processor loads before the weights, so a processor fault shows before a
  large download.
- **Provenance:** each score names the checkpoint and its revision (`not pinned` until a load
  test fixes it), the depth hypothesis and the readout layer.

## What a model gets

| Output | Given | Reason when not |
|---|---|---|
| neural V1, V2, V4, IT, VWFA | all | |
| neural language_system, from text | all | |
| neural language_system during an answer | SmolVLM, InternVL, Qwen2.5-VL | Qwen3-VL and Gemma 3 refuse such a session before any compute; needs a general change (below) |
| clips (`vision:video`), from the vision part | Qwen3-VL | the other processors give no flattened clip patches; Qwen2.5-VL's clip path is not checked yet |
| behavior: choice, probabilities, log_probabilities, rating, rt, text | all | |
| perturbation (lesion, stimulation, pharmacological) | Qwen3-VL, Qwen2.5-VL | the other families record their vision layers as PCA components, and a lesion addresses the module's units |
| motor (grid game), from the observation | all | |

## Needs a general change

These abilities need a change of shared code; the study does not declare them:

- Qwen3-VL: the language layer during an answer. Its language model gets `visual_pos_masks`,
  the positions of the picture tokens; the recording refuses every mask other than
  `attention_mask` (`brainscore/model_helpers/_extractors.py`). Pre-flight cannot tell a session
  without an answer from one with an answer, so the model refuses the second when it starts.
- Gemma 3: the same, because its language model gets its attention mask as a mapping of causal
  masks, which the text recording cannot read (`brainscore/model_helpers/text_wrapper.py`).
- Qwen3-VL: clips reaching the language layer: a region's input tuple cannot name two forms of
  one input family (`brainscore/brainscore_model.py`).
- A vision part split over several accelerators: the vision extractor moves its module to one
  device (`brainscore/model_helpers/vlm_vision_wrapper.py`); such a placement is refused.

## Add a model

Add one entry to `model_configs.json` and one registry line to `__init__.py`. The values:

- `hf_id`: the instruction-tuned checkpoint on the Hub; `revision`: the commit of its files
  (`not pinned` until a load test fixes it). `model_class`: `architectures` in its `config.json`;
  it must exist in the pinned transformers.
- `vision`, `language`: the module path of each part, then the path of its list of repeated
  blocks. Build the class from its config on the meta device (resolve the class first) and print
  it; `test.py` checks the paths of each class this way.
- `dtype`: `bfloat16`. `download_gb`: the sum of the `*.safetensors` files on the Hub page.
- `image_processor_kwargs`: `null` when the processor gives flattened patches with
  `image_grid_thw` (Qwen-VL); otherwise the settings that give one whole image per picture
  (`do_image_splitting`, `do_pan_and_scan`, `crop_to_patches` set to false).
- `video`: `true` only for the flattened-patch layout with a video processor.
- `device_map`: `"auto"` when the weights (2 bytes per parameter) take more than about two thirds
  of one accelerator.
- `extra_forward_inputs`: what the language model gets beyond its tokens and a 2-D attention mask
  in an answering pass (read the model's forward); the language layer is then recorded only in a
  session without an answer.
- A package the processor imports beyond the reference environment goes in `requirements.txt`,
  the one statement; a load without it names it before any download.

Then run `pytest brainscore/models/umi_dev_models/test.py -m "not slow"` here, and on a GPU machine
`BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK=1 pytest brainscore/models/umi_dev_models/test.py -m slow -k <name>`
(the variable allows the first download).

## What a model of this study must not do

Change anything outside this folder. An ability that needs shared code is listed above and not
declared until that change exists.
