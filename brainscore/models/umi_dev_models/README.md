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
  The readout reads the vision part only, so these two formats come from pictures: a task on text
  or clips only is N/A in pre-flight and refused before any compute; a trial that shows a picture
  with other inputs is answered from the picture. It needs the task's fitting stimuli. The provenance names the readout layer.
- **Pictures:** each picture is one whole image, for a recording and for an answer (the same
  stated settings reach both); a processor that still gives tiles or crops for a recording is
  refused.
- **Placement:** each part on one accelerator; a device map that offloads weights or splits a
  part is refused. The processor loads before the weights, so a processor fault shows before a
  large download.
- **Provenance:** each score names the checkpoint and its revision (the commit of its Hub files;
  a full commit; the plugin test refuses any other value), the depth hypothesis and the readout layer.
- **One session for regions, one session for an answer:** the language layer is recorded from text
  in a passive session, and a session that asks for it under a task (also with neural outputs only)
  is refused before any compute. A recording during an answer has to be verified for each family
  apart, and no benchmark of the study asks for it.

## What a model gets

| Output | Given | Reason when not |
|---|---|---|
| neural V1, V2, V4, IT, VWFA | all | |
| neural language_system, from text | all | |
| neural language_system during an answer | none | the study's rule above (Qwen3-VL and Gemma 3 also need a general change, below) |
| vision layers as LayerPCA(1000) of the patch tokens | SmolVLM2, InternVL3, Gemma 3 | Qwen-VL records the hidden width, no PCA. Gemma 3 first takes the mean of each 4 x 4 block of its 64 x 64 patch grid (256 tokens, the pool of Gemma 3's own projector; `token_pool` in the values, named in the provenance): its whole grid (4096 tokens x 1152) gives a PCA fit that SVD refuses |
| clips (`vision:video`), from the vision part | Qwen3-VL, Qwen2.5-VL | the other processors give no flattened clip patches. The setting `video_sampling` (in the values, named in the provenance) states the frame sampling of a clip: the model authors' tool `qwen-vl-utils`: 2 frames per second, at least 4 frames, at most 768, sampling on. It equals the class defaults of the Qwen2.5-VL and Qwen3-VL video processors (Qwen2.5-VL's class has no rate and sampling off; the study sets them), so a 16-frame clip at 30 Hz gives 4 frames |
| behavior: choice, probabilities, log_probabilities, rating, rt, text | all | |
| perturbation (lesion, stimulation, pharmacological) | Qwen3-VL, Qwen2.5-VL | the other families record their vision layers as PCA components, and a lesion addresses the module's units |
| motor (grid game), from the observation | all | |

## Needs a general change

The study leaves out the language layer during an answer by its own rule, for every family (on v2
it would run for SmolVLM, InternVL3 and Qwen2.5-VL). These abilities also need a change of shared
code before a later study could declare them:

- Qwen3-VL and Gemma 3: the language layer during an answer. Qwen3-VL's language model gets
  `visual_pos_masks`, which the recording refuses (`brainscore/model_helpers/_extractors.py`), and
  Gemma 3's gets its attention mask as a mapping of causal masks, which the text recording cannot
  read (`brainscore/model_helpers/text_wrapper.py`). Pre-flight cannot tell a passive session from
  a session with a task, so a study model refuses the second when it starts.
- Qwen3-VL: clips reaching the language layer: a region's input tuple cannot name two forms of
  one input family (`brainscore/brainscore_model.py`).
- A vision part split over several accelerators: the vision extractor moves its module to one
  device (`brainscore/model_helpers/vlm_vision_wrapper.py`); such a placement is refused.

## Add a model

Add one entry to `model_configs.json` and one registry line to `__init__.py`. The values:

- `hf_id`: the instruction-tuned checkpoint on the Hub; `revision`: the commit of its files
  (a full commit; the plugin test refuses any other value). `model_class`: `architectures` in its `config.json`;
  it must exist in the pinned transformers.
- `vision`, `language`: the module path of each part, then the path of its list of repeated
  blocks. Build the class from its config on the meta device (resolve the class first) and print
  it. `test.py` checks the paths on each class built from a small configuration of its own
  (`SMALL_CONFIGS`), and the qualification on the class built from the real settings made small.
- `dtype`: `bfloat16`. `download_gb`: an estimate of the checkpoint's size, for the download check.
- `image_processor_kwargs`: `null` when the processor gives flattened patches with
  `image_grid_thw` (Qwen-VL); otherwise the settings that give one whole image per picture
  (`do_image_splitting`, `do_pan_and_scan`, `crop_to_patches` set to false).
- `video`: `true` only for the flattened-patch layout with a video processor.
- `video_sampling` (optional): `fps`, `min_frames`, `max_frames` and `do_sample_frames` of a video
  processor, with the `source` of the values.
- `token_pool` (optional): the side of the block of the patch grid that a whole-picture layer averages
  before its PCA; only when the whole grid is too large for the fit (Gemma 3: 4).
- `device_map`: `"auto"` when the weights (2 bytes per parameter) take more than about two thirds
  of one accelerator.
- `packages`: the packages the model's processor or forward imports beyond the reference
  environment, by name; each is stated once, as a line of `requirements.txt` (the run environment of
  the study installs that file; the line holds no version yet). A load without one names it before
  any download.

Then prove that the model complies, in three steps from the folder `unified`:

1. Offline, on a stand-in of the stated layout:
   `pytest brainscore/models/umi_dev_models/test.py -m "not slow and not qualification"`.
2. The qualification of the family with no weights: the real processor from the real settings files
   and a tiny random model of the real class, through the study's own `get_model`. Fetch the
   settings files only (an allow-list, so no weight file can come; a gated model needs a token):

   ```bash
   python -c "from huggingface_hub import snapshot_download; snapshot_download('<hf_id>', allow_patterns=['config.json', 'generation_config.json', 'preprocessor_config.json', 'processor_config.json', 'video_preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.json', 'chat_template.jinja', 'vocab.json', 'merges.txt', 'tokenizer.model', 'added_tokens.json'])"
   HF_HUB_OFFLINE=1 pytest brainscore/models/umi_dev_models/test.py -m qualification -k <name>
   ```

   A new class needs a small configuration in `SMALL_CONFIGS` and a reduction in `_small` (the tests
   fail and say so). A selected qualification fails when a package or a settings file is missing.
3. On a GPU machine, the real weights (the variable allows the first download):
   `BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK=1 pytest brainscore/models/umi_dev_models/test.py -m slow -k <name>`.

## What a model of this study must not do

Change anything outside this folder. An ability that needs shared code is listed above and not
declared until that change exists.
