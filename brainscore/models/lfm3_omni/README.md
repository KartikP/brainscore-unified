# lfm3-omni-2b

`LiquidAI/LFM3-Omni-2B` at revision `c01718a568e53ae936e6b5e1ff5ab2de7d7f388b` (2.17B parameters, bf16; not
gated). Licence: LFM Open License v1.0. It is used here for research. The model runs its own code from the same
revision (`trust_remote_code`): a vision encoder (14 layers), an audio encoder (17 layers) and a language model of
24 layers (width 2048) that mixes short convolutions and attention. Its processor also loads the speech codec of
the same revision (`audio_tokenizer/`); no speech is generated here.

## Environment (separate from `bsu`)

The model code needs transformers 5.20.2 or later (the model card), and `torchcodec`. `bsu` has transformers
4.57.6. Version 5.20.2 is inside the packages' bound (`transformers>=4.57,<6`), so no bound changes. The plugin
runs in a clone of `bsu` that differs in transformers (and `torchcodec`), with the four code copies on
`PYTHONPATH`:

```bash
conda create -y --clone bsu -p <fast disk>/envs/lfm3                 # bsu itself is not changed
conda activate <fast disk>/envs/lfm3
pip install "transformers==5.20.2" torchcodec                        # the torchcodec release for the clone's torch
conda list --explicit > lfm3-conda.txt
pip freeze --all > lfm3-pip.txt
sha256sum lfm3-conda.txt lfm3-pip.txt                                # recorded in the run's provenance
PYTHONPATH=<core>:<unified>:<vision>:<language> python <run script>
```

- The load refuses a transformers version below 5.20.2, with this remedy.
- If transformers 5.20.2 needs a newer torch than the clone has, the run stops and asks: that is a larger change.
- Each score stamps its environment. Scores from this environment sit outside the bit-for-bit reference table.
- Not run yet: the model code needs transformers 5.20.2, which is not on the development machine. The offline
  tests use a stand-in model and processor; the real load, the token counts and the frame clock are checked in the
  first cloud probe.

## Inputs and declarations

- Inputs: `vision:video`, `audio`, `text`, `vision:image` and `instruction`. None is required.
- A clip gives 8 frames, evenly spaced, passed as decoded frames with their metadata (`fps = 8 / 1.49`,
  `do_sample_frames=False`), so the processor does not sample again (its default is 2 frames a second, at least 4).
  The frames go in groups of two, each with its time stamp. With the sound (`use_audio_in_video=True`), the sound of
  each group follows it.
- Without a sound, the row is the clip alone; without a clip, the sound alone.
- Sound: a mono 16 kHz WAV. Another rate is refused.
- Text of a recording: `<|startoftext|><|im_start|>user\n<|video|><|audio|><|im_end|>\n<|im_start|>assistant\n<think></think>`,
  with no system turn. The empty thinking block is what the model's template writes.
- Regions: `V1`, `V4`, `IT` and `A1` are all language-model layers after the point where the clip and the sound
  meet, so each states both inputs (rule 2.5). `A1` reads the sound tokens: a request for `A1` without sound is
  refused before any pass. A session with a sound alone that asks for a region is refused when it starts.
- Written answers: `behavior:choice`, `behavior:rating`, `behavior:text` from the reply, through the model's own
  `generate` (`generation_mode='text'`) on the inputs of the shared chat parts, greedy unless the task states its
  settings; the log probability of each option, `behavior:probabilities` (softmax) and `behavior:rt` from the
  likelihood of the shared chat parts. Each answer lists text, pictures and clips as its inputs, so a sound-only
  request for an answer is N/A in pre-flight.

## Recorded layers (one pass a row)

- Language-model states `embed` and `layers.0` to `layers.23`. Each is pooled three ways: `/picture`, `/sound`,
  `/last`.
- Encoder layers `vision_tower.layers.{0,4,9,13}/mean` and `audio_encoder.lfm.layers.{0,5,11,16}/mean`.
- Token counts of each row: `tokens/picture`, `tokens/sound`.

## Layer rule (fixed before any run)

The rule of `qwen2_5_omni` (README there), with N=24: `V1` the 4 vision-encoder layers, `embed` and layers 0-7;
`V4` 8-15; `IT` 16-23; `A1` the 4 audio-encoder layers, `embed` and all layers (sound pool). If the rule picks an
encoder layer for `A1`, `A1` states `audio` alone.

**State: the map is a placeholder** (the middle layer of each band) until the rule runs. No score of this folder
is registered yet.

## Limits

- One row a pass: the batch test of this model has not run.
- No context: a row has no context from the rows before it.
- Pooled layer names are recording addresses only.
- No speech output: the catalog has no speech output channel.

## Not wired yet

- A written answer with sound: the shared chat parts refuse a sound part.
- A picture alone to the regions: the layers are chosen on films; a picture map needs its own selection.
- The `language_system` region: no layer was chosen on language data.
- The intervention part: the pooled names are not module addresses.
- Screen acts: the coordinate grid of this model is not measured.
