# minicpm-o-4.5

`openbmb/MiniCPM-o-4_5` at revision `fb09628e98b1f55d8d5b863e0eccfcdd11bff499` (Apache-2.0, not gated; 9.4B
parameters, bf16). The model runs its own code from the same revision (`trust_remote_code`): a SigLIP vision
encoder (27 blocks) with a resampler, a Whisper audio encoder (24 layers) and a Qwen3 language model (36
layers, width 4096). The speech part is not loaded (`init_tts=False`). `USE_FLAGOS` must be unset: with it, the
model code replaces the RMSNorm of transformers for every model in the process; the load refuses it.

## Environment (separate from `bsu`)

The model code needs transformers 4.51.0, the version its card pins. It does not run on transformers 4.57.6 (the
version of `bsu`), checked on 2026-11-28 with a small random configuration and the real processor:

- Its Whisper layer (`modeling_minicpmo.py:3848`) unpacks three values from `WhisperAttention`; 4.57.6 returns
  two (`modeling_whisper.py:368`). A row with sound fails.
- Its generation step (`modeling_minicpmo.py:5025`) reads `DynamicCache.seen_tokens`, which 4.57.6 does not have.
  A written reply fails.
- The picture path, the language model and the likelihood run on 4.57.6.
- The load refuses a transformers version other than 4.51.0, with this remedy.

The packages are not changed. The plugin runs in a clone of `bsu` that differs only in transformers, with the four
code copies on `PYTHONPATH`, as a proof run uses them:

```bash
conda create -y --clone bsu -p <fast disk>/envs/minicpmo     # bsu itself is not changed
conda activate <fast disk>/envs/minicpmo
pip install transformers==4.51.0                              # pip states the packages' bound >=4.57 as a conflict
conda list --explicit > minicpmo-conda.txt
pip freeze --all > minicpmo-pip.txt
sha256sum minicpmo-conda.txt minicpmo-pip.txt                 # recorded in the run's provenance
PYTHONPATH=<core>:<unified>:<vision>:<language> python <run script>
```

- The packages' bound (`transformers>=4.57,<6`) is declared metadata. At import, core warns that the environment
  is outside the bound (`environment_check.warn_on_drift`); it does not stop.
- Each score stamps its environment. Scores from this environment sit outside the bit-for-bit reference table.
- `librosa` and `soundfile`, which the model code imports, are in `bsu`, so they are in the clone.

## Inputs and declarations

- Inputs: `vision:video`, `audio`, `text`, `vision:image` and `instruction`. None is required.
- A scan is read as the model reads a film: units of one second, each one frame and then the sound of that
  second. A scan of 1.49 s is two units: the frame at 0 s and the sound 0-1 s, then the frame at 1 s and the
  sound 1-1.49 s. Each frame is one slice (`max_slice_nums=1`, 64 picture tokens). The 5-frames-a-unit mode is
  not used.
- Without a sound, the units hold the frames alone; without a clip, the sound alone.
- Sound: a mono 16 kHz WAV. Another rate is refused.
- Text of a recording: one user turn holding the units, with no system turn, then
  `<|im_start|>assistant\n<think>\n\n</think>\n\n` (thinking off, as the model's template writes it).
- Regions: `V1`, `V4`, `IT` and `A1` are all language-model layers after the point where the clip and the sound
  meet, so each states both inputs (rule 2.5). `A1` reads the sound tokens: a request for `A1` without sound is
  refused before any pass. A session with a sound alone that asks for a region is refused when it starts.
- Written answers (`answer_parts`, the model's own input path): `behavior:choice`, `behavior:rating`,
  `behavior:text` from the reply (greedy unless the task states settings, thinking off); the log probability of
  each option, `behavior:probabilities` (softmax) and `behavior:rt` from the likelihood. The task's instruction is
  the system turn. A clip in an answer is 8 frames taken evenly, each one slice. Under a `verbatim` choice the
  likelihood scores the plain text on the language model. Each answer lists text, pictures and clips as its
  inputs, so a sound-only request for an answer is N/A in pre-flight.

## Recorded layers (one pass a row)

- Language-model states `embed` and `layers.0` to `layers.35`. Each is pooled three ways: `/picture`, `/sound`,
  `/last`.
- Encoder blocks `vpm.encoder.layers.{0,9,17,26}/mean` and `apm.layers.{0,8,15,23}/mean`.
- Token counts of each row: `tokens/picture`, `tokens/sound`. With the real processor, a scan of 1.49 s gives 128
  picture tokens (2 x 64) and 15 sound tokens (10 + 5).

## Layer rule (fixed before any run)

The rule of `qwen2_5_omni` (README there), with N=36: `V1` the 4 vision-encoder blocks, `embed` and layers 0-11;
`V4` 12-23; `IT` 24-35; `A1` the 4 audio-encoder layers, `embed` and all layers (sound pool). If the rule picks an
encoder layer for `A1`, `A1` states `audio` alone.

**State: the map is a placeholder** (the middle layer of each band) until the rule runs. No score of this folder
is registered yet.

## Limits

- A session with a sound alone that asks for a region is refused when it starts, before any pass: an error, not N/A. Every region states the clip and the sound (rule 2.5), so pre-flight lets it through.
- One row a pass. The model's config sets `stream_input`, which allows one row with sound a pass
  (`modeling_minicpmo.py:748-749`); with that switch off, its code fills the sound of the last row of a batch only
  (`:762-797`).
- A scan is units of one second with one frame each: a scan of 1.49 s shows 2 frames.
- No context: a row has no context from the rows before it.
- Pooled layer names are recording addresses only.
- No speech output: the speech part is not loaded, and the catalog has no speech output channel.

## Not wired yet

- A written answer with sound, earlier trials as earlier turns, and a `Segments` task (`answer_parts`).
- A picture alone to the regions: the layers are chosen on films; a picture map needs its own selection.
- The `language_system` region: no layer was chosen on language data.
- The intervention part: the pooled names are not module addresses.
- Screen acts: the coordinate grid of this model is not measured.
