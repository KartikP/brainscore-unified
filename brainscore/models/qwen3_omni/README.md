# qwen3-omni-30b-a3b-thinking

The thinker of `Qwen/Qwen3-Omni-30B-A3B-Thinking` (bf16, `device_map='auto'`; about 16 GB on each of 4 L40S).
One forward pass takes the clip and its sound of a row together (`use_audio_in_video=True`).

## Inputs and declarations

- Inputs: `vision:video` (required) and `audio` (optional). Each row's clip gives 8 frames, evenly spaced. The
  model gets `fps = 8 / 1.49`, so the picture and sound positions match in time.
- Sound: a mono 16 kHz WAV, the form the Algonauts 2025 benchmark prepares. Another rate is refused.
- Text: the model's chat template for one user turn holding the clip, then the start of the assistant turn.
- Regions (committed map):
  - `V1` = `layers.14/last`, `V4` = `layers.26/last`, `IT` = `layers.32/last`. They are recorded from the clip.
    When a sound is given, it reaches these tokens too, so they state both inputs.
  - `A1` = `layers.23/sound`, the mean over the sound tokens. A session without sound that asks for `A1` is N/A in
    pre-flight.
  - With sound, the sound tokens also follow earlier picture tokens in the interleaved sequence.

## Recorded layers (one pass)

- Thinker states `embed` and `layers.0` to `layers.47` (width 2048). Each is pooled three ways:
  - `/picture`: the mean over the picture tokens;
  - `/sound`: the mean over the sound tokens;
  - `/last`: the last token.
- Encoder blocks `visual.blocks.{0,8,17,26}/mean` and `audio_tower.layers.{0,10,21,31}/mean`.

## Layer rule

The `REGION_LAYER_MAP` was chosen once, on films that the registered Friends score does not use:
- Participant 1, `friends_s01e06a` to `friends_s01e10b`, with sound.
- The benchmark scores at `friends_s01e01a` to `friends_s01e05b`. The run script asserts that the two sets are
  disjoint.

Each candidate layer is scored alone, as one band, with the benchmark's own fit: the stacking, the exclusions, the
5 folds over films, and the median r over the 1000 parcels. Candidates:

- `V1`: the 4 vision-encoder blocks, plus `embed` and `layers.0` to `layers.15` (picture and last pools).
- `V4`: `layers.16` to `layers.31` (picture and last).
- `IT`: `layers.32` to `layers.47` (picture and last).
- `A1`: the 4 audio-encoder layers, plus `embed` and `layers.0` to `layers.47` (sound pool).

The highest median r wins. On a tie, the shallower layer wins. A candidate that is constant over scans
(`embed/last`, the embedding of a fixed text token) has no usable fit and is skipped. The scores of the candidates
are in the run record.

Selection result (median r, one band): `layers.14/last` 0.168, `layers.26/last` 0.169, `layers.32/last` 0.160,
`layers.23/sound` 0.163.
- The best picture pool, `layers.42/picture`, is 0.142.
- The best encoder blocks: `visual.blocks.17/mean` 0.120, `audio_tower.layers.31/mean` 0.124.

## Limits

- Batches:
  - Rows run in batches of 8, in table order.
  - A row's thinker values differ slightly from a run of that row alone. This is bf16 kernel rounding. Per-unit r
    over 24 Friends scans, `layers.23`: picture 0.999, sound 0.995, last 0.980 (medians).
  - The same run twice is identical.
- The audio encoder: it encodes the sounds of a batch one at a time. In transformers 4.57 its attention keeps the
  rows of a batch apart only under FlashAttention 2.
- No context: a row has no context from the rows before it.
- Pooled layer names are recording addresses only.
