# qwen2.5-omni-7b, qwen2.5-omni-3b

The thinker of `Qwen/Qwen2.5-Omni-7B` and of `Qwen/Qwen2.5-Omni-3B` (class
`Qwen2_5OmniThinkerForConditionalGeneration`: no talker, no speaker file; bf16, `device_map='auto'`). Both load
natively in transformers 4.57.6. One forward pass takes the clip and the sound of a row together
(`use_audio_in_video=True`). The thinker also writes answers to text, pictures and clips.

| Identifier | Weights, revision | Licence | Thinker |
|---|---|---|---|
| `qwen2.5-omni-7b` | `Qwen/Qwen2.5-Omni-7B` `ae9e1690543ffd5c0221dc27f79834d0294cba00` | Apache-2.0 | 28 layers, width 3584 |
| `qwen2.5-omni-3b` | `Qwen/Qwen2.5-Omni-3B` `f75b40e3da2003cdd6e1829b1f420ca70797c34e` | Qwen Research License: research use, not for commercial use | 36 layers, width 2048 |

Both have a vision encoder of 32 blocks and an audio encoder of 32 layers. Neither is gated.

## Inputs and declarations

- Inputs: `vision:video`, `audio`, `text`, `vision:image` and `instruction`. None is required.
  - Brain recordings take the clip, the sound, or both. A clip gives 8 frames, evenly spaced (temporal patch 2,
    so 4 picture groups), with `fps = 8 / 1.49`, so the picture and sound positions match in time.
  - A scan of 1.49 s is shorter than one 2 s chunk of the processor: all picture tokens come first, then all
    sound tokens. Their time positions (25 a second) overlap.
  - The picture size is the processor's default limit (128 x 28 x 28 to 768 x 28 x 28 pixels a frame).
  - A row with a sound and no clip runs the sound alone (`<|audio_bos|><|AUDIO|><|audio_eos|>`).
- Sound: a mono 16 kHz WAV, the form the Algonauts 2025 benchmark prepares. Another rate is refused.
- Text of a recording: one user turn `<|vision_bos|><|VIDEO|><|vision_eos|>`, with no system turn. The model's
  template adds "You are a helpful assistant." when no system turn is given; the recording leaves it out.
- Regions: `V1`, `V4`, `IT` and `A1` are all thinker layers after the point where the clip and the sound meet,
  so each states both inputs (rule 2.5).
  - A session with the clip, with or without the sound, runs `V1`, `V4` and `IT`.
  - `A1` reads the sound tokens. A request for `A1` without sound is refused before any pass.
  - A session with a sound alone that asks for a region is refused when it starts, before any pass.
- Written answers: `behavior:choice`, `behavior:rating`, `behavior:text` (from the reply), and
  `behavior:log_probabilities`, `behavior:probabilities` (softmax) and `behavior:rt` (from the likelihood). The
  shared chat parts (`chat_parts`) on the thinker; the task's instruction is the system turn. This model has no
  thinking mode. Each answer lists text, pictures and clips as its inputs, so a sound-only request for an answer
  is N/A in pre-flight.

## Recorded layers (one pass)

- Thinker states `embed` and `layers.0` to the last layer. Each is pooled three ways: `/picture` (the mean over
  the picture tokens), `/sound` (the mean over the sound tokens), `/last` (the last token).
- Encoder blocks `visual.blocks.{0,10,21,31}/mean` and `audio_tower.layers.{0,10,21,31}/mean`.
- Token counts of each row: `tokens/picture`, `tokens/sound`.

## Layer rule (fixed before any run)

The map is chosen once, on films that the Friends score does not use: participant 1, `friends_s01e06a` to
`friends_s01e10b`, with sound. The scored films are `friends_s01e01a` to `friends_s01e05b`; the run script
asserts that the two sets are disjoint. Each candidate is one band, with the benchmark's own fit. The highest
median r wins; a tie goes to the shallower layer; a constant candidate is skipped.

For a thinker of N layers:
- `V1`: the 4 vision-encoder blocks, `embed`, and `layers.0` to `layers.{ceil(N/3)-1}` (picture and last pools).
- `V4`: up to `layers.{ceil(2N/3)-1}`. `IT`: the rest.
- `A1`: the 4 audio-encoder layers, `embed` and all layers (sound pool).

7B (N=28): 0-9, 10-18, 19-27. 3B (N=36): 0-11, 12-23, 24-35.

If the rule picks an encoder layer for `A1`, `A1` states `audio` alone (no picture reaches it).

**State: the map is a placeholder** (the middle layer of each band) until the rule runs. No score of this folder
is registered yet.

## Limits

- A session with a sound alone that asks for a region is refused when it starts, before any pass: an error, not N/A. Every region states the clip and the sound (rule 2.5), so pre-flight lets it through.
- Batches: rows run in batches of 8, in table order. The batch-8 difference of the thinker (bf16) is measured
  before a scored pass and stated here.
- No context: a row has no context from the rows before it.
- Pooled layer names are recording addresses only.
- No speech output: the thinker class has no talker, and the catalog has no speech output channel.
- In a written answer, the time positions of a clip use the processor's default rate.

## Not wired yet

- A written answer with sound: the shared chat parts refuse a sound part. This is work on the shared chat parts.
- A picture alone to the regions: the layers are chosen on films; a picture map needs its own selection.
- The `language_system` region: no layer was chosen on language data.
- The intervention part: hooks on module paths would work, but the pooled names are not module addresses.
- Screen acts: the coordinate grid of this model is not measured.
