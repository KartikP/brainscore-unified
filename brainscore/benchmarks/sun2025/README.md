# Sun 2025 motion benchmark

Registered identifier: **Sun2025-motion-vector-error**, version 1.

The subject watches a short clip and reports the direction and speed of the motion
at the center. The benchmark compares the reports with the three participants of
Sun et al. 2025 ([doi:10.1038/s42256-025-01068-w](https://doi.org/10.1038/s42256-025-01068-w)).

## What it asks

- Inputs: the clip on `vision:video` and the instruction (`instruction`).
- Output: `behavior:text`, one JSON object `{"direction_deg": ..., "speed_px_per_frame": ...}`.
  Direction is counterclockwise from right in degrees; speed is in pixels per frame
  of the 600 by 600 pixel clip. The benchmark parses the text itself: the answer is a
  pair of numbers, and no behavior format holds a pair.
- Generation: at most 64 tokens, greedy (`Text(max_tokens=64, temperature=0)`).
- Trial memory: `none`. Each of the 1,120 clips (40 scenes, 7 modulations, 4 flips)
  is its own trial, in a fixed order (seed 0).

## The score

The four flips of a scene and modulation are unflipped and averaged. The score is
the mean endpoint error between that vector and each participant's mean report,
over the 280 groups and the three participants: displayed pixels per frame,
**lower is closer to the participants**. It is raw (`normalized=False`); no ceiling
is estimated. The reference is the three participants; `floor` is the error of the
no-motion answer (speed 0 on every trial), computed from the reference.

A reply that is not one JSON object with the two numbers in range is `invalid`, kept
with its raw text in `reports` and counted (`n_invalid`). Any invalid or missing
trial makes the score NaN and `result_validity='invalid'`, so a hard group never
leaves the mean.

## An adaptation of the study

`score.attrs['adaptation']` lists what differs from the study:

- a numeric report replaces the interactive motion matching of the participants;
- the clips are reconstructed from the published frames (16 frames at 30 Hz, a 600
  pixel circular aperture, an OpenCV raster), not the PsychoPy display;
- no monitor calibration, visual angle, feedback, blanks or matching loop;
- the reference is each participant's mean of four unflipped reports, with no
  human reliability ceiling.

The benchmark is not on the leaderboard (`leaderboard_eligible = False`).

## Run it

Prepare the data first ([data plugin](../../data/sun2025/README.md)), then:

```python
import brainscore

score = brainscore.score(brainscore.load_model('<a model with vision:video input>'),
                         'Sun2025-motion-vector-error')
```

`brainscore.load_benchmark('Sun2025-motion-vector-error', scenes=[1])` scores a named
subset: its identifier ends in `-subset-<hash of its stimuli and reference>` and
the score says `full_study=False`. A model with no `vision:video` input is N/A before
any compute; `chance-baseline` is N/A too (it does not give text).
