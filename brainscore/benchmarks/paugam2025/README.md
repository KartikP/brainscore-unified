# CNeuroMod mario fMRI benchmark (Paugam et al. 2025)

Registered identifiers, version 1: `Paugam2025-fMRI-mario-cortex` (brain) and
`Paugam2025-behavior-mario-buttons` (behavior).

Five participants of the Courtois NeuroMod project played Super Mario Bros. (NES, 22 levels) in
the scanner, about 15 hours each, and heard the game sound (TR 1.49 s). The data set paper is
Paugam et al. 2025 (bioRxiv, https://doi.org/10.1101/2025.11.28.691119). The benchmark compares
the model's response to each participant's own play with the participant's brain activity.

## What it asks

- Input, one row per TR that shows a play: the clip of the TR's frames on `vision:video` (60 frames
  per second, up to 1.49 s), its middle frame on `vision:image` and its sound on `audio` (16 kHz,
  mono). A model gets the channels it declares; `audio` is optional. The subject watches the play;
  it does not play.
- Output: `neural:cortex`. A model takes part when its `region_layer_map` names its layers for
  `cortex` (at most 12, one band each); without them, the pair is not applicable and nothing runs.
- Trial memory: `none`.

## Data

- The time series: `package_data('cneuromod-mario-timeseries', part='sub-01')` (and sub-02, 03, 05,
  06), made on this machine from the public fMRIPrep outputs; see `brainscore/data/cneuromod/README.md`.
- The media of each TR, from the public play videos of `mario.replays`:
  `python -m brainscore.data.cneuromod.data_packaging.mario --packaged <packaged folder>`.
  Brain-Score hosts no brain file and no frame. Super Mario Bros. is (c) Nintendo.

## The score

Per participant: the response for `cortex`, one band per layer, each layer first projected to at
most 1000 units by one fixed Gaussian matrix, then reduced to at most 100 components on the
training TRs; a TR without a play has features 0; each band convolved per run with the SPM
double-gamma HRF. Split by run: `test_runs.txt` holds 88 of 483 runs (about 20 percent of each
participant's runs, drawn once with seed 2026 within groups of the run's main level); the paper used
80/10/10 of the plays, balanced by level. Each parcel z-scored within each run. Banded ridge on the
training runs, one penalty tuple for all parcels chosen on one held-out fold of training runs;
Pearson r per parcel (1134 parcels of the participant's own cneuromod2026 atlas) on the TRs of the
test runs. The value is the median r over parcels, raw: no ceiling is measured. The benchmark
value is the mean over the five participants; each participant's median r and per-parcel r are on
the score (`per_participant`).

References on the score (`references`): a ridge on the task alone (the seconds of play and the presses
of each button per TR, with the HRF; no model) reaches 0.154 (sub-01 0.196, sub-02 0.140, sub-03 0.152,
sub-05 0.148, sub-06 0.133), and random-vit-b-32 reaches 0.170. Much of a model's score here can come
from the timing of play and presses; a model must be above this reference to show more than that.

## The behavior form

`Paugam2025-behavior-mario-buttons` asks `behavior:choice` on the same input rows: which buttons
the participant held in each TR. The label of a TR is the set of buttons held for at least half of
the TR, joined by `+` in the order A, B, LEFT, RIGHT, UP, DOWN (`none`: no button); a label under 1
percent of the training TRs of all participants is `other`; the choices are in the order of their
training frequency, the most frequent first. Per participant, one session over the
TRs of the test runs (the same `test_runs.txt`), with every 8th TR of the training runs as the
fitting stimuli (a readout fits on them). The value is the accuracy on the test TRs, raw, mean over
the participants. Chance is on the score: the test rate of the most frequent training label
(`chance_training_majority`), the rate of the most frequent label of the test TRs (`chance_most_frequent`),
a guess drawn from the training label rates (`chance_by_rates`) and a uniform guess (`chance_uniform`).
`chance-baseline` answers every TR with the first choice (its probabilities tie; ties go to the first
label), so it lands on `chance_training_majority`.
