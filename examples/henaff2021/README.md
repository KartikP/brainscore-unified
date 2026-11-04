# Image viewing and macaque V1 straightening

This example turns the main image-viewing experiment in **Hénaff et al. (2021),
Primary visual cortex straightens natural video trajectories**, into a UMI
experiment. It includes a comparison against the authors' archived macaque
curvature estimates. It is an exploratory static-frame adaptation, not a
registered, ceiling-normalized Brain-Score benchmark.

[Paper](https://doi.org/10.1038/s41467-021-25939-z) ·
[Source archive](https://doi.org/10.17605/OSF.IO/VF2XK)

**Status: parked draft for future work.** The experiment and its remote
[validation evidence](validation/README.md) are preserved for review. Resolve
the scientific questions under "Results and fidelity boundaries" before
registering a scored benchmark.

## What the model does

The model sees shuffled still images. It does not watch a movie or answer a
question. A neural recorder saves its declared V1 representation for every image.
Afterward, the analysis arranges the responses in each movie's original frame
order and measures how sharply that path turns. It compares that curvature with
the pixel trajectory and with the archived macaque estimates.

The original study also presented still frames in randomized order, with blanks
between them. This isolates the contribution of spatial image processing. A
conventional image model can therefore participate without video input, language
generation, or session memory.

The stimulus factors are 10 movies, two spatial scales, and natural versus
artificial sequences. Each of the 40 sequences contains 11 frames. Artificial
sequences share their first and last frames with natural sequences, so there are
440 sequence-frame entries but **400 distinct images**. Shared endpoint responses
are reused when reconstructing trajectories, as in the archived spike arrays.

## The UMI objects

| Object | Role in this example |
|---|---|
| Subject | A registered model, default `clip-vit-b-32`, declaring a usable vision input and `neural:V1` output |
| StimulusSet | 400 distinct images with stable IDs, shuffled using a recorded seed |
| TaskContext | `TaskContext(task_type='passive')`: plain image viewing |
| TaskProtocol | One session per independent repetition, with resets before and after |
| Trial | One image presentation within the session |
| RecordNeural | Requests and saves the model's declared V1 representation |
| Experiment | Validates compatibility, runs the protocol, and saves its measurements |
| Analysis | Restores movie order, calculates curvature, joins the macaque reference |

Natural/artificial and scale are stimulus factors within a session. They are not
separate UMI conditions or separate visits. A UMI repetition is a fresh visit;
the original acquisition's 50 repeated image passes are not renamed as 50 UMI
repetitions. The default model experiment uses one pass. Extra UMI repetitions
use the same seeded order, with fresh resets, and are reported separately.

The code uses existing `TaskProtocol`, `Experiment`, and `RecordNeural` APIs.
It adds no Subject methods, model-kind subclasses, channels, or core changes.
Compatibility remains declaration-based. Current TaskProtocol image rows use
the `vision` channel; registered CLIP declares it. The recorder addresses `V1`,
not a model layer. The model's registration owns that mapping.

## Run on a remote worker

Use the unified repository and the peer revisions in `install/peer-revisions.json`.
The example needs NumPy, pandas, SciPy and Pillow. Raw spike inspection also
needs h5py. The optional source-workbook regression check needs openpyxl.
Preparing the MATLAB archive needs several GB of memory; run preparation,
inference and tests remotely.

First extract the downloaded `vf2xk-osfstorage-archive` directory, then:

```bash
python -m brainscore.experiments.henaff2021 prepare \
  --archive /path/to/vf2xk-osfstorage-archive \
  --output /path/to/prepared-henaff2021

RESULTCACHING_DISABLE=1 python -m brainscore.experiments.henaff2021 run \
  --prepared /path/to/prepared-henaff2021 \
  --model clip-vit-b-32 --region V1 --seed 0 \
  --output /path/to/runs/clip-henaff2021
```

Both commands require a new output directory. Preparation records source and
image checksums. Loading prepared data checks for changed tables or images.
Use another registered model by changing `--model`; it must satisfy the
protocol's declared inputs and outputs. No fitting to the macaque targets occurs.
For a result you intend to compare or share, pass `--provenance run-provenance.json`
with the model/checkpoint revision and hashes, dtype, device and execution
settings. This dictionary travels in the experiment and analysis metadata. The
experiment does not discover model internals to infer those settings. The
qualified example's receipt records them; an omitted receipt leaves them
unspecified, rather than claiming that package versions alone reproduce weights.

The same workflow in Python is:

```python
import brainscore
from brainscore.experiments.henaff2021.data import load_prepared
from brainscore.experiments.henaff2021.experiment import make_experiment, analyze

study = load_prepared('/path/to/prepared-henaff2021')
model = brainscore.load_model('clip-vit-b-32')
experiment = make_experiment(model, study, '/path/to/runs/new-run', region='V1')
experiment.validate()
result = experiment.run()
activity = result.neural('V1', repetition=0)
summary = analyze(activity, study, '/path/to/runs/new-run/analysis-0')
```

`make_experiment` composes the familiar objects as follows:

```python
protocol = TaskProtocol(study.stimuli(seed=0), TaskContext(task_type='passive'))
experiment = Experiment(
    subject=model, protocol=protocol, tools=[RecordNeural(['V1'])],
    output_dir='/path/to/runs/new-run')
```

To inspect biological measurements independently:

```python
from brainscore.experiments.henaff2021.data import load_spike_counts
counts = load_spike_counts('/path/to/archive', 'ay2_u003_image_sequences')
# NeuroidAssembly: 440 sequence-frame entries x 39 units x 50 acquisition repeats.
```

## What the comparison means

Discrete curvature is the average angle between successive displacement vectors,
measured in degrees. Negative relative curvature means straightening.

- **Model:** raw recorded feature curvature minus the curvature of the exported
  PNG pixels. No PCA, unit normalization, neural fitting, or layer search is used.
- **Macaque:** archived `Delta Curvature: neural` minus `Delta Curvature: control`.
  This retains the authors' noise correction and null-model correction.
- **Archived LN-LN model:** the authors' provided model-relative curvature. It
  is a reference comparison, not a model refitted by this example.

The primary analysis uses stride 2. Both interleaved phases contribute their
local angles: four angles from frames 1,3,5,7,9,11 and three from frames 2,4,6,8,10.
The secondary stride-1 analysis uses all nine local angles. Phase means are not
given equal weight. Zero-length displacements make curvature undefined; they
are saved as invalid, never converted to zero or silently omitted from a score.

The biological inclusion rules follow `plot_summary.m`: distance > 0.25, and
squared fit correlations > 0.75 for means, > 0.5 for variances, > 0.25 for
covariances. Its sequential MATLAB deletions remove original recording IDs
2 and 8, namely `ay2_u007` and `ay5_u002`. The loader keeps excluded rows and
their reasons. The CSV's Index restarts at 1 for each category/stride block;
it is not a globally unique ID. Within each recording/category/stride group,
the movie mapping follows `summary.m`, with scale 1 then scale 2.

Retained populations are averaged per movie/scale. Pearson correlation and
absolute differences are then reported separately for natural and artificial
sequences, across 20 movie/scale pairs. A pooled natural/artificial correlation
would mix a category effect with movie-specific agreement, so it is not reported.
The provided population SEM is descriptive; populations and the two scales of
one movie must not be treated as independent animals or movies.

## Results and fidelity boundaries

The first complete run used registered CLIP ViT-B/32 and its V1 mapping. It
produced valid trajectories for all 40 sequences at both strides. At stride 2,
the model-to-macaque movie-level correlations were approximately **0.850 for
natural sequences** and **0.387 for artificial sequences**. These are exploratory
agreement statistics, not Brain-Score leaderboard scores. Strong correlation
does not establish matching absolute curvature or biological mechanism.
For natural sequences, median model-relative curvature was +2.485 degrees,
while the median of the macaque movie/scale means was -9.167 degrees. This
model run therefore did not reproduce natural-sequence straightening.

The reference loader recovers the paper's 127 retained natural trajectories
(median -9.9346 degrees) and 119 artificial trajectories (median +24.4364 degrees).
Movie-level macaque and LN-LN values also match the paired values in the
independent `Figure7e` source-data worksheet to numerical precision.

Important limits are saved in the artifacts and must travel with results:

1. **Static presentation.** No physical display clock, 200 ms exposure dynamics,
   100 ms blank response, adaptation, or angular field-of-view calibration is
   simulated. Source timing is retained as metadata. Extra calibration/contrast
   images outside the 40 analyzed sequences are omitted. This is an adaptation
   for static image representations, not an exact timed or stateful replay.
2. **Image conversion.** The archive contains floating pixels with slight
   interpolation overshoot. Preparation explicitly clips to [0,1] and rounds
   to 8-bit grayscale PNGs, without per-image normalization or resizing. The
   model applies its own registered preprocessing. Original-float and PNG pixel
   baselines are both saved; their curvature differs by up to 2.636 degrees.
3. **Archive baseline discrepancy.** Recomputed float-pixel curvature agrees
   with the LN-LN file's pixel baseline within 0.018 degrees. It differs from
   the inference CSV's pixel column by up to 21.936 degrees. The cause has not
   been established. The reported macaque relative values and source-workbook
   comparisons remain the authors' archived values. Baselines are never silently
   substituted or treated as interchangeable.
4. **Different estimators.** Raw model-feature geometry is not the paper's
   noise-adjusted neural discriminability geometry. An exact reproduction of
   the biological inference and LN-LN fitting requires additional author code
   and validation. The paper lists analysis code as available upon request.
5. **V1 mapping.** A registered model's V1 region is a modeling hypothesis.
   The example does not discover or optimize its layer using these targets.

## Saved files and checks

Preparation saves `frames.csv`, `pixels.csv`, `macaque.csv`, PNGs and a checksum
manifest. The experiment saves its standard UMI manifest and neural recording.
Each `analysis-N` directory contains:

- `trajectories.csv`: model, pixel and macaque results joined by category,
  scale, movie and stride, with validity and population counts.
- `macaque_inclusion.csv`: all 720 archived estimates, including exclusions.
- `summary.json`: comparisons, definitions, pixel diagnostics, source hashes,
  and package versions.

Run the focused checks remotely:

```bash
HENAFF2021_ARCHIVE=/path/to/archive python -m pytest \
  tests/test_henaff2021.py tests/test_task_protocol.py -q
```

Tests cover known geometric controls, pooled stride phases, undefined geometry,
stimulus identity after shuffling, missing/duplicate responses, UMI participation
and resets, incompatible subjects, source-row reordering, raw spike axes and
shared endpoints, published inclusion totals and medians, and the paired
source-workbook comparison. Data-dependent checks skip when the archive is
absent. Large data, weights and recordings are not checked into the repository.
