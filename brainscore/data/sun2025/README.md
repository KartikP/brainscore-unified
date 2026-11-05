# Sun 2025 second-order motion data

Registered dataset and stimulus set: **Sun2025-second-order-motion**.
This plugin owns acquisition, source conversion, human responses, stimulus
metadata and provenance. Experiment and comparison code import its outputs.
No source packaging or dataset registration lives in a benchmark module.
The registered [Sun2025-motion-vector-error benchmark](../../benchmarks/sun2025/README.md)
consumes these data through the registries and requires the full prepared study.

Study: Sun et al., [Machine learning modelling for multi-order human visual
motion processing](https://doi.org/10.1038/s42256-025-01068-w), Nature Machine
Intelligence (2025).

## Sources and retained evidence

- [Authors' code and participant data](https://github.com/anoymized/multi-order-motion-model),
  revision c5886559b20be313125db0e1ba96baf464eab342, Apache-2.0.
- [Public stimulus archive](https://huggingface.co/datasets/sunana/mateiral-controlled-motion-dataset),
  revision a7e83195986131df109e530efda1c4cbb1f21d6d, dataset card license MIT.
  The source is human_static.zip, 9,878,720,999 bytes.
- The download retains the original PNGs used, per-scene spreadsheets, full
  shuffled trial table, all three human CSVs, the preprocessing notebook,
  presentation code, README and code license.
- source-manifest.json records source URLs/revisions, byte sizes, SHA-256s and
  ZIP CRCs. ZIP reads verify the requested HTTP range and member CRC.
- The prepared manifest records all clip and table hashes, selected scenes,
  conversion settings and source-manifest hash. Native sources stay intact.
  The full build is 1,120 clips. Subsets have distinct content-derived identities
  and complete=false.

## Download and package

Run these commands on a compute worker. Required packaging packages are numpy,
pandas, openpyxl and opencv-python with the FFV1 encoder.
Start with one scene (28 trials). Omit --scenes to fetch all 40 scenes.
Choose new output directories; existing data is never overwritten.

    python -m brainscore.data.sun2025.data_packaging.fetch /data/sun2025/native-pilot --scenes 1
    python -m brainscore.data prepare Sun2025-second-order-motion \
        --source /data/sun2025/native-pilot --output /data/sun2025/prepared-pilot

The native download uses bounded archive ranges, so it does not also store a
second, full 9.88-GB ZIP. It retains selected source frames for reproducibility.
Full acquisition and prepared clips need more storage than a pilot.

    import brainscore
    stimuli = brainscore.load_stimulus_set(
        'Sun2025-second-order-motion', root='/data/sun2025/prepared-pilot')
    human = brainscore.load_dataset(
        'Sun2025-second-order-motion', root='/data/sun2025/prepared-pilot')

Alternatively set BRAINSCORE_SUN2025_ROOT. Loaders do no downloading.
The human assembly has presentation, participant and axis dimensions.
There are three participants and two image-coordinate axes: x right, y down.
Its presentation is a scene/modulation pair after averaging four unflipped
human judgments. Stimulus rows additionally identify the flip. Human responses
and ground truth never appear in the stimulus table sent to the subject.

## Exact transformations and limits

The source presentation code uses sixteen 1024-pixel frames at 30 Hz.
Packaging enlarges them 1.5 times, centers the spreadsheet's probed location,
and cuts out a 600-pixel circular aperture. It creates original, horizontal,
vertical and both-axis flips; applies the source's sinusoidal opacity envelope;
and draws the central probe on frame indices 5 through 10 plus the four location
markers. Each output is a lossless FFV1 AVI with 16 frames at 30 Hz.
Where an aperture extends past the source image, the renderer uses the display's
gray background. It does not repeat the nearest image-edge pixels.

This is a raster reconstruction of the target interval. OpenCV interpolation,
8-bit rasterization and dot boundaries are explicitly specified; they are not
claimed to match PsychoPy's GPU rasterization pixel for pixel. Monitor
calibration, visual angle, the cursor/ring interface, feedback, blanks and
repeated interactive noise-matching cycles are not reproduced. The model
experiment requests a numerical motion report instead of adjustment with a
mouse. It is an adaptation with matched measured quantities.

The code's 16/30-second target differs from the paper's rounded 500-ms
description. The model source-selection path in the authors' inference script
also trims to 15 frames. This plugin follows the published human presentation
code's 16 frames, and records that choice.

The preprocessing notebook orders human rows by scene 1 through 40, reverses
the four flips, and averages x/y vectors separately. The loader checks this
alignment using every scene's archived GT coordinates before assigning IDs.
The archived GT is source optical flow multiplied by 1.5/sqrt(2), because the
trial-generation script divided its radius by sqrt(2). This is retained as
archived GT and is not relabeled as physical pixel displacement. Human reported
velocities remain in their recorded displayed-pixels-per-frame units.

Conditions retain the authors' modulation IDs 1..7; no unverified ID-to-name
mapping is invented. Comparison averages model vectors across flips before
calculating direction or speed. All four valid answers are required.

## Files

- data_packaging/fetch.py: pinned source acquisition and source manifest.
- data_packaging/prepare.py: reference alignment, rendering and packaging.
- data.py: checked stimulus and human-assembly loading.
- __init__.py: lazy dataset, stimulus-set and data-builder registration.
- ../../../tests/test_sun2025.py: parser, identity, reference and integration checks.

There are no macaque or human neural recordings in this plugin. Optional model
neural recordings are measurements for inspection, without a neural alignment
score.
