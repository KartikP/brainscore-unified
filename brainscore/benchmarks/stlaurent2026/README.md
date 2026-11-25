# CNeuroMod-THINGS fMRI benchmarks (St-Laurent et al. 2026)

Registered identifiers, version 1: `StLaurent2026-fMRI-things-<region>` for V1, V2, V3, V4, FFA,
OFA, pSTS, PPA, OPA, MPA and EBA.

Participants of the Courtois NeuroMod project saw 4320 THINGS images (720 categories, 6 exemplars
each), each image three times, while they were scanned with fMRI (St-Laurent et al. 2026,
Scientific Data 13:141). The benchmark compares the model's response for one region with the
trial responses of the participants' voxels in that region.

## What it asks

- Input: each image on `vision:image` (10 degrees of visual angle in the scanner).
- Output: `neural:<region>`. A model takes part when its `region_layer_map` names a layer for the
  region; without one, the pair is not applicable and nothing runs. `V4` is the human homolog hV4.
- Trial memory: `none`.

## Data

- The trial responses: `package_data('cneuromod-things', part='sub-01')` (and `sub-02`, `sub-03`),
  see `brainscore/data/cneuromod/README.md`. Brain-Score hosts no brain file.
- The images: `images_THINGS.zip` from https://osf.io/jum2f (academic terms), its `object_images`
  folder at `$BRAINSCORE_DATA_HOME/things-images`.

## The score

Per participant: the images with three answered trials; 576 categories for the fit and 144 for the
score (`test_categories.txt`, one fixed list); the target is the mean of the three viewings; the
voxels of the participant's own mask with a noise ceiling of at least 10 percent on the training
images (at least 50, else the participant does not count, and the score says so); ridge with the
penalty chosen on held-out training categories; Pearson r per voxel on the test images. The value
is the median over voxels of r / c, c = sqrt(noise ceiling / 100) of the voxel on the test images.
The benchmark value is the mean over the participants; each participant's value, raw median r,
median ceiling and counts are on the score (`per_participant`).
