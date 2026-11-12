# Algonauts 2025 benchmark

The benchmark predicts fMRI responses to multimodal naturalistic movies. The data come from four CNeuroMod subjects (1, 2, 3 and 5) with a 1000-parcel Schaefer target and TR = 1.49 s. The challenge ships audio, video and transcripts. The post-challenge leaderboard on Codabench stays open.

## Registered benchmarks

`__init__.py` registers 14 identifiers: three splits for each of four subjects, one video variant and one video-and-sound variant.

| Identifier | Split | How to use it |
|---|---|---|
| `Algonauts2025-friends-sub{01,02,03,05}` | Friends S1-S6 and Movie10, with measured fMRI | `benchmark(model)` scores the model with cross-validation. |
| `Algonauts2025-friends-s7-sub{...}` | Friends S7, held out | Call `generate_predictions(model, out_dir)`. |
| `Algonauts2025-ood-sub{...}` | Out-of-distribution movies, held out | Call `generate_predictions(model, out_dir)`. |
| `Algonauts2025-friends-sub01-video` | Friends S1-S6 and Movie10 for participant 1, shown as clips | `benchmark(model)` scores the model with cross-validation. |
| `Algonauts2025-friends-sub01-video-audio` | Friends S1-S6 and Movie10 for participant 1, shown as clips with their sound | `benchmark(model)` scores the model with cross-validation. |

Calling a held-out benchmark directly raises `ValueError`, because the data has no ground truth. `generate_predictions` writes one `sub-<NN>_<split>.npy` file of shape `(n_TRs, 1000)`. Zip these files in the Codabench layout and upload them. The repository ships no zip bundler.

## What the benchmark sends

Each benchmark declares its input channels and uses `trial_memory = 'none'`. The `form` parameter selects what the benchmark sends for each TR. The forms differ only in this.

- `form='image'` (default) sends the frame at the middle of each TR on `vision:image`. All identifiers except the two variants of participant 1 use this form.
- `form='video'` sends the clip of each TR on `vision:video`: the movie from `t x 1.49 s` to `(t+1) x 1.49 s`, cut at 224 x 224 with 30 frames. It sends no sound and no speech. The identifier ends in `-video`.
- `form='video+audio'` sends the clip on `vision:video` and the sound of the same 1.49 s on `audio`, in one table row. The identifier ends in `-video-audio`.

A model that takes none of the inputs of a form gets N/A. A model that requires an input the form does not send also gets N/A.

Clips are cut once per movie with one ffmpeg call and kept in `<assembly_root>/clips/<movie>/<t>.mp4`. Each clip starts on a key frame, and all clips have the same frame count and duration. Where a movie ends before the last TR of its scan, the missing clips are copies of the last whole clip.

The sound is cut the same way into `<assembly_root>/sound/<movie>/<t>.wav`: 16 kHz, mono, 23840 samples per chunk. Chunk `t` is the sound from `t x 1.49 s` to `(t+1) x 1.49 s`, counted from the start of the sound track. A missing chunk at the end of a movie is a copy of the last whole chunk. If a clip or a chunk could not be cut, the benchmark raises `FileNotFoundError` before a model reads anything.

## What the benchmark asks for

The benchmark names the regions it requests for each form:

| Form | Requested outputs |
|---|---|
| `image` | `neural:V1`, `neural:V4`, `neural:IT` |
| `video` | `neural:V1`, `neural:V4`, `neural:IT` |
| `video+audio` | `neural:V1`, `neural:V4`, `neural:IT`, `neural:A1` |

A model that does not declare every requested region gets N/A, before anything is cut, both through `score()` and when the benchmark is called directly. A model that declares a region but cannot answer it from the inputs of the form also gets N/A: its `output_inputs` declaration says which inputs each region needs. For example, a video-and-sound model whose `A1` reads sound gets N/A on the `video` form.

The benchmark predicts all 1000 parcels from the model's layers for the regions the benchmark names. It asks for each region in its own session, on its own channel, because two towers can answer on different time axes. It averages each answer over its time bins. Each region is one band. No model region stands for a brain region: one regression predicts every parcel from all bands.

## How the benchmark scores

The method follows multi-layer encoding with banded ridge (himalaya, Dupre la Tour et al. 2022). This is version 2 of the benchmark. The benchmark holds out stimuli in 5 folds. All runs of one stimulus stay on one side, because Movie10 shows some clips twice, in two runs with one `stimulus_id`. In each fold it:

1. prepares each band on the rows of the training stimuli only: it centres the band with their mean, reduces it to at most 100 components with a truncated SVD (a PCA, because the band is centred), and standardizes the components with their mean and spread. It applies the same preparation to all rows;
2. stacks `stimulus_window` TRs (default 5) per sample and shifts them by `hrf_delay` TRs (default 3), inside each run;
3. drops `excluded_samples_start` TRs at the start and `excluded_samples_end` TRs (default 5) at the end of each run. The start default is `hrf_delay + stimulus_window - 1` (7), so no kept sample has an empty lag; a smaller value raises `ValueError`;
4. chooses, for each parcel, one penalty per band on held-out stimuli inside the training stimuli (one inner split). The grid has one value per decade from 1e-2 to 1e8. When the full grid has more than 60 combinations (from two bands on), a random search tries 60, among them every combination with one shared penalty;
5. fits a banded ridge with an intercept on the training stimuli, one solve for each distinct chosen combination, and predicts the test stimuli. All solves are in float64.

With one band, this is a ridge with an intercept and a penalty for each parcel. The inner choice uses the preparation fitted on all training stimuli of the fold, so the inner held-out stimuli took part in that fit. The test stimuli of the fold took part in no fit.

No ceiling is measured. The score is the raw median per-parcel Pearson r, with `normalized = False` and `ceiling_status = not_estimated`. The score also lists:

- `raw`, `metric`, `mean_r`, `n_parcels_scored`, `n_TRs`, and `per_parcel_r` (the score of each parcel in Schaefer order);
- `stimulus_window`, `hrf_delay`, `excluded_samples_start`, `excluded_samples_end` and `cv_groups = stimulus_id`;
- `form`, `presentation`, `bands`, `band_channels` and `components_per_band` (the real number for each band);
- `alpha_grid`, `alpha_candidates`, `penalty_protocol = per_parcel_on_one_inner_split_of_held_out_stimuli` and `feature_projection_protocol = centred_svd_and_scale_fitted_on_training_stimuli`;
- `penalties_per_fold`: for each fold, the number of distinct chosen combinations, the share of parcels with a penalty at an edge of the grid, and each combination with its number of parcels.

The benchmark can present the table in two ways, for every form. With `streaming=False` (default) the subject gets the whole table at once. With `streaming=True` the subject gets one row at a time and answers it before it gets the next row. A subject without trial memory gives the same features both ways. The score records the way in `presentation` (`table` or `stream`).

For a short run, `movies` limits the benchmark to named movies. Use the `stimulus_id` values of the assembly, for example `['friends_s01e01a', 'friends_s01e01b']`. Only those movies are cut and scored, and the score lists them in `movies`. The movies must give at least 7 stimuli, because each training fold is split into 5 again. With fewer, the benchmark raises `ValueError`.

## Held-out prediction

`generate_predictions(model, out_dir)` uses the same method without the outer folds. It records the same bands on the subject's Friends training films and on the held-out split. It fits the preparation of each band on the training films, chooses the penalties for each parcel on held-out training stimuli, fits one banded ridge on all training films and predicts every TR of the held-out split. Training and prediction must use the same subject, form, stimulus window, HRF delay and `streaming`. The `movies` of the training benchmark limit the training films; the `movies` of the held-out benchmark limit the predicted films. The result lists the bands and the chosen penalties.

## Files

- `benchmark.py`: the `Algonauts2025Friends`, `Algonauts2025FriendsS7` and `Algonauts2025OOD` classes, and the frame extractor, clip cutter and sound cutter.
- `scoring.py`: the banded ridge with the preparation fitted per fold.
- `projection.py`: the reduction of one band.
- `brainscore/data/algonauts2025/data_packaging/prepare_assembly.py`: builds the assemblies from the native files. See the data [README](../../data/algonauts2025/README.md).

## Data

Without prepared data, a registered benchmark raises `FileNotFoundError`. Prepare the data first with the command in the data README. Each assembly has the shape `(time_bin, neuroid)` with 1000 Schaefer parcels per subject. Metadata maps each TR to its movie, episode, split and run.

## Choices to know about

1. Each subject gets its own ridge fits, so the compute is four times that of a pooled fit.
2. The default `stimulus_window=5` and `hrf_delay=3` follow the challenge baseline. You can use `hrf_convolve` from `brainscore.temporal` instead of the fixed shift.
3. The memory pre-flight probes the width of one region and plans one band per requested region.
4. Codabench allows one submission per cycle on the OOD set. Iterate on Friends S7 first.
