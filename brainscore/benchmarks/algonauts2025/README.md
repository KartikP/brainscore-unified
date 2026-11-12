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

The benchmark requests `neural:all`, the address of every region the model maps. A model with a region map declares it. A model with no region map gets N/A.

The benchmark predicts all 1000 parcels from every layer the model maps. It asks for each `neural:<region>` that the model declares, each region on its own channel, because two towers can answer on different time axes. It averages each answer over its time bins. Each region is one band. No model region stands for a brain region: one regression predicts every parcel from all bands.

The benchmark asks for every declared region. It cannot yet tell which regions a model can answer from the inputs of a form, because no declaration says which input a region needs. A model that declares a region it cannot answer from these inputs (for example a text region on the image form) stops the run with an error before any extraction.

## How the benchmark scores

The method follows multi-layer encoding with banded ridge (himalaya, Dupre la Tour et al. 2022). The benchmark holds out whole runs in 5 folds. In each fold it:

1. reduces each band by itself to 100 components (truncated SVD). It fits the reduction on the rows of the training runs only and applies it to all rows;
2. stacks `stimulus_window` TRs (default 5) per sample and shifts them by `hrf_delay` TRs (default 3), inside each run;
3. drops `excluded_samples_start` and `excluded_samples_end` TRs (default 5 each) at the edges of each run;
4. chooses one penalty per band on held-out runs inside the training runs. The grid is 1 to 10000 in decades. When the full grid has more than 60 combinations, a random search tries 60, among them every combination with one shared penalty;
5. fits the banded ridge on the training runs and predicts the test runs.

With one band, this is a ridge with a tuned penalty. The inner choice uses the reduction fitted on all training runs of the fold, so the inner held-out runs took part in that fit. The test runs of the fold took part in no fit.

The score is the median per-parcel Pearson r. The score lists `bands`, `band_channels`, the penalty of each band in each fold (`chosen_alphas_per_fold`), `components_per_band`, `form`, `presentation` and `feature_projection_protocol = svd_fitted_on_training_runs_per_fold`. `per_parcel_r` holds the score of each parcel in Schaefer order.

The benchmark can present the table in two ways, for every form. With `streaming=False` (default) the subject gets the whole table at once. With `streaming=True` the subject gets one row at a time and answers it before it gets the next row. A subject without trial memory gives the same features both ways. The score records the way in `presentation` (`table` or `stream`).

For a short run, `movies` limits the benchmark to named movies. Use the `stimulus_id` values of the assembly, for example `['friends_s01e01a', 'friends_s01e01b']`. Only those movies are cut and scored, and the score lists them in `movies`. The movies must give at least 7 runs, because each training fold is split into 5 again. With fewer, the benchmark raises `ValueError`.

## Held-out prediction

`generate_predictions(model, out_dir)` uses the same method without the outer folds. It records the same bands on the subject's Friends training films and on the held-out split. It fits the reduction of each band on the training films, chooses the penalties on held-out training runs, fits one banded ridge on all training films and predicts every TR of the held-out split. Training and prediction must use the same subject, form, stimulus window and HRF delay. The result lists the bands and the chosen penalties.

## Files

- `benchmark.py`: the `Algonauts2025Friends`, `Algonauts2025FriendsS7` and `Algonauts2025OOD` classes, and the frame extractor, clip cutter and sound cutter.
- `scoring.py`: the banded ridge with the reduction fitted per fold.
- `projection.py`: the reduction of one band.
- `brainscore/data/algonauts2025/data_packaging/prepare_assembly.py`: builds the assemblies from the native files. See the data [README](../../data/algonauts2025/README.md).

## Data

Without prepared data, a registered benchmark raises `FileNotFoundError`. Prepare the data first with the command in the data README. Each assembly has the shape `(time_bin, neuroid)` with 1000 Schaefer parcels per subject. Metadata maps each TR to its movie, episode, split and run.

## Choices to know about

1. Each subject gets its own ridge fits, so the compute is four times that of a pooled fit.
2. The default `stimulus_window=5` and `hrf_delay=3` follow the challenge baseline. You can use `hrf_convolve` from `brainscore.temporal` instead of the fixed shift.
3. The memory pre-flight plans for at most 10 bands. A model that maps more regions needs more memory than the estimate.
4. Codabench allows one submission per cycle on the OOD set. Iterate on Friends S7 first.
