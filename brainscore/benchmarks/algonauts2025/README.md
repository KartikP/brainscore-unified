# Algonauts 2025 benchmark

The benchmark predicts fMRI responses to multimodal naturalistic movies. The data come from four CNeuroMod subjects (1, 2, 3 and 5) with a 1000-parcel Schaefer target and TR = 1.49 s. The challenge ships audio, video and transcripts. The post-challenge leaderboard on Codabench stays open.

## Registered benchmarks

`__init__.py` registers 13 identifiers: three splits for each of four subjects, and one video variant.

| Identifier | Split | How to use it |
|---|---|---|
| `Algonauts2025-friends-sub{01,02,03,05}` | Friends S1-S6 and Movie10, with measured fMRI | `benchmark(model)` scores the model with cross-validation. |
| `Algonauts2025-friends-s7-sub{...}` | Friends S7, held out | Call `generate_predictions(model, out_dir)`. |
| `Algonauts2025-ood-sub{...}` | Out-of-distribution movies, held out | Call `generate_predictions(model, out_dir)`. |
| `Algonauts2025-friends-sub01-video` | Friends S1-S6 and Movie10 for participant 1, shown as clips | `benchmark(model)` scores the model with cross-validation. |

Calling a held-out benchmark directly raises `ValueError`, because the data has no ground truth. `generate_predictions` fits a ridge encoder on the same subject's Friends data. It writes one `sub-<NN>_<split>.npy` file of shape `(n_TRs, 1000)`. Zip these files in the Codabench layout and upload them. The repository ships no zip bundler.

## What the benchmark asks of a model

Each benchmark declares one visual form as its required input channel. It requests the `neural:IT` output channel and uses `trial_memory = 'none'`. The `form` parameter of the benchmark class selects the form:

- `form='image'` (default) declares `vision:image`. The benchmark sends the frame at the middle of each TR. All identifiers except the video variant use this form.
- `form='video'` declares `vision:video`. The benchmark sends the clip of each TR: the movie from `t x 1.49 s` to `(t+1) x 1.49 s`, cut at 224 x 224 with 30 frames. The identifier ends in `-video`. A model that takes only images gets N/A on this variant, and a model that takes only video gets N/A on the others.

The video form sends no sound and no speech yet. The people in the scanner heard the movie, so a variant with sound and speech will follow as a separate identifier. The score of the video form carries `pipeline = algonauts_<mode>_clip` and `form = video`.

Clips are cut once per movie with one ffmpeg call and kept in `<assembly_root>/clips/<movie>/<t>.mp4`. Each clip starts on a key frame, and all clips have the same frame count and duration. Where a movie ends before the last TR of its scan, the missing clips are copies of the last whole clip.

For both forms, the benchmark stacks `stimulus_window` TRs (default 5), shifts them by `hrf_delay` TRs (default 3), fits a ridge regression per subject and reports the median per-parcel Pearson r.

`mode` selects the feature blocks: `video_only` (default), `audio_only`, `language_only`, `concat`, `per_modality` or `banded`. The modes that need audio or language features raise an error until the model supplies those blocks. The published Wav2Vec2 and MiniLM baseline reproduction is not part of the maintained benchmark.

Feature projection: `generate_predictions()` fits the SVD on Friends training frames and applies the same transform to S7 and OOD frames. The Friends cross-validation path fits the SVD on all input frames before it splits runs. That step is transductive, so the score carries `feature_projection_protocol=transductive_svd_all_input_frames`. Neither path fits the SVD to held-out BOLD targets. A held-out prediction must use the same subject, stimulus window and HRF delay as its training encoder.

## Files

- `benchmark.py`: the `Algonauts2025Friends`, `Algonauts2025FriendsS7` and `Algonauts2025OOD` classes, and the frame extractor and clip cutter.
- `projection.py`, `scoring.py`: the SVD projection and the scoring modes.
- `brainscore/data/algonauts2025/data_packaging/prepare_assembly.py`: builds the assemblies from the native files. See the data [README](../../data/algonauts2025/README.md).

## Data

Without prepared data, a registered benchmark raises `FileNotFoundError`. Prepare the data first with the command in the data README. Each assembly has the shape `(time_bin, neuroid)` with 1000 Schaefer parcels per subject. Metadata maps each TR to its movie, episode, split and run.

## Choices to know about

1. Each subject gets its own ridge fits, so the compute is four times that of a pooled fit.
2. The default `stimulus_window=5` and `hrf_delay=3` follow the challenge baseline. You can use `hrf_convolve` from `brainscore.temporal` instead of the fixed shift.
3. Codabench allows one submission per cycle on the OOD set. Iterate on Friends S7 first.
