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

Calling a held-out benchmark directly raises `ValueError`, because the data has no ground truth. `generate_predictions` fits a ridge encoder on the same subject's Friends data. It writes one `sub-<NN>_<split>.npy` file of shape `(n_TRs, 1000)`. Zip these files in the Codabench layout and upload them. The repository ships no zip bundler.

## What the benchmark asks of a model

Each benchmark declares its input channels and requested output channels, and uses `trial_memory = 'none'`. The `form` parameter of the benchmark class selects the form:

- `form='image'` (default) declares `vision:image`. The benchmark sends the frame at the middle of each TR. All identifiers except the two variants of participant 1 use this form.
- `form='video'` declares `vision:video`. The benchmark sends the clip of each TR: the movie from `t x 1.49 s` to `(t+1) x 1.49 s`, cut at 224 x 224 with 30 frames. The identifier ends in `-video`. A model that takes only images gets N/A on this variant, and a model that takes only video gets N/A on the others.
- `form='video+audio'` declares `vision:video` and `audio`. Each row of the table holds the clip of one TR and the sound of the same 1.49 s. The benchmark requests `neural:IT` and `neural:A1`. The identifier ends in `-video-audio`. A model that cannot give both outputs gets N/A.

The image and video forms request `neural:IT` only. The video form sends no sound and no speech. The score of the video form carries `pipeline = algonauts_<mode>_clip` and `form = video`.

Clips are cut once per movie with one ffmpeg call and kept in `<assembly_root>/clips/<movie>/<t>.mp4`. Each clip starts on a key frame, and all clips have the same frame count and duration. Where a movie ends before the last TR of its scan, the missing clips are copies of the last whole clip.

The sound is cut the same way into `<assembly_root>/sound/<movie>/<t>.wav`: 16 kHz, mono, 23840 samples per chunk. Chunk `t` is the sound from `t x 1.49 s` to `(t+1) x 1.49 s`, counted from the start of the sound track. A missing chunk at the end of a movie is a copy of the last whole chunk. If a clip or a chunk could not be cut, the benchmark raises `FileNotFoundError` before a model reads anything.

### The video-and-sound form

The video band comes from `neural:IT` and the audio band from `neural:A1`. The benchmark projects each band to at most 1000 features and stacks and shifts each band in the same way. The default mode of this form is `banded`: banded ridge picks one penalty per band. The score carries `pipeline = algonauts_<mode>_clip_sound`, `form = video+audio`, `band_channels` and the penalties chosen in each fold.

The benchmark can present the table in two ways. With `streaming=False` (default) the subject gets the whole table at once. With `streaming=True` the subject gets one row at a time and answers it before it gets the next row. A subject without trial memory gives the same features both ways. The score records the way in `presentation` (`table` or `stream`). Only this form takes `streaming=True`.

For a short run, `movies` limits the benchmark to named movies. Use the `stimulus_id` values of the assembly, for example `['friends_s01e01a', 'friends_s01e01b']`. Only those movies are cut and scored, and the score lists them in `movies`. Scoring holds out whole runs in 5 folds, so the movies must give at least 5 runs. Banded mode needs at least 7, because it splits each training fold into 5 again. With fewer, the benchmark raises `ValueError`.

For every form, the benchmark stacks `stimulus_window` TRs (default 5), shifts them by `hrf_delay` TRs (default 3), fits a ridge regression per subject and reports the median per-parcel Pearson r.

`mode` selects the feature blocks: `video_only` (default for the image and video forms), `audio_only`, `language_only`, `concat`, `per_modality` or `banded` (default for the video-and-sound form). A mode that needs a band the form does not give raises an error. No form gives a language band yet. The published Wav2Vec2 and MiniLM baseline reproduction is not part of the maintained benchmark.

Feature projection: `generate_predictions()` takes the image and video forms only. It fits the SVD on Friends training frames and applies the same transform to S7 and OOD frames. The Friends cross-validation path fits the SVD on all input frames before it splits runs. That step is transductive, so the score carries `feature_projection_protocol=transductive_svd_all_input_frames`. Neither path fits the SVD to held-out BOLD targets. A held-out prediction must use the same subject, stimulus window and HRF delay as its training encoder.

## Files

- `benchmark.py`: the `Algonauts2025Friends`, `Algonauts2025FriendsS7` and `Algonauts2025OOD` classes, and the frame extractor, clip cutter and sound cutter.
- `projection.py`, `scoring.py`: the SVD projection and the scoring modes.
- `brainscore/data/algonauts2025/data_packaging/prepare_assembly.py`: builds the assemblies from the native files. See the data [README](../../data/algonauts2025/README.md).

## Data

Without prepared data, a registered benchmark raises `FileNotFoundError`. Prepare the data first with the command in the data README. Each assembly has the shape `(time_bin, neuroid)` with 1000 Schaefer parcels per subject. Metadata maps each TR to its movie, episode, split and run.

## Choices to know about

1. Each subject gets its own ridge fits, so the compute is four times that of a pooled fit.
2. The default `stimulus_window=5` and `hrf_delay=3` follow the challenge baseline. You can use `hrf_convolve` from `brainscore.temporal` instead of the fixed shift.
3. Codabench allows one submission per cycle on the OOD set. Iterate on Friends S7 first.

## Whole model to whole brain

`Algonauts2025-friends-sub01-wholebrain` is the video and sound form with `record='all'`. The benchmark asks for `neural:all`. The model answers with every layer in its region map, in one pass, and one regression predicts all brain parcels from all of them. No model region stands for a brain region here. `all` means every mapped layer, not every layer of the network: a model that wants to offer more layers maps more.
