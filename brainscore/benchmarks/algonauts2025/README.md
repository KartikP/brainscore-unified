# Algonauts 2025 benchmark

Predicts fMRI responses to multimodal naturalistic movies. Three modalities
(audio + visual + transcript), four CNeuroMod subjects, 1000-parcel Schaefer
target. Post-challenge leaderboard open indefinitely on Codabench.

## Status

**Scaffold only — no data, no model scoring yet.** Files in this directory
define the benchmark structure; data acquisition + assembly preparation
happen on EC2 (see below). Running any of the registered benchmarks
without the data present will raise `FileNotFoundError`.

## Phase plan

1. **Data acquisition** (EC2, 1–6 hours). Run
   `unified/scripts/download_algonauts_data.sh` on the validation
   instance. Pulls ~100 GB across stimuli + 4-subject fMRI .h5 files.
2. **Assembly preparation** (EC2, ~30 min).
   `python -m brainscore.data.algonauts2025.prepare_assembly`
   reads the per-subject .h5 files, converts to a brainio
   `NeuroidAssembly` keyed by (subject, movie_split, TR), uploads to
   our S3 bucket. One-time cost.
3. **Baseline reproduction** (EC2, ~1 hour). Reproduce the SlowR50 +
   librosa + text-embedding baseline that the challenge organizers
   ship. Calibration check: our scores should match theirs within
   rounding.
4. **Our models** (EC2, ~2 hours). Score V-JEPA + Wav2Vec2 + GPT-2 (or
   a stronger LM) via banded ridge through the multimodal pipeline.
5. **Codabench submission** (local, instant). Format predicted Friends
   S7 and OOD parcel arrays, upload .npy archive to Codabench.

## Why this benchmark

It's the M12-full target — long-form continuous-naturalistic A+V+T
fMRI prediction. All the wrappers, banded ridge, temporal alignment,
and validation protocol we built for Lahner generalize directly.

| Lahner (already done) | Algonauts 2025 |
|---|---|
| 3-second clips, GLM-betas | Continuous TRs, time-series |
| 1026 stimuli | ~65 hours of stimuli |
| 1 dataset | Friends + Movie10 + OOD |
| Audio + Video | Audio + Video + **Transcript** |
| 10 subjects, cross-subject CV | 4 subjects, **per-subject** training |
| fsaverage5 surface | Schaefer 1000-parcel |

## Files (what each thing does)

- `benchmark.py` — `Algonauts2025Friends`, `Algonauts2025FriendsS7`,
  `Algonauts2025OOD` classes. Inherit from `BenchmarkBase`. Score via
  banded ridge with three feature groups (V/A/L). Stub `__call__`
  raises NotImplementedError until assembly is built.
- `brainscore/data/algonauts2025/prepare_assembly.py` — One-time data
  pipeline. Run on EC2 after download. Reads .h5 files, builds
  `NeuralAssembly`, writes the local data-plugin artifacts.
- `experiments/algonauts2025/submit_codabench.py` — Take a benchmark's
  per-parcel predictions, format as Codabench expects, save .zip for upload.
- `__init__.py` — Registry entries for 3 splits × 4 subjects = 12
  benchmark identifiers.

## Data layout (after `python -m brainscore.data.algonauts2025.prepare_assembly` runs)

```
~/.brainio/<sha>/algonauts2025_friends_sub01.nc   (~5 GB per subject)
~/.brainio/<sha>/algonauts2025_friends_sub02.nc
...
~/.brainio/<sha>/algonauts2025_movie10_sub01.nc
~/.brainio/<sha>/algonauts2025_friends_s7_sub01.nc   (held out — no fMRI)
~/.brainio/<sha>/algonauts2025_ood_sub01.nc          (held out — no fMRI)
~/.brainio/<sha>/algonauts2025_stim_friends.csv      (movie_split metadata)
~/.brainio/<sha>/algonauts2025_transcripts/          (per-TR word lists)
```

Each `.nc` file: `(time_bin, neuroid)` shape with TR=1.49s, 1000 Schaefer
parcels per subject. Stim metadata maps each TR to (movie, episode,
split, run).

## Caveats / open questions

1. **Per-subject training.** Each subject gets their own ridge fits.
   That's standard practice in encoding-model literature but means we
   need 4× the compute compared to cross-subject pooled.
2. **Transcript handling.** The .tsv has per-TR aligned word lists
   with onset times. Two options for the language tower:
   - Run TextWrapper (`per_token` mode) on each TR's word list,
     mean-pool features per TR.
   - Run TextWrapper on a longer context window (e.g., the full
     episode), then slice by word onset to get per-TR features.
   The second is more powerful but more involved. Plan A first.
3. **Stimulus window.** The challenge baseline uses `stimulus_window=5`
   TRs (concatenated 5-TR feature stacks per fMRI sample). Banded
   ridge will scale linearly with feature count; with 5×1024 video +
   5×768 audio + 5×768 text features, X is ~13k columns. Manageable.
4. **HRF delay.** Baseline uses `hrf_delay=3` (shift features 3 TRs
   forward). Or use `hrf_convolve` with a proper double-gamma. Both
   are options — need to pick one before scoring.
5. **OOD test set held out.** We get one shot per submission cycle on
   Codabench. Iterate on Friends-S7 leaderboard first.
