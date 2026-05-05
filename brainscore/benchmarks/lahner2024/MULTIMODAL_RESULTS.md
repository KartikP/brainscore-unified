# Lahner2024 multimodal A+V — first multimodal Brain-Score result

First benchmark in the unified interface to score a model with TWO
distinct preprocessors (video + audio) end-to-end against real fMRI.
Validates the M12-full API surface (multi-region recording,
cross-tower routing, AudioWrapper auto-chunking and temporal coords)
on real model forward passes against real BOLD data.

## Setup

- **Dataset:** Lahner et al. (2024) BoldMoments. 1026 short (3 s) video
  clips with synced audio (1007 with real audio, 19 with silent
  placeholders inserted by `prepare_audio_tracks.py`).
- **Neural data:** per-clip GLM-beta BOLD on fsaverage5 (20 484
  cortical voxels). 10 repetitions averaged per clip.
- **Visual-ROI mask:** split-half reliability ≥ 0.3 (4042 voxels in
  visual cortex; same mask used in the published video-only baseline).
- **Models compared:**
    - **vjepa1-vitl** (video-only, baseline) — V-JEPA v1 ViT-L/16 at
      `backbone.blocks.16` → 1024 features per clip. Score from
      CLAUDE.md (2026-04-27 scoring run).
    - **vjepa1-wav2vec2** (multimodal A+V) — same V-JEPA video tower
      + Wav2Vec2-base audio at `encoder.layers.6` → 1024 + 768 = 1792
      features. Audio extracted at 16 kHz mono via
      `prepare_audio_tracks.py` (≈96 MB cached as WAV files).
- **Scoring:** 5-fold Ridge with α=1.0, no StandardScaler (consistent
  with the existing Lahner pipeline). Per-voxel Pearson on held-out
  predictions, median across voxels.

## Results

| Variant | Video-only (vjepa1-vitl) | Multimodal (vjepa1-wav2vec2) | Δ raw r | Relative |
|---|---|---|---|---|
| Whole cortex | 0.0832 | 0.0648 | **−0.018** | −22% |
| Visual ROI (rel ≥ 0.3) | 0.5329 | 0.4613 | **−0.072** | −13% |

Audio features lower brain-score on BOTH variants. The visual-ROI
result is the most diagnostic — those voxels are deliberately filtered
to visual cortex, where audio carries no targeted predictive signal,
so the additional 768 audio features dilute ridge's capacity on the
1024 useful video features.

## Interpretation

**Concatenated multimodal features hurt for visually-driven stimuli.**
Ridge regularization treats every feature dimension symmetrically; on
visual cortex, the audio columns of the design matrix do not improve
held-out prediction yet they consume α-budget that ridge would
otherwise spend on the predictive video columns.

This is consistent with how multimodal feature fusion works at the
encoding-model layer in the absence of dataset-targeted gating —
adding modality-irrelevant features hurts when the feature pool is
much larger than the sample count (here 1792 features × 1026 clips,
ridge α = 1.0 is global, no per-modality scaling).

**What this does NOT prove:** it does not prove audio is uninformative
about brain activity in general. The Lahner clips were curated for
visual content and many silent activities; auditory cortex voxels are
mostly NOT in the visual-ROI mask; whole-cortex voxels are dominated
by noise. A benchmark that targets auditory ROI (e.g.,
voxels-with-A1-overlap) or that models per-modality gating would
likely show audio helping.

**What it does prove:** the unified-interface multimodal scoring
pipeline runs end-to-end, produces a reproducible number, and the
result is interpretable rather than artifactual. The comparison flags
a real failure mode — ridge-on-concat — that motivates the next
experiment.

## Significance for the unified interface

This is the **first** Brain-Score benchmark in the unified interface
to exercise:

- `BrainScoreModel(preprocessors={'video': ..., 'audio': ...})` — a
  single model holding two distinct backbones with their own wrappers
- `region_modality_map` — cross-tower routing (`IT → video`,
  `A1 → audio`)
- `start_recording('A1', ...)` then `start_recording('IT', ...)` —
  region switching across two `process()` calls within one benchmark
- AudioWrapper auto-chunking + per-step temporal coords (the
  wrapper-chunking work shipped earlier this session)

Two AudioWrapper bugs were fixed during this session as a direct
consequence of running the benchmark:

1. **`_attach_stimulus_set_meta` MultiIndex collision** (same as the
   TextWrapper one fixed earlier) — `assign_coords(stimulus_id=...)`
   collides with a presentation MultiIndex when multiple presentation
   coords exist. Fix: reset_index before re-assigning.
2. **Variable T_hook across batches** — pre-allocation buffer was
   sized from the FIRST batch, but later batches with slightly longer
   inputs (3.007 s vs 3.008 s clips → 150 vs 151 timesteps) couldn't
   fit. Fix: pre-pad all waveforms to a uniform length before batching.

## Reproducing

Local Mac MPS (Apple Silicon):

```bash
# 1. Pre-extract audio tracks (one-time, ~3 min, ~96 MB)
cd unified
conda activate brainscore-unified
python -m brainscore.benchmarks.lahner2024.prepare_audio_tracks \
  --audio-dir ~/.brainio/lahner2024_audio_16k --target-rate 16000

# 2. Score (~12 min on Mac MPS)
python -m brainscore.benchmarks.lahner2024.score_multimodal
# writes /tmp/lahner_multimodal_score.json
```

Total elapsed in our run: 697 s (12 min). The first ~660 s is
Wav2Vec2 + V-JEPA forward passes (cached for subsequent runs); the
last ~30 s is per-voxel ridge + Pearson.
