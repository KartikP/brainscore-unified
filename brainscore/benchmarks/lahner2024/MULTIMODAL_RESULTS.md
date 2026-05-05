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

### Whole-cortex vs visual-ROI (concat mode only)

| Variant | Video-only (vjepa1-vitl) | Multimodal (vjepa1-wav2vec2, concat) | Δ raw r |
|---|---|---|---|
| Whole cortex | 0.0832 | 0.0648 | −0.018 (−22%) |
| Visual ROI (rel ≥ 0.3) | 0.5329 | 0.4613 | −0.072 (−13%) |

### Four-mode decomposition on visual-ROI

| Mode | What ridge sees | Visual-ROI raw r | Δ vs video-only |
|---|---|---|---|
| **video_only** | 1024 video features | **0.5325** | — |
| audio_only | 768 audio features | 0.0602 | (near-zero) |
| concat | 1792 [video\|audio] | 0.4613 | −0.071 |
| per_modality | sep ridges, summed | 0.4330 | **−0.099 (worst)** |

The video_only score (0.5325) matches the V-JEPA v1 standalone baseline
(0.5329 in CLAUDE.md) within rounding — confirms the multimodal
benchmark's video pipeline reproduces the existing video-only path.

Audio features carry essentially no information about visual-ROI BOLD
on Lahner stimuli (audio_only ≈ 0.06).

**Surprise:** per-modality ridge underperforms concat. With α=1.0
forced equal across modalities, audio's separate ridge fits training
noise and contributes test-set predictions whose variance is unrelated
to y; adding those to video's well-tuned predictions lowers
correlation. Concat dilutes but at least implicitly down-weights audio
columns during the joint fit; per-modality gives audio a full
α-budget for noise.

## Interpretation

**Both naive multimodal modes lose to video-only on visual cortex.**

- **concat** (joint ridge on `[video|audio]`) suffers feature-space
  dilution: 768 audio columns absorb α-budget that would otherwise
  shrink video columns toward signal. Net: −0.071.
- **per_modality** (separate ridges, summed predictions) suffers
  noise injection: audio's ridge fits training noise, test-set
  audio predictions have variance unrelated to y, adding them to
  video's clean predictions perturbs the joint output. Net: −0.099,
  *worse* than concat.

The clean diagnosis: **flat α across modalities of asymmetric utility
is wrong.** Banded ridge with per-modality α tuned via cross-
validation (Nunez-Elizalde 2019; Gallant lab `himalaya`) would shrink
audio toward zero contribution and let video do its job. The natural
fifth mode for this benchmark.

**What this does NOT prove:** does not prove audio is uninformative
for brain activity in general. Lahner targets visual cortex; auditory
voxels are mostly outside the visual-ROI mask; whole-cortex voxels
are dominated by noise. An auditory-ROI variant would likely flip
the asymmetry.

**What it does prove:** the unified-interface multimodal pipeline
runs end-to-end, produces a reproducible number, and the four-mode
decomposition is interpretable. Flagging a real, well-known failure
mode (joint ridge on grouped features of asymmetric utility) is a
useful empirical contribution; the fix (banded ridge) is the next
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

Then the four-mode driver:

```bash
# 3. Score 4 modes on visual-ROI (cached features → ~3.5 min)
python -u -m brainscore.benchmarks.lahner2024.score_multimodal_modes
# writes /tmp/lahner_multimodal_modes.json
```

Reuses the per-modality activation caches written by step 2.
