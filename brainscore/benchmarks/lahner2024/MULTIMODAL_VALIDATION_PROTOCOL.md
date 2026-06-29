# Validation protocol for multimodal Brain-Score benchmarks

A multimodal A+V (or A+V+T) benchmark is "trusted" only when each of
the criteria below is satisfied. Run this every time a new multimodal
benchmark variant or a new audio/video backbone is added.

The same protocol applies to both the GLM-beta variant
(`Lahner2024-fMRI-naturalistic-multimodal-*`) and the TR-resolved
variant (`Lahner2024-fMRI-naturalistic-timeresolved-multimodal-*`).

## Why this protocol exists

The GLM-beta multimodal scoring on Lahner visual-ROI initially
appeared to show banded ridge beating video-only by +0.044 — a
seemingly clean multimodal lift. The α-ablation overturned that
reading entirely: the gain came from better-tuned video α, not from
audio. Without a structured validation protocol, this kind of
artifact slips through. The protocol below is what would have caught
it on day one.

## The five validation tests

### 1. Null floor — chance + double-null at zero

The chance baseline (uniform predictions) should give raw r ≈ 0 on
any voxel-prediction benchmark. The double-null
(`random-vjepa1-random-wav2vec2`) — both towers random-init — should
also be near zero, with the residual r reflecting whatever ridge can
overfit from random features under our 5-fold CV.

**Pass criterion:** raw r < 0.05 for both controls on visual-ROI and
auditory-ROI variants.

### 2. Single-null asymmetry — one tower informative, the other not

`random-vjepa1-wav2vec2` (null video + signal audio) should score
roughly equal to `audio_only` of the full model on both ROIs.
Symmetrically, `vjepa1-random-wav2vec2` (signal video + null audio)
should score roughly equal to `video_only` of the full model.

**Pass criterion:** |single-null score − corresponding single-tower
score of the signal model| < 0.02. Tests that the null random-init
of one tower truly zeroes out that modality's contribution while the
other tower still extracts what it normally would.

### 3. Mode-curve coherence — concat dilutes, banded matches video α

For ANY signal-bearing model on ANY ROI:
- `concat(α=1)` should be **worse** than `video_only(α=1)` if audio
  carries less target-aligned signal than video (visual-ROI case).
  Reverse if audio dominates (no current ROI has shown this).
- `per_modality(α=1)` should be **worse than concat** for the same
  reason — naive sum injects noise predictions.
- `banded` should match or beat the better single-tower α-tuned
  baseline. If banded < `video_only(α=10)` at video's optimum α, the
  banded grid was too coarse OR α tuning is unstable.

**Pass criterion:** the ordering `per_modality < concat < banded ≥
max(video_only(α*), audio_only(α*))` holds within 0.01 raw r for the
signal-bearing model on at least one ROI.

### 4. Region × modality specificity

On visual-ROI the video tower should dominate; on auditory-ROI the
audio tower should at least catch up to the video tower (if not
overtake it — Lahner's auditory-ROI has audiovisual-integration regions
so video may still lead).

**Pass criterion:** (`video_only_visualROI` − `video_only_auditoryROI`)
> (`video_only_visualROI` − `audio_only_visualROI`). I.e., dropping the
ROI to auditory-cortex should hurt video more than dropping the
modality to audio-only on the visual ROI. This catches the case where
the benchmark has no ROI specificity (e.g., a bug returning the same
mask for both variants).

### 5. Reproducibility — bit-for-bit on rerun

Score the same model twice with the cache cleared between runs. raw r
should be identical to 4 decimals. This catches non-determinism in
the audio loader, the V-JEPA frame sampler, the chunking, or the inner
banded-ridge α-selection.

**Pass criterion:** identical raw r across two cold-cache runs.

## How to run

```bash
cd unified
conda activate brainscore-unified

# Step 1: pre-extract audio (one-time, ~3 min)
python -m brainscore.data.lahner2024.prepare_audio_tracks \
  --audio-dir ~/.brainio/lahner2024_audio_16k --target-rate 16000

# Step 2: validation harness (runs all checks for both ROIs)
python -m brainscore.benchmarks.lahner2024.validate_multimodal
# Writes /tmp/lahner_multimodal_validation.json with pass/fail per check.
```

## What this protocol does NOT validate

- **Cross-stimulus generalization.** Lahner has 1026 stimuli, all
  3-second clips. Doesn't test long-context or naturalistic-narrative
  representations.
- **Noise-ceiling normalization.** Raw r is reported; tighter
  benchmarks would normalize against a per-voxel split-half ceiling.
- **Per-voxel banded α.** Current banded uses a single shared
  (α_v, α_a) across all voxels. Some voxels may benefit from
  per-voxel tuning (Gallant lab's `himalaya` does this).
- **More than 2 modalities.** A+V only. M12-full's third tower (text
  / transcript) needs its own validation pass once added.
