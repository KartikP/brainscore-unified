"""
Lahner 2024 BOLDMoments — TR-resolved fMRI variant.

Same stimuli as the existing GLM-beta variant (Lahner2024BOLDMoments), but
predicts the per-TR BOLD time-series rather than per-clip beta estimates.
This is the first benchmark to actually exercise the temporal kit
(`temporal_bin`, `hrf_convolve`, `contiguous_block_cv`, absolute timestamps)
end-to-end.

Assembly shape (from `prepare_timeresolved_assembly.py` / S3):
    (time_bin = N_TR_per_clip,  neuroid = 20484,  presentation = 10260)

where:
    - presentation = 1026 stimuli × 10 reps  (matches GLM-beta variant)
    - neuroid = 20484 fsaverage5 cortical vertices
    - time_bin = N_TR_per_clip TR-aligned bins covering each 3 s clip + HRF tail

## Pipeline (per model):

    1. Build a video stimulus set (one row per video; reuses parent class helper).
    2. model.start_recording('IT', time_bins=<TR-aligned bins>) and process(stim)
       → per-clip activations at the model's native rate.
    3. HRF-convolve the model features along the time axis (model rate).
    4. Resample model time bins to TR rate via temporal_bin (or linear interp).
    5. Contiguous-block CV over clips (block_size = 10 clips, ~10s scanner block).
    6. Per-(voxel, TR) ridge regression.
    7. Per-voxel Pearson averaged across TRs → median across voxels.

## Why this is M12-lite

This is single-region, single-modality, single-subject-axis (averaged across
the 10 BOLDMoments subjects). It validates the entire temporal kit on real
TR-resolved fMRI without the multi-region or multi-modal complexity of the
full M12 (cross-modal coherence) benchmark. If this works end-to-end, M12-full
is an additive extension, not a fundamental rewrite.

## Comparison story

Same 1026 videos, same 6 models (V-JEPA v1, V-JEPA v2, VideoMAE, CLIP-frame,
Qwen, BLIP-2), but the model has to predict a TR-resolved trajectory rather
than a single per-clip beta. Score delta between the two variants quantifies
how much temporal information the model adds (or how much the GLM beta
already captured).
"""

from typing import List, Optional

import numpy as np
import xarray as xr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuronRecordingAssembly,
)
from brainscore_core.supported_data_standards.brainio.s3 import (
    load_assembly_from_s3,
)
from brainscore_core.temporal import (
    contiguous_block_cv,
    double_gamma_hrf,
    hrf_convolve,
    temporal_bin,
)

# Reuse stimulus loader + bibtex + S3 paths from the existing variant
from .benchmark import (
    BIBTEX,
    STIMULUS_BUCKET,
    VIDEO_DURATION_MS,
    Lahner2024BOLDMoments,        # parent class — provides _videos_stimulus_set, _expand_videos
    load_stimulus_set,
)


# ── S3 metadata for the TR-resolved assembly ──────────────────────────
# These get filled in once `prepare_timeresolved_assembly.py` runs on EC2
# and uploads the artifact. Until then, loading raises explicitly.

TIMERESOLVED_ASSEMBLY_ID = 'Lahner2024-fMRI-timeresolved'
TIMERESOLVED_ASSEMBLY_VERSION_ID: Optional[str] = None    # TODO: fill from prep script output
TIMERESOLVED_ASSEMBLY_SHA1: Optional[str] = None          # TODO: fill from prep script output

# Matches what prepare_timeresolved_assembly.py picks. We store as a
# constant here too so the benchmark can sanity-check the loaded assembly.
EXPECTED_PER_CLIP_WINDOW_SEC = 12.0


# ── Loader ────────────────────────────────────────────────────────────

def load_timeresolved_assembly(merge_stimulus_set_meta: bool = True) -> NeuronRecordingAssembly:
    if TIMERESOLVED_ASSEMBLY_VERSION_ID is None or TIMERESOLVED_ASSEMBLY_SHA1 is None:
        raise RuntimeError(
            "Lahner2024 TR-resolved assembly is not yet hosted on S3. "
            "Run `prepare_timeresolved_assembly.py` on EC2 to download from "
            "OpenNeuro ds005165, build the assembly, and upload — then paste "
            "the resulting (version_id, sha1) into benchmark_timeresolved.py."
        )
    return load_assembly_from_s3(
        identifier=TIMERESOLVED_ASSEMBLY_ID,
        version_id=TIMERESOLVED_ASSEMBLY_VERSION_ID,
        sha1=TIMERESOLVED_ASSEMBLY_SHA1,
        bucket=STIMULUS_BUCKET,
        cls=NeuronRecordingAssembly,
        stimulus_set_loader=load_stimulus_set,
        merge_stimulus_set_meta=merge_stimulus_set_meta,
    )


# ── Benchmark class ───────────────────────────────────────────────────

class Lahner2024BOLDMoments_timeresolved(BenchmarkBase):
    """Predict per-TR BOLD time-series for each 3 s BoldMoments clip.

    Inherits stimulus-handling helpers from `Lahner2024BOLDMoments`
    (the GLM-beta variant) — stimuli are identical between the two.
    Only the neural target and the regression machinery differ.
    """

    def __init__(
        self,
        ceiling: Optional[float] = None,
        reliability_threshold: Optional[float] = None,
        cv_block_size_clips: int = 10,
        identifier_suffix: str = '-timeresolved',
        # Allow the prep script to be re-run with a different window;
        # this constant pins the runtime expectation so we error if mismatched.
        expected_per_clip_window_sec: float = EXPECTED_PER_CLIP_WINDOW_SEC,
    ):
        self._reliability_threshold = reliability_threshold
        self._cv_block_size_clips = cv_block_size_clips
        self._expected_per_clip_window_sec = expected_per_clip_window_sec
        self._assembly: Optional[NeuronRecordingAssembly] = None
        self._stimulus_set = None
        self._voxel_mask: Optional[np.ndarray] = None

        # We delegate stimulus prep to a sibling instance of the GLM-beta variant
        # so we don't duplicate _expand_videos / _videos_stimulus_set logic.
        self._stim_helper = Lahner2024BOLDMoments(reliability_threshold=None)

        super().__init__(
            identifier=f'Lahner2024-fMRI-naturalistic{identifier_suffix}',
            version=1,
            parent='naturalistic',
            ceiling=Score(1.0 if ceiling is None else float(ceiling)),
            bibtex=BIBTEX,
        )

    @property
    def assembly(self) -> NeuronRecordingAssembly:
        if self._assembly is None:
            self._assembly = load_timeresolved_assembly()
            self._sanity_check_assembly(self._assembly)
        return self._assembly

    def _sanity_check_assembly(self, assembly):
        # Verify the loaded assembly matches what the prep script promised.
        assert 'time_bin_start_ms' in assembly.coords, \
            "TR-resolved assembly must have time_bin_start_ms coord"
        assert 'time_bin_end_ms' in assembly.coords, \
            "TR-resolved assembly must have time_bin_end_ms coord"
        n_tr = assembly.sizes['time_bin']
        if n_tr <= 1:
            raise RuntimeError(
                f"TR-resolved assembly has {n_tr} time_bin(s); expected >1. "
                f"Looks like the GLM-beta variant assembly was loaded by mistake."
            )

    @property
    def stimulus_set(self):
        return self._stim_helper.stimulus_set

    def _average_repetitions(self) -> NeuronRecordingAssembly:
        """Average the 10 repetitions per video → one (time_bin, neuroid) per video."""
        a = self.assembly
        # group by stimulus_id, mean over repetitions of presentation axis
        return a.groupby('stimulus_id').mean('presentation')

    def __call__(self, candidate) -> Score:
        # 1. Configure recording. We pass TR-aligned time_bins from the
        #    assembly — the model's VideoWrapper will produce activations at
        #    its native rate; the benchmark resamples to these bins before
        #    regression.
        target_assembly = self._average_repetitions()
        # target shape: (stimulus_id, time_bin, neuroid)  (after groupby)
        # OR (time_bin, neuroid, stimulus_id) depending on dim ordering — handle both.

        target_time_bins = list(zip(
            target_assembly['time_bin_start_ms'].values.tolist(),
            target_assembly['time_bin_end_ms'].values.tolist(),
        ))

        candidate.start_recording('IT', time_bins=target_time_bins)

        # 2. Get model activations. Same dispatch as the GLM-beta variant —
        #    video-native models go straight; image models get the frame-aggregated
        #    pseudo-temporal path.
        if 'video' in getattr(candidate, 'supported_modalities', set()):
            pipeline_mode = 'video_native'
            video_stim = self._stim_helper._videos_stimulus_set()
            model_features = candidate.process(video_stim)
            # model_features dims: (presentation, time_bin_model, neuroid)
            # model_time may not match target_time_bins yet — we'll resample below.
        else:
            pipeline_mode = 'frame_aggregation'
            frame_stim = self._stim_helper._expand_videos()
            per_frame = candidate.process(frame_stim)
            # For image models we need per-clip per-TR features. The simplest
            # consistent approach: use the existing per-clip mean (one feature
            # vector per clip) and broadcast to all TRs. This means image
            # models can't beat their GLM-beta-variant score — they have no
            # temporal information to add.
            per_video = temporal_bin(per_frame, time_bins=[(0, VIDEO_DURATION_MS)])
            model_features = self._broadcast_to_tr_axis(per_video, n_tr=len(target_time_bins))

        # 3. Apply HRF convolution to model features along the time axis.
        #    Sampling rate = 1000 / (mean TR width in ms).
        tr_width_ms = float(np.mean(np.diff(
            np.array([s for s, _ in target_time_bins]))))
        sample_rate_hz = 1000.0 / tr_width_ms
        model_features = self._hrf_convolve_features(model_features,
                                                     sample_rate_hz=sample_rate_hz)

        # 4. Resample model features to the target TR grid via temporal_bin
        #    on absolute timestamps.
        model_features_aligned = self._resample_to_target_grid(
            model_features, target_time_bins=target_time_bins,
            pipeline_mode=pipeline_mode)

        # 5. Per-(voxel, TR) ridge with contiguous-block CV.
        per_voxel_per_tr_r = self._cross_validated_pearson(
            model=model_features_aligned, neural=target_assembly,
            block_size_clips=self._cv_block_size_clips,
        )
        # per_voxel_per_tr_r shape: (n_voxels, n_tr)

        # Optional voxel mask
        if self._reliability_threshold is not None:
            mask = self._stim_helper._get_voxel_mask()  # NB: trained on GLM-beta variant
            # TODO: decide whether to recompute reliability on TR-resolved data
            # OR reuse the GLM-beta-derived mask. Reusing is faster + comparable
            # across variants; recomputing is more rigorous.
            if mask is not None:
                per_voxel_per_tr_r = per_voxel_per_tr_r[mask]

        # 6. Aggregate. Two sensible options:
        #    a) median over voxels of mean-over-TRs → single number
        #    b) median over voxels at the peak HRF TR (~5-6s post-stim) → comparable to GLM-beta
        # We report both.
        mean_over_tr_per_voxel = np.nanmean(per_voxel_per_tr_r, axis=1)
        median_r = float(np.nanmedian(mean_over_tr_per_voxel))

        # Peak-TR score: pick the TR closest to the HRF peak (~5s post stimulus onset)
        peak_tr_idx = self._pick_peak_hrf_tr_idx(target_time_bins, peak_sec=5.0)
        peak_tr_r = per_voxel_per_tr_r[:, peak_tr_idx]
        median_r_peak = float(np.nanmedian(peak_tr_r))

        score = Score(median_r / float(self.ceiling))
        score.attrs['raw'] = Score(median_r)
        score.attrs['median_r_peak_hrf_tr'] = median_r_peak
        score.attrs['peak_hrf_tr_idx'] = int(peak_tr_idx)
        score.attrs['n_tr'] = len(target_time_bins)
        score.attrs['n_voxels_scored'] = int(per_voxel_per_tr_r.shape[0])
        score.attrs['pipeline'] = pipeline_mode
        score.attrs['cv_block_size_clips'] = self._cv_block_size_clips
        return score

    # ── Helpers (each one is a placeholder / TODO that EC2 work fills in) ──

    def _broadcast_to_tr_axis(self, per_video_assembly, n_tr: int):
        """Broadcast a (presentation, 1, neuroid) frame-aggregated assembly
        to (presentation, n_tr, neuroid) by duplicating along the new time axis.
        Image models have no temporal information to add."""
        raise NotImplementedError(
            "TODO: broadcast per_video_assembly's single time_bin across n_tr time_bins. "
            "Use np.broadcast_to + reconstruct DataArray with new time_bin coord."
        )

    def _hrf_convolve_features(self, features_assembly, sample_rate_hz: float):
        """Convolve features along the time_bin axis with the canonical HRF.

        Wraps `core/brainscore_core/temporal.py::hrf_convolve` which expects
        a 2-D (n_time, n_features) numpy input. We'll need to reshape per-clip
        slices, convolve, and reassemble.
        """
        raise NotImplementedError(
            "TODO: per clip, slice features along time_bin → 2-D (n_time, n_neuroid) → "
            "hrf_convolve → reassemble. Or: vectorize across clips by treating "
            "(presentation × neuroid) as the feature axis and convolving along time."
        )

    def _resample_to_target_grid(self, model_features, target_time_bins, pipeline_mode: str):
        """Resample model time axis to TR grid.

        Uses `temporal_bin` if model_features carries `frame_time_ms` per timestamp.
        For video-native models with internal step indices, we first attach
        absolute timestamps using the start_recording time_bins as the target,
        and the model's known native rate to back-compute step → ms.
        """
        raise NotImplementedError(
            "TODO: handle (a) the video-native case where features come with "
            "step-indexed time_bin (need to attach absolute timestamps), and "
            "(b) the frame-aggregation case where features are already "
            "broadcast to TR grid (no resample needed)."
        )

    def _cross_validated_pearson(self, model, neural, block_size_clips: int):
        """Per-(voxel, TR) ridge with contiguous-block CV over the clip axis.

        Uses `core/brainscore_core/temporal.py::contiguous_block_cv` for splits.
        Returns (n_voxels, n_tr) of Pearson r values.
        """
        from sklearn.linear_model import Ridge

        # Align by stimulus_id
        # model shape:  (n_clips, n_tr_model, n_neuroid_model)
        # neural shape: (n_clips, n_tr_neural, n_voxels)
        # n_tr_model and n_tr_neural should match after _resample_to_target_grid

        raise NotImplementedError(
            "TODO: for each TR index t in 0..n_tr-1:\n"
            "  X = model[:, t, :]  →  (n_clips, n_features)\n"
            "  y = neural[:, t, :] →  (n_clips, n_voxels)\n"
            "  for train_idx, test_idx in contiguous_block_cv(n_clips, block_size_clips):\n"
            "    fit Ridge on train, predict test → fold_preds\n"
            "  per-voxel Pearson(y_true, y_fold_preds) at this TR\n"
            "Stack across TRs → (n_voxels, n_tr) result."
        )

    def _pick_peak_hrf_tr_idx(self, time_bins, peak_sec: float = 5.0) -> int:
        """Find the TR whose center is closest to peak_sec post-stimulus onset."""
        starts_ms = np.array([s for s, _ in time_bins])
        ends_ms = np.array([e for _, e in time_bins])
        centers_sec = ((starts_ms + ends_ms) / 2.0) / 1000.0
        return int(np.argmin(np.abs(centers_sec - peak_sec)))


def Lahner2024BOLDMoments_timeresolved_visualROI():
    """Visual-ROI variant of the TR-resolved benchmark."""
    return Lahner2024BOLDMoments_timeresolved(
        reliability_threshold=0.3,
        identifier_suffix='-timeresolved-visualROI',
    )
