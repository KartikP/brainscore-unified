"""
Lahner 2024 BOLDMoments — TR-resolved fMRI variant (continuous-time encoding).

Same stimuli as the existing GLM-beta variant (`Lahner2024BOLDMoments`), but
predicts the per-TR BOLD time-series of each scanning run rather than a
per-clip GLM beta. M12-lite — the first benchmark that actually exercises
the temporal kit (`temporal_bin`, `hrf_convolve`, `contiguous_block_cv`,
absolute timestamps) end-to-end.

## Design: continuous-time encoding

For each (subject, run), the brain produces a continuous BOLD time-series at
TR=1.75 s across the whole run (~263 TRs covering ~113 stimulus events at
4 s SOA). The model produces per-stimulus features. We HRF-convolve and
place the per-stimulus features at the right TR offsets within run-time,
producing a continuous "expected response" time-series at TR resolution.
Per-voxel ridge regression maps that to the actual BOLD signal, with
leave-one-run-out cross-validation.

This avoids the rapid-event-related design's overlap problem (per-trial
windowing with K>2 TRs has neighboring-trial contamination at SOA=4 s
relative to TR=1.75 s). Continuous-time encoding is the standard
naturalistic-fMRI approach (Huth lab, Friends/Sherlock benchmarks).

## Pipeline (per model)

    1. Discover unique stimuli from events (1102 unique videos).
    2. Extract per-stimulus features ONCE via model.process(unique_stim_set):
       - video-native models: per-clip features at the model's native time bins
         (mean-pooled if per-frame, kept temporal otherwise)
       - frame-aggregation models: per-clip features (mean of N still frames)
    3. For each (subject, run):
       a. Build a feature time-series at TR resolution by placing each clip's
          features at floor(onset_sec / TR) within the run.
       b. HRF-convolve along the time axis using the canonical SPM double-gamma.
       c. Crop / pad to the run's actual TR length.
    4. Concatenate (subject, run) feature time-series into the design matrix X.
       Concatenate (subject, run) BOLD time-series into Y.
    5. Per-voxel ridge regression with leave-one-run-out CV via
       contiguous_block_cv at the run-block level.
    6. Per-voxel Pearson on held-out predictions, median across voxels.

## Comparison story

Same 1026+ videos, same models, but model has to predict TR-resolved
trajectories rather than per-clip betas. Score delta between this variant
and `Lahner2024-fMRI-naturalistic` quantifies how much temporal information
the model adds (or how much the GLM beta already captured).
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
)

# Reuse stimulus loader + bibtex + S3 paths from the existing variant
from .benchmark import (
    BIBTEX,
    STIMULUS_BUCKET,
    Lahner2024BOLDMoments,        # parent class — provides _videos_stimulus_set
    load_stimulus_set,
)


# ── S3 metadata for the TR-resolved artifacts ─────────────────────────
# Filled in once `prepare_timeresolved_assembly.py` uploads to S3.
# Until then, loading raises with an explicit handoff message.

TIMERESOLVED_ASSEMBLY_ID = 'Lahner2024-fMRI-timeresolved'
TIMERESOLVED_ASSEMBLY_VERSION_ID: Optional[str] = '6VFZOroMxaNg1P_xSQAYwGq7.Fc63zz_'
TIMERESOLVED_ASSEMBLY_SHA1: Optional[str]       = '4d06589dde6dddf273a489a64da1337940d5fafd'
TIMERESOLVED_EVENTS_VERSION_ID: Optional[str]   = 'MYar7u8KE_D.i4MXZ83GurH1EDECZ.jo'
TIMERESOLVED_EVENTS_SHA1: Optional[str]         = '783c5b33a75f121812a49b8b0a7639b9ed07c6a3'

# Confirmed scanner / paradigm parameters (from EC2 recon, ds005165 v1.0.4)
TR_SEC = 1.75
SOA_SEC = 4.0

# Cap on per-stimulus feature dim before ridge — guards a 64GB OOM on g5.4xlarge
# when n_TR_obs ≈ 127k and flattened ViT features ≈ 38k. See `_extract_per_stimulus_features`.
FEATURE_DIM_CAP = 512
CLIP_DURATION_SEC = 3.0


# ── Loader ────────────────────────────────────────────────────────────

def load_timeresolved_assembly(merge_stimulus_set_meta: bool = False) -> NeuronRecordingAssembly:
    if TIMERESOLVED_ASSEMBLY_VERSION_ID is None or TIMERESOLVED_ASSEMBLY_SHA1 is None:
        raise RuntimeError(
            "Lahner2024 TR-resolved assembly is not yet hosted on S3. "
            "Run `prepare_timeresolved_assembly.py` on EC2 to download from "
            "OpenNeuro ds005165 v1.0.4, downsample fsaverage→fsaverage5, build "
            "the per-(subject, run) assembly, and upload — then paste the "
            "resulting (version_id, sha1) tuples into benchmark_timeresolved.py."
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


def load_timeresolved_events():
    """Load the per-trial events sidecar CSV (long format).

    Columns: subject, session, run, task, trial_idx, stimulus_id,
             onset_sec, duration_sec, trial_type.
    """
    import io
    import boto3
    import pandas as pd

    if TIMERESOLVED_EVENTS_VERSION_ID is None or TIMERESOLVED_EVENTS_SHA1 is None:
        raise RuntimeError(
            "Lahner2024 TR-resolved events sidecar is not yet hosted on S3. "
            "Same handoff as load_timeresolved_assembly."
        )
    parts = STIMULUS_BUCKET.split('/', 1)
    bucket = parts[0]
    key = (f'{parts[1]}/Lahner2024-fMRI-timeresolved-events.csv'
           if len(parts) > 1 else 'Lahner2024-fMRI-timeresolved-events.csv')
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key, VersionId=TIMERESOLVED_EVENTS_VERSION_ID)
    body = obj['Body'].read()
    return pd.read_csv(io.BytesIO(body))


# ── Benchmark class ───────────────────────────────────────────────────

class Lahner2024BOLDMoments_timeresolved(BenchmarkBase):
    """Predict per-TR BOLD time-series for each BoldMoments scanning run.

    Inherits stimulus-handling helpers from `Lahner2024BOLDMoments` (the
    GLM-beta variant) — stimuli are identical between the two variants.
    Only the neural target and the regression machinery differ.
    """

    def __init__(
        self,
        ceiling: Optional[float] = None,
        cv_n_held_out_runs: int = 1,
        identifier_suffix: str = '-timeresolved',
    ):
        self._cv_n_held_out_runs = cv_n_held_out_runs
        self._assembly: Optional[NeuronRecordingAssembly] = None
        self._events = None
        self._stimulus_set = None

        # Delegate stimulus prep to the GLM-beta variant — identical stimuli.
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

    @property
    def events(self):
        if self._events is None:
            self._events = load_timeresolved_events()
        return self._events

    @property
    def stimulus_set(self):
        return self._stim_helper.stimulus_set

    def _sanity_check_assembly(self, assembly):
        # MultiIndex level coords are exposed via .indexes[dim].names in xarray 2022.3,
        # not via .coords (top-level coords are just the dim coords themselves).
        time_bin_levels = list(assembly.indexes['time_bin'].names)
        presentation_levels = list(assembly.indexes['presentation'].names)
        assert 'time_bin_start_ms' in time_bin_levels, time_bin_levels
        assert 'subject' in presentation_levels, presentation_levels
        assert 'run' in presentation_levels, presentation_levels
        assert 'n_valid_TR' in presentation_levels, presentation_levels
        assert assembly.sizes['neuroid'] == 20484
        assert assembly.sizes['time_bin'] > 1, \
            "TR-resolved assembly must have time_bin > 1; got the GLM-beta variant by mistake?"

    def __call__(self, candidate) -> Score:
        # 1. Get per-stimulus model features (one feature vector per unique stimulus).
        per_stim_features, stimulus_ids = self._extract_per_stimulus_features(candidate)
        # per_stim_features shape: (n_stimuli, n_features)
        # stimulus_ids: list of length n_stimuli

        stim_idx = {sid: i for i, sid in enumerate(stimulus_ids)}

        # 2. For each (subject, run) presentation: build feature time-series at TR
        #    resolution by placing per-stim features at onset TRs, then HRF-convolve.
        assembly = self.assembly
        n_TR_max = assembly.sizes['time_bin']
        n_features = per_stim_features.shape[1]
        n_runs = assembly.sizes['presentation']

        feature_ts = np.zeros((n_runs, n_TR_max, n_features), dtype=np.float32)
        for run_idx in range(n_runs):
            run_id = (str(assembly['subject'].values[run_idx]),
                      str(assembly['session'].values[run_idx]),
                      str(assembly['run'].values[run_idx]))
            run_events = self._events_for_run(*run_id)
            for _, ev in run_events.iterrows():
                if ev['stimulus_id'] not in stim_idx:
                    continue   # oddball or stimulus the model couldn't process
                onset_TR = int(np.floor(ev['onset_sec'] / TR_SEC))
                if 0 <= onset_TR < n_TR_max:
                    feature_ts[run_idx, onset_TR, :] += per_stim_features[stim_idx[ev['stimulus_id']]]

        # HRF-convolve along time axis. hrf_convolve expects (n_time, n_features).
        hrf_kernel = double_gamma_hrf(duration_sec=32.0, sampling_rate_hz=1.0/TR_SEC)
        feature_ts_convolved = np.zeros_like(feature_ts)
        for run_idx in range(n_runs):
            feature_ts_convolved[run_idx] = hrf_convolve(
                feature_ts[run_idx], sampling_rate_hz=1.0/TR_SEC, hrf=hrf_kernel)

        # 3. Concatenate (run, TR) → flat (n_obs, n_features); same for BOLD.
        #    Mask padded entries via n_valid_TR.
        X_flat, Y_flat, run_idx_per_obs = self._concatenate_with_mask(
            feature_ts_convolved, assembly)

        # 4. Per-voxel ridge regression with leave-one-run-out CV.
        per_voxel_r = self._cross_validated_ridge(
            X_flat, Y_flat, run_idx_per_obs,
            n_held_out=self._cv_n_held_out_runs,
        )

        # 5. Aggregate.
        per_voxel_r_finite = per_voxel_r[~np.isnan(per_voxel_r)]
        median_r = float(np.median(per_voxel_r_finite))
        mean_r = float(np.mean(per_voxel_r_finite))

        score = Score(median_r / float(self.ceiling))
        score.attrs['raw'] = Score(median_r)
        score.attrs['mean_r'] = mean_r
        score.attrs['n_voxels_scored'] = int(len(per_voxel_r_finite))
        score.attrs['n_runs'] = n_runs
        score.attrs['n_observations'] = int(len(Y_flat))
        score.attrs['cv_n_held_out_runs'] = self._cv_n_held_out_runs
        score.attrs['pipeline'] = 'continuous_time_encoding'
        return score

    # ── Helpers (each TODO has detailed instructions) ──────────────

    def _extract_per_stimulus_features(self, candidate):
        """One forward pass per unique stimulus → (n_stimuli, n_features).

        Builds a unique-stimulus set from events (≤1102 unique videos), then
        dispatches based on candidate's modality support:
        - video-native: use _videos_stimulus_set, mean-pool over time_bin axis
        - frame-aggregation: use _expand_videos + temporal_bin to one bin
        Returns (features array (n_stimuli, n_features), stimulus_ids list).
        """
        from .benchmark import VIDEO_DURATION_MS
        from brainscore_core.temporal import temporal_bin

        # Unique stimulus_ids in events that ALSO appear in our stimulus_set
        unique_event_ids = set(self.events['stimulus_id'].unique().tolist())
        full_stim = self._stim_helper.stimulus_set
        stim_set_ids = set(full_stim['stimulus_id'].astype(str).tolist())
        unique_ids = sorted(unique_event_ids & stim_set_ids)

        # Activate the model's recording layer. Mirrors the GLM-beta variant:
        # IT-mapped features predict whole-cortex BOLD via per-voxel ridge.
        candidate.start_recording('IT', time_bins=[(0, VIDEO_DURATION_MS)])

        if 'video' in getattr(candidate, 'supported_modalities', set()):
            video_stim = self._stim_helper._videos_stimulus_set()
            # Restrict to the unique_ids we need
            video_stim = video_stim[video_stim['stimulus_id'].isin(unique_ids)]
            result = candidate.process(video_stim)
            data = result.values
            if data.ndim == 3:    # (presentation, time_bin, neuroid) → mean over time
                features = data.mean(axis=1)
            else:                  # (presentation, neuroid)
                features = data
            stimulus_ids = list(
                result.indexes['presentation'].get_level_values('stimulus_id'))
        else:
            frame_stim = self._stim_helper._expand_videos()
            frame_stim = frame_stim[frame_stim['clip_id'].isin(unique_ids)]
            per_frame = candidate.process(frame_stim)
            per_video = temporal_bin(per_frame, time_bins=[(0, VIDEO_DURATION_MS)])
            features = per_video.values[:, 0, :]
            stimulus_ids = list(
                per_video.indexes['presentation'].get_level_values('clip_id'))

        # Cap feature dim before per-voxel ridge — flattened ViT layer outputs
        # (e.g. CLIP encoder.layers.10 = 50 tokens × 768 = 38400) make X scale
        # to (127k TR-obs × 38k feat × 4B) ≈ 19 GB and OOM the 64GB box.
        # TruncatedSVD to 512 keeps almost all variance and shrinks X 75×.
        if features.shape[1] > FEATURE_DIM_CAP:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=FEATURE_DIM_CAP, random_state=0)
            features = svd.fit_transform(features).astype(np.float32)

        return features, stimulus_ids

    def _events_for_run(self, subject, session, run):
        """Filter the events DataFrame to one (subject, session, run)."""
        ev = self.events
        mask = ((ev['subject'] == subject) &
                (ev['session'] == session) &
                (ev['run'] == run))
        return ev[mask]

    def _concatenate_with_mask(self, feature_ts_convolved, assembly):
        """Flatten (run, TR, ...) into long (n_obs, ...) tables, masking padding.

        For each presentation idx, take only the first n_valid_TR rows;
        stack across runs.
        """
        n_valid = assembly['n_valid_TR'].values.astype(int)   # (n_runs,)
        # assembly dims: (time_bin, neuroid, presentation). Reorder to
        # (presentation, time_bin, neuroid) for slicing.
        bold = assembly.transpose('presentation', 'time_bin', 'neuroid').values

        X_chunks, Y_chunks, run_chunks = [], [], []
        for run_idx, n_TR in enumerate(n_valid):
            X_chunks.append(feature_ts_convolved[run_idx, :n_TR, :])
            Y_chunks.append(bold[run_idx, :n_TR, :])
            run_chunks.append(np.full(n_TR, run_idx, dtype=np.int32))
        X_flat = np.concatenate(X_chunks, axis=0).astype(np.float32)
        Y_flat = np.concatenate(Y_chunks, axis=0).astype(np.float32)
        run_idx_per_obs = np.concatenate(run_chunks, axis=0)
        return X_flat, Y_flat, run_idx_per_obs

    def _cross_validated_ridge(self, X_flat, Y_flat, run_idx_per_obs, n_held_out: int):
        """Leave-K-runs-out ridge per voxel; return (n_voxels,) Pearson array."""
        from sklearn.linear_model import Ridge

        n_runs = int(run_idx_per_obs.max() + 1)
        n_obs, n_voxels = Y_flat.shape
        held_out_preds = np.full_like(Y_flat, np.nan)

        for train_runs, test_runs in contiguous_block_cv(
                n_samples=n_runs, block_size_samples=n_held_out):
            train_mask = np.isin(run_idx_per_obs, train_runs)
            test_mask  = np.isin(run_idx_per_obs, test_runs)
            reg = Ridge(alpha=1.0).fit(X_flat[train_mask], Y_flat[train_mask])
            held_out_preds[test_mask] = reg.predict(X_flat[test_mask])

        # Per-voxel Pearson: ignore any rows that didn't get a prediction
        # (shouldn't happen with full coverage, but guard).
        per_voxel_r = np.full(n_voxels, np.nan, dtype=np.float32)
        valid = ~np.isnan(held_out_preds[:, 0])
        if not valid.any():
            return per_voxel_r
        Yt = Y_flat[valid]
        Yp = held_out_preds[valid]
        Yt_c = Yt - Yt.mean(axis=0, keepdims=True)
        Yp_c = Yp - Yp.mean(axis=0, keepdims=True)
        num = (Yt_c * Yp_c).sum(axis=0)
        den = np.sqrt((Yt_c ** 2).sum(axis=0) * (Yp_c ** 2).sum(axis=0))
        with np.errstate(divide='ignore', invalid='ignore'):
            per_voxel_r = np.where(den > 0, num / den, np.nan).astype(np.float32)
        return per_voxel_r
