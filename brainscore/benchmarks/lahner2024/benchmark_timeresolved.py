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
from brainscore_core.temporal import (
    contiguous_block_cv,
    double_gamma_hrf,
    hrf_convolve,
)
from brainscore import load_dataset
from brainscore.benchmarks._scoring_utils import (
    masked_ridge_predictions,
    valid_prediction_per_unit_pearson,
)
from brainscore.data.lahner2024 import MOTION_COLUMNS

# Reuse stimulus handling + bibtex from the GLM-beta variant.
from .benchmark import (
    BIBTEX,
    Lahner2024BOLDMoments,        # parent class — provides _videos_stimulus_set
)


# Confirmed scanner / paradigm parameters (from EC2 recon, ds005165 v1.0.4)
TR_SEC = 1.75
SOA_SEC = 4.0

# Cap on per-stimulus feature dim before ridge — guards a 64GB OOM on g5.4xlarge
# when n_TR_obs ≈ 127k and flattened ViT features ≈ 38k. See `_extract_per_stimulus_features`.
FEATURE_DIM_CAP = 512
CLIP_DURATION_SEC = 3.0


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
        cv_n_held_out_runs: int = 52,
        reliability_threshold: Optional[float] = None,
        identifier_suffix: str = '-timeresolved',
        apply_motion_regression: bool = False,
        cv_mode: str = 'subject_out',           # 'subject_out' | 'within_subject'
        within_subject_n_held_out: int = 10,    # block size when cv_mode='within_subject'
    ):
        if cv_mode not in ('subject_out', 'within_subject'):
            raise ValueError(f"cv_mode must be 'subject_out' or 'within_subject', got {cv_mode!r}")
        self._cv_n_held_out_runs = cv_n_held_out_runs
        self._reliability_threshold = reliability_threshold
        self._apply_motion_regression = apply_motion_regression
        self._cv_mode = cv_mode
        self._within_subject_n_held_out = within_subject_n_held_out
        self._voxel_mask: Optional[np.ndarray] = None
        self._assembly: Optional[NeuronRecordingAssembly] = None
        self._events = None
        self._motion = None
        self._stimulus_set = None

        # Delegate stimulus prep to the GLM-beta variant — identical stimuli.
        # We also reuse its reliability machinery for visualROI variants
        # (TR-resolved has no per-stim reps to compute split-half on).
        self._stim_helper = Lahner2024BOLDMoments(
            reliability_threshold=reliability_threshold)

        super().__init__(
            identifier=f'Lahner2024-fMRI-naturalistic{identifier_suffix}',
            version=1,
            parent='naturalistic',
            ceiling=Score(1.0 if ceiling is None else float(ceiling)),
            bibtex=BIBTEX,
        )
        # Accepts native-video AND still-image models (frame-aggregation).
        self.accepted_modalities = {'video', 'vision'}

    @property
    def assembly(self) -> NeuronRecordingAssembly:
        if self._assembly is None:
            self._assembly = load_dataset('Lahner2024-fMRI-timeresolved')
            self._sanity_check_assembly(self._assembly)
        return self._assembly

    @property
    def events(self):
        if self._events is None:
            self._events = load_dataset(
                'Lahner2024-fMRI-timeresolved-events')
        return self._events

    @property
    def motion(self):
        """Lazy-load motion sidecar; returns None if motion regression is off."""
        if not self._apply_motion_regression:
            return None
        if self._motion is None:
            self._motion = load_dataset(
                'Lahner2024-fMRI-timeresolved-motion')
        return self._motion

    @property
    def stimulus_set(self):
        return self._stim_helper.stimulus_set

    @property
    def voxel_mask(self) -> Optional[np.ndarray]:
        """Lazy boolean mask over the 20484 fsaverage5 voxels.

        Computed via the GLM-beta variant's split-half reliability (which has
        per-stim reps); the same mask is meaningful here because both variants
        share the identical fsaverage5 voxel space.
        """
        if self._reliability_threshold is None:
            return None
        if self._voxel_mask is None:
            self._voxel_mask = self._stim_helper._get_voxel_mask()
        return self._voxel_mask

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
        #    Optionally regress motion confounds out of BOLD per-run before
        #    z-score. Track subject per-obs so within_subject CV can split
        #    runs within each subject independently.
        X_flat, Y_flat, run_idx_per_obs, subject_per_obs = self._concatenate_with_mask(
            feature_ts_convolved, assembly)

        # 4. Per-voxel ridge regression. Two CV modes:
        #    - 'subject_out' (default): leave-1-subject-out global CV (10 folds).
        #    - 'within_subject': per-subject leave-K-runs-out, average per-voxel
        #      r across subjects (closer to standard encoding-lit practice).
        if self._cv_mode == 'within_subject':
            per_voxel_r = self._cross_validated_ridge_within_subject(
                X_flat, Y_flat, run_idx_per_obs, subject_per_obs,
                n_held_out_per_subject=self._within_subject_n_held_out,
            )
        else:
            per_voxel_r = self._cross_validated_ridge(
                X_flat, Y_flat, run_idx_per_obs,
                n_held_out=self._cv_n_held_out_runs,
            )

        # 5. Aggregate. Apply voxel mask first if visualROI variant.
        mask = self.voxel_mask
        if mask is not None:
            per_voxel_r = per_voxel_r[mask]
        per_voxel_r_finite = per_voxel_r[~np.isnan(per_voxel_r)]
        median_r = float(np.median(per_voxel_r_finite))
        mean_r = float(np.mean(per_voxel_r_finite))

        score = Score(median_r / float(self.ceiling))
        score.attrs['raw'] = Score(median_r)
        score.attrs['ceiling'] = self.ceiling   # uniform score-attr contract
        score.attrs['mean_r'] = mean_r
        score.attrs['n_voxels_scored'] = int(len(per_voxel_r_finite))
        score.attrs['n_runs'] = n_runs
        score.attrs['n_observations'] = int(len(Y_flat))
        score.attrs['cv_mode'] = self._cv_mode
        score.attrs['cv_n_held_out_runs'] = (self._within_subject_n_held_out
                                              if self._cv_mode == 'within_subject'
                                              else self._cv_n_held_out_runs)
        score.attrs['motion_regressed'] = bool(self._apply_motion_regression)
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

        # route on raw input_modalities: channel unification canonicalizes
        # video->vision in supported_modalities, so a native-video model reports
        # 'vision' there and would misroute to frame-aggregation.
        if 'video' in getattr(candidate, 'input_modalities', set()):
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

        Pipeline per run:
          1. Slice to n_valid_TR (drop padding).
          2. (Optional) regress motion confounds out of BOLD per voxel — OLS
             of BOLD on [trans_x..rot_z, FD, csf, white_matter] + intercept,
             keep residuals. Standard fMRI denoising step (Power 2014, Ciric
             2017). Adds noise-cleaning before z-score.
          3. Per-voxel z-score the residual BOLD. Removes subject-specific DC
             that would otherwise leak into the ridge intercept across folds.

        Returns:
            X_flat: (n_obs, n_features)
            Y_flat: (n_obs, n_voxels)
            run_idx_per_obs: (n_obs,) int — index into assembly's presentation
                axis. Two presentations may differ in subject; same run_idx ⇒
                same run.
            subject_per_obs: (n_obs,) object — subject string per observation.
                Threaded through so within-subject CV can split runs WITHIN
                each subject independently.
        """
        n_valid = assembly['n_valid_TR'].values.astype(int)
        bold = assembly.transpose('presentation', 'time_bin', 'neuroid').values
        subjects_per_run = assembly['subject'].values
        sessions_per_run = assembly['session'].values
        runs_per_run     = assembly['run'].values
        tasks_per_run    = assembly['task'].values

        # Pre-index motion sidecar by (subject, session, task, run) for O(1) lookup.
        motion_df = self.motion
        motion_lookup = None
        if motion_df is not None:
            motion_lookup = {
                (s, ses, t, r): g[MOTION_COLUMNS].values.astype(np.float32)
                for (s, ses, t, r), g in motion_df.groupby(
                    ['subject', 'session', 'task', 'run'], sort=False)
            }

        X_chunks, Y_chunks, run_chunks, subj_chunks = [], [], [], []
        for run_idx, n_TR in enumerate(n_valid):
            X_chunks.append(feature_ts_convolved[run_idx, :n_TR, :])
            Y_run = bold[run_idx, :n_TR, :].astype(np.float32)

            if motion_lookup is not None:
                key = (str(subjects_per_run[run_idx]),
                       str(sessions_per_run[run_idx]),
                       str(tasks_per_run[run_idx]),
                       str(runs_per_run[run_idx]))
                M = motion_lookup.get(key)
                if M is not None and len(M) >= n_TR:
                    M = M[:n_TR]
                    # OLS with intercept: residuals of Y on [1, M].
                    M_design = np.concatenate([np.ones((n_TR, 1), dtype=np.float32), M],
                                              axis=1)
                    # solve M_design @ B = Y → B (n_motion+1, n_voxels)
                    B, *_ = np.linalg.lstsq(M_design, np.nan_to_num(Y_run), rcond=None)
                    Y_run = Y_run - M_design @ B

            mu = np.nanmean(Y_run, axis=0, keepdims=True)
            sd = np.nanstd(Y_run, axis=0, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                Y_run = np.where(sd > 0, (Y_run - mu) / sd, 0.0)
            Y_chunks.append(Y_run)
            run_chunks.append(np.full(n_TR, run_idx, dtype=np.int32))
            subj_chunks.append(np.full(n_TR, str(subjects_per_run[run_idx]), dtype=object))

        X_flat = np.concatenate(X_chunks, axis=0).astype(np.float32)
        Y_flat = np.concatenate(Y_chunks, axis=0).astype(np.float32)
        run_idx_per_obs = np.concatenate(run_chunks, axis=0)
        subject_per_obs = np.concatenate(subj_chunks, axis=0)
        return X_flat, Y_flat, run_idx_per_obs, subject_per_obs

    def _cross_validated_ridge(self, X_flat, Y_flat, run_idx_per_obs, n_held_out: int):
        """Leave-K-runs-out ridge per voxel; return (n_voxels,) Pearson array.

        Voxel-batched: holding the full held_out_preds (n_obs × 20484 × 4B = 10 GB)
        plus sklearn's internal Y copies during fit blew past 64GB. We batch the
        voxel axis so peak memory is bounded by VOXEL_BATCH × n_obs × 4B (~1 GB).
        """
        n_runs = int(run_idx_per_obs.max() + 1)
        n_obs, n_voxels = Y_flat.shape
        per_voxel_r = np.full(n_voxels, np.nan, dtype=np.float32)

        # Precompute fold masks once.
        fold_masks = [
            (np.isin(run_idx_per_obs, train_runs), np.isin(run_idx_per_obs, test_runs))
            for train_runs, test_runs in contiguous_block_cv(
                n_samples=n_runs, block_size_samples=n_held_out)
        ]

        VOXEL_BATCH = 2000
        for v_start in range(0, n_voxels, VOXEL_BATCH):
            v_end = min(v_start + VOXEL_BATCH, n_voxels)
            Y_sub = Y_flat[:, v_start:v_end]
            held_out_preds = masked_ridge_predictions(
                X_flat, Y_sub, fold_masks, alpha=1.0, dtype=None)
            per_voxel_r[v_start:v_end] = valid_prediction_per_unit_pearson(
                Y_sub, held_out_preds)
        return per_voxel_r

    def _cross_validated_ridge_within_subject(self, X_flat, Y_flat,
                                              run_idx_per_obs, subject_per_obs,
                                              n_held_out_per_subject: int):
        """Per-subject leave-K-runs-out ridge; average per-voxel r across subjects.

        For each subject independently: do leave-K-runs-out CV across that
        subject's runs only (block_size_samples=n_held_out_per_subject), fit
        ridge on subject-internal train, predict subject-internal held-out.
        Compute per-voxel pearson r on held-out predictions stitched across
        the subject's CV folds. Final per-voxel r = mean across subjects
        (NaN-aware).

        This matches standard fMRI encoding practice (Conwell, Allen, NSD)
        where train/test splits are within-subject, and avoids the cross-
        subject mean-leakage that plagues subject_out CV.
        """
        n_obs, n_voxels = Y_flat.shape
        unique_subjects = np.unique(subject_per_obs)
        per_subject_r = np.full((len(unique_subjects), n_voxels), np.nan,
                                dtype=np.float32)

        VOXEL_BATCH = 2000
        for s_idx, subj in enumerate(unique_subjects):
            sub_obs = (subject_per_obs == subj)
            X_sub = X_flat[sub_obs]
            Y_sub_full = Y_flat[sub_obs]
            run_sub = run_idx_per_obs[sub_obs]
            sub_unique_runs = np.unique(run_sub)
            n_sub_runs = len(sub_unique_runs)

            fold_masks = []
            for train_local, test_local in contiguous_block_cv(
                    n_samples=n_sub_runs, block_size_samples=n_held_out_per_subject):
                train_runs = sub_unique_runs[train_local]
                test_runs  = sub_unique_runs[test_local]
                fold_masks.append((np.isin(run_sub, train_runs),
                                   np.isin(run_sub, test_runs)))

            for v_start in range(0, n_voxels, VOXEL_BATCH):
                v_end = min(v_start + VOXEL_BATCH, n_voxels)
                Y_sub_batch = Y_sub_full[:, v_start:v_end]
                held_out_preds = masked_ridge_predictions(
                    X_sub, Y_sub_batch, fold_masks, alpha=1.0, dtype=None)
                per_subject_r[s_idx, v_start:v_end] = (
                    valid_prediction_per_unit_pearson(
                        Y_sub_batch, held_out_preds))

        # Mean per-voxel r across subjects, ignoring NaNs.
        return np.nanmean(per_subject_r, axis=0).astype(np.float32)


def Lahner2024BOLDMoments_timeresolved_visualROI():
    """ROI variant: only voxels with split-half reliability >= 0.3.

    Mirrors the GLM-beta `-visualROI` variant. Reliability is computed on the
    GLM-beta assembly (which has 10 reps/stim) and applied to the TR-resolved
    encoding result; both share the identical fsaverage5 voxel space.
    """
    return Lahner2024BOLDMoments_timeresolved(
        reliability_threshold=0.3,
        identifier_suffix='-timeresolved-visualROI',
    )


def Lahner2024BOLDMoments_timeresolved_improved():
    """`-improved` variant: motion-regression + within-subject CV.

    Two upgrades over the baseline TR-resolved variant, both standard fMRI
    encoding practice:

    1. **Motion regression.** Per-run OLS regression of BOLD on 9 fmriprep
       confound regressors (6 motion params + framewise displacement +
       csf/white-matter signal); residuals replace raw BOLD before z-score.
       Removes scanner/physiology variance that confounds stimulus-driven
       signal. (Power 2014, Ciric 2017.)
    2. **Within-subject CV.** Leave-K-runs-out CV inside each subject
       independently; per-voxel r averaged across subjects. Avoids the
       cross-subject mean-leakage that hurt the baseline `subject_out` CV
       and matches NSD / Conwell / Allen encoding pipelines.

    Together expected to lift TR-resolved-ROI raw r from ~0.066 (baseline)
    toward ~0.13–0.18 (literature norm for single-trial encoding without
    GLMsingle-style denoising).
    """
    return Lahner2024BOLDMoments_timeresolved(
        identifier_suffix='-timeresolved-improved',
        apply_motion_regression=True,
        cv_mode='within_subject',
        within_subject_n_held_out=10,
    )


def Lahner2024BOLDMoments_timeresolved_improved_visualROI():
    """`-improved-visualROI`: motion-regression + within-subject CV + ROI mask."""
    return Lahner2024BOLDMoments_timeresolved(
        reliability_threshold=0.3,
        identifier_suffix='-timeresolved-improved-visualROI',
        apply_motion_regression=True,
        cv_mode='within_subject',
        within_subject_n_held_out=10,
    )
