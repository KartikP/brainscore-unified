"""
Lahner 2024 BOLDMoments — TR-resolved fMRI variant (continuous-time encoding).

Same stimuli as the existing GLM-beta variant (`Lahner2024BOLDMoments`), but
predicts the per-TR BOLD time-series of each scanning run rather than a
per-clip GLM beta. M12-lite — the first benchmark that exercises the
temporal kit (`hrf_convolve`, `contiguous_block_cv`) end-to-end.

## Design: clip-mean onset-impulse HRF model

For each (subject, session, task, run), the brain produces a continuous BOLD
time-series at TR=1.75 s across the whole run (~263 TRs covering ~113
stimulus events at 4 s SOA). The model produces ONE feature vector per clip:
any native time bins (video tubelets, audio frames) are averaged before the
design is built, so within-clip dynamics do not enter the regression. Each
clip vector is added as a single impulse at ``floor(onset_sec / TR)`` (the
sub-TR part of the onset is dropped, and the event duration is not used —
no boxcar), the impulse train is convolved with the canonical SPM
double-gamma HRF separately within each run, and per-voxel ridge maps the
result to the run's BOLD.

This is a voxelwise encoding model, but a temporal approximation: it is not
a model of the stimulus's native time course. A full temporal protocol
(native feature timestamps, exact onsets, durations, scan timing) would be a
separate benchmark.

## Pipeline (per model)

    1. Discover unique stimuli from events (1102 unique videos).
    2. Record per-stimulus features ONCE through participate (neural_response):
       - video-native models: per-clip features at the model's native time
         bins, always mean-pooled over the time_bin axis to one vector
       - frame-aggregation models: per-clip features (mean of N still frames)
       If the feature width exceeds FEATURE_DIM_CAP, a TruncatedSVD is fit on
       ALL stimuli (including those in held-out runs) — a transductive
       projection, recorded on the score.
    3. For each (subject, session, task, run):
       a. Place each clip's vector at floor(onset_sec / TR) within the run.
       b. HRF-convolve along the time axis (canonical SPM double-gamma).
       c. Crop to the run's valid TR length.
    4. Concatenate the runs' feature time-series into the design matrix X and
       the runs' z-scored BOLD into Y.
    5. Per-voxel ridge (alpha=1.0) with whole-run held-out CV:
       - cv_mode='subject_out' (default): contiguous blocks of
         ``cv_n_held_out_runs`` (default 52) runs in assembly order; this
         holds out one participant per fold only if the assembly lists each
         participant's runs together (checked and recorded on the score).
       - cv_mode='within_subject': per participant, contiguous blocks of
         ``within_subject_n_held_out`` runs; per-voxel r averaged over
         participants.
       Every held-out stimulus is also shown in training runs, so the score
       measures generalization to new runs of seen stimuli, not to new
       stimuli.
    6. Per-voxel Pearson on held-out predictions, median across voxels.

## Versions

- version 2 (current): events are selected by (subject, session, task, run).
- version 1: events were selected by (subject, session, run) only. The two
  BOLDMoments tasks ('train', 'test') reuse these keys, so 240 of the 520
  runs also received the other task's events (defect F6). Kept reachable
  with ``version=1`` (e.g. ``load_benchmark(identifier, version=1)``) so
  historical scores can be reproduced; its scores carry
  ``attrs['known_defects']``.

## Comparison with the GLM-beta variant

Same 1026+ videos and models, but the target (single-run BOLD time series
vs. trial-averaged GLM betas), the CV protocol, the voxel noise level and
the feature design all differ. A score difference between this variant and
`Lahner2024-fMRI-naturalistic` therefore does not isolate the contribution
of temporal information.
"""

from typing import List, Optional

import numpy as np
import xarray as xr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuronRecordingAssembly,
)
from brainscore.temporal import (
    contiguous_block_cv,
    double_gamma_hrf,
    hrf_convolve,
)
from brainscore import load_dataset
from brainscore.metrics.encoding import SklearnRidge, cross_validated_r, ridge
from brainscore.data.lahner2024 import MOTION_COLUMNS

# Reuse stimulus handling + bibtex from the GLM-beta variant.
from .benchmark import (
    BIBTEX,
    Lahner2024BOLDMoments,        # parent class — provides _videos_stimulus_set
)
from ._util import read_stimulus_ids, record_clips, record_frames_per_clip


# Confirmed scanner / paradigm parameters (from EC2 recon, ds005165 v1.0.4)
TR_SEC = 1.75
SOA_SEC = 4.0

# Voxels fitted together. It bounds memory (held-out predictions for all 20484
# voxels are about 10 GB). A banded penalty is chosen per batch (finding C2).
VOXEL_BATCH = 2000

# Cap on per-stimulus feature dim before ridge — guards a 64GB OOM on g5.4xlarge
# when n_TR_obs ≈ 127k and flattened ViT features ≈ 38k. See `_extract_per_stimulus_features`.
FEATURE_DIM_CAP = 512
CLIP_DURATION_SEC = 3.0

# ── Protocol versions ─────────────────────────────────────────────────
# Version 2 selects each run's events by the full run key, task included.
# Version 1 omitted task; it stays reachable so historical scores can be
# reproduced, and its scores are labelled with the defect.
CURRENT_VERSION = 2
EVENT_KEY_BY_VERSION = {
    1: ('subject', 'session', 'run'),
    2: ('subject', 'session', 'task', 'run'),
}
F6_TASK_KEY_DEFECT = (
    "F6: events are selected by (subject, session, run) without task. The "
    "BOLDMoments 'train' and 'test' tasks reuse these keys, so 240 of 520 "
    "runs also receive the other task's events. Fixed in version 2.")
KNOWN_DEFECTS_BY_VERSION = {1: (F6_TASK_KEY_DEFECT,), 2: ()}

# Labels that describe the method the score was computed with (3A). They do
# not change any number; they say what the number means.
METHOD_ATTRS = {
    'encoding_model': 'clip_mean_onset_impulse_hrf',
    'feature_timing': ('one feature vector per clip (native time bins '
                       'averaged), added as an impulse at floor(onset_sec / '
                       'TR); event duration not used'),
    'onset_rounding': 'floor_to_TR',
    'event_duration_used': False,
    'tr_sec': TR_SEC,
    'hrf': 'spm_double_gamma_32s_unit_sum_convolved_per_run',
    'stimulus_generalization': 'seen_stimuli',
}
TRANSDUCTIVE_SVD = 'transductive_truncated_svd_fit_on_all_stimuli'


# ── Benchmark class ───────────────────────────────────────────────────

class Lahner2024BOLDMoments_timeresolved(BenchmarkBase):
    """Predict per-TR BOLD time-series for each BoldMoments scanning run.

    Inherits stimulus-handling helpers from `Lahner2024BOLDMoments` (the
    GLM-beta variant) — stimuli are identical between the two variants.
    Only the neural target and the regression machinery differ.
    """

    # Sends the clips on `vision:video`, or frames sampled from them on `vision:image`, by the
    # subject's declared form, and asks for the region it records. Each clip is an independent trial.
    required_input_channels = frozenset({'vision:image', 'vision:video'})
    requested_output_channels = frozenset({'neural:IT'})
    trial_memory = 'none'

    def __init__(
        self,
        ceiling: Optional[float] = None,
        cv_n_held_out_runs: int = 52,
        reliability_threshold: Optional[float] = None,
        identifier_suffix: str = '-timeresolved',
        apply_motion_regression: bool = False,
        cv_mode: str = 'subject_out',           # 'subject_out' | 'within_subject'
        within_subject_n_held_out: int = 10,    # block size when cv_mode='within_subject'
        version: int = CURRENT_VERSION,
    ):
        if cv_mode not in ('subject_out', 'within_subject'):
            raise ValueError(f"cv_mode must be 'subject_out' or 'within_subject', got {cv_mode!r}")
        if version not in EVENT_KEY_BY_VERSION:
            raise ValueError(
                f"version must be one of {sorted(EVENT_KEY_BY_VERSION)}, got {version!r}")
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
        # Set by feature extraction; records whether a projection was fit.
        self._feature_projection: Optional[str] = None

        # Delegate stimulus prep to the GLM-beta variant — identical stimuli.
        # We also reuse its reliability machinery for visualROI variants
        # (TR-resolved has no per-stim reps to compute split-half on).
        self._stim_helper = Lahner2024BOLDMoments(
            reliability_threshold=reliability_threshold)

        super().__init__(
            identifier=f'Lahner2024-fMRI-naturalistic{identifier_suffix}',
            version=version,
            parent='naturalistic',
            ceiling=Score(1.0 if ceiling is None else float(ceiling)),
            bibtex=BIBTEX,
        )

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
        assert 'task' in presentation_levels, presentation_levels   # event key (v2)
        assert 'n_valid_TR' in presentation_levels, presentation_levels
        assert assembly.sizes['neuroid'] == 20484
        assert assembly.sizes['time_bin'] > 1, \
            "TR-resolved assembly must have time_bin > 1; got the GLM-beta variant by mistake?"

    def __call__(self, subject) -> Score:
        # 1. Get per-stimulus model features (one feature vector per unique stimulus).
        per_stim_features, stimulus_ids = self._extract_per_stimulus_features(subject)
        # per_stim_features shape: (n_stimuli, n_features)
        # stimulus_ids: list of length n_stimuli

        # 2. For each run: place per-stim features at onset TRs, HRF-convolve.
        assembly = self.assembly
        n_runs = assembly.sizes['presentation']
        feature_ts_convolved = self._build_design(
            per_stim_features, stimulus_ids, assembly)

        # 3. Concatenate (run, TR) → flat (n_obs, n_features); same for BOLD.
        #    Optionally regress motion confounds out of BOLD per-run before
        #    z-score. Track the participant per observation so within_subject CV can split
        #    runs within each participant independently.
        X_flat, Y_flat, run_idx_per_obs, participant_per_obs = self._concatenate_with_mask(
            feature_ts_convolved, assembly)

        # 4. Per-voxel ridge regression. Two CV modes:
        #    - 'subject_out' (default): contiguous blocks of cv_n_held_out_runs
        #      (52) runs in assembly order; one participant per block only if the
        #      assembly groups each participant's runs (recorded on the score).
        #    - 'within_subject': per-participant contiguous blocks of K runs,
        #      per-voxel r averaged across participants.
        if self._cv_mode == 'within_subject':
            per_voxel_r = self._cross_validated_ridge_within_participant(
                X_flat, Y_flat, run_idx_per_obs, participant_per_obs,
                n_held_out_per_participant=self._within_subject_n_held_out,
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
        score.attrs.update(self._cv_policy_attrs(assembly))
        score.attrs.update(self._method_attrs())
        score.attrs['ridge_alpha'] = 1.0
        score.attrs['inner_validation'] = 'none_fixed_alpha'
        return score

    # ── Score labels ───────────────────────────────────────────────

    def _method_attrs(self) -> dict:
        """Version, event key, known defects and method description (3A).

        Additive score attrs; they do not change any number.
        """
        attrs = dict(METHOD_ATTRS)
        attrs['benchmark_version'] = int(self.version)
        attrs['event_key'] = EVENT_KEY_BY_VERSION[self.version]
        attrs['known_defects'] = KNOWN_DEFECTS_BY_VERSION[self.version]
        attrs['feature_projection_protocol'] = (
            self._feature_projection or 'not_recorded')
        attrs['feature_dim_cap'] = FEATURE_DIM_CAP
        return attrs

    def _cv_policy_attrs(self, assembly) -> dict:
        """Describe the outer CV the parent ridge actually applies."""
        if self._cv_mode == 'within_subject':
            return {
                'cv_policy': 'within_subject_contiguous_run_blocks',
                'cv_block_runs': int(self._within_subject_n_held_out),
            }
        participants = [str(s) for s in assembly['subject'].values]
        return {
            'cv_policy': 'contiguous_run_blocks_in_assembly_order',
            'cv_block_runs': int(self._cv_n_held_out_runs),
            'cv_blocks_hold_out_whole_subjects': _blocks_hold_out_whole_participants(
                participants, self._cv_n_held_out_runs),
        }

    # ── Design matrix ──────────────────────────────────────────────

    def _build_design(self, per_stim_features, stimulus_ids, assembly) -> np.ndarray:
        """Clip-mean onset-impulse design, HRF-convolved within each run.

        Returns ``(n_runs, n_TR_max, n_features)``. Each event adds its
        clip's feature vector at ``floor(onset_sec / TR)``; the sub-TR part
        of the onset and the event duration are not used.
        """
        stim_idx = {sid: i for i, sid in enumerate(stimulus_ids)}
        n_TR_max = assembly.sizes['time_bin']
        n_features = per_stim_features.shape[1]
        n_runs = assembly.sizes['presentation']

        feature_ts = np.zeros((n_runs, n_TR_max, n_features), dtype=np.float32)
        for run_idx, run_events in enumerate(self._events_per_run(assembly)):
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
        return feature_ts_convolved

    def _events_per_run(self, assembly):
        """Yield each assembly run's events, selected by this version's key."""
        n_runs = assembly.sizes['presentation']
        participants = assembly['subject'].values
        sessions = assembly['session'].values
        runs = assembly['run'].values
        if self.version == 1:
            for run_idx in range(n_runs):
                yield self._events_for_run_without_task(
                    str(participants[run_idx]), str(sessions[run_idx]),
                    str(runs[run_idx]))
            return
        tasks = assembly['task'].values
        groups = self._events_by_run_key()
        for run_idx in range(n_runs):
            yield self._events_for_run(
                str(participants[run_idx]), str(sessions[run_idx]),
                str(tasks[run_idx]), str(runs[run_idx]), groups=groups)

    # ── Helpers ────────────────────────────────────────────────────

    def _extract_per_stimulus_features(self, subject):
        """One forward pass per unique stimulus → (n_stimuli, n_features).

        Builds a unique-stimulus set from events (≤1102 unique videos), then
        dispatches based on subject's modality support:
        - video-native: use _videos_stimulus_set, mean-pool over time_bin axis
        - frame-aggregation: use _expand_videos + temporal_bin to one bin
        Returns (features array (n_stimuli, n_features), stimulus_ids list).
        """
        from .benchmark import VIDEO_DURATION_MS

        # Unique stimulus_ids in events that ALSO appear in our stimulus_set
        unique_event_ids = set(self.events['stimulus_id'].unique().tolist())
        full_stim = self._stim_helper.stimulus_set
        stim_set_ids = set(full_stim['stimulus_id'].astype(str).tolist())
        unique_ids = sorted(unique_event_ids & stim_set_ids)

        # Record neural:IT through participate. Mirrors the GLM-beta variant:
        # IT-mapped features predict whole-cortex BOLD via per-voxel ridge.
        time_bins = [(0, VIDEO_DURATION_MS)]

        # Clips natively when the subject declares vision:video, frames otherwise.
        from .benchmark import Lahner2024BOLDMoments
        if Lahner2024BOLDMoments._is_native_video(subject):
            video_stim = self._stim_helper._videos_stimulus_set()
            # Restrict to the unique_ids we need
            video_stim = video_stim[video_stim['stimulus_id'].isin(unique_ids)]
            result = record_clips(subject, video_stim, 'IT', time_bins)
            data = result.values
            if data.ndim == 3:    # (presentation, time_bin, neuroid) → mean over time
                features = data.mean(axis=1)
            else:                  # (presentation, neuroid)
                features = data
            stimulus_ids = read_stimulus_ids(result)
        else:
            frame_stim = self._stim_helper._expand_videos()
            frame_stim = frame_stim[frame_stim['clip_id'].isin(unique_ids)]
            per_video = record_frames_per_clip(subject, frame_stim, 'IT', time_bins)
            features = per_video.values[:, 0, :]
            stimulus_ids = list(
                per_video.indexes['presentation'].get_level_values('clip_id'))

        # Cap feature dim before per-voxel ridge — flattened ViT layer outputs
        # (e.g. CLIP encoder.layers.10 = 50 tokens × 768 = 38400) make X scale
        # to (127k TR-obs × 38k feat × 4B) ≈ 19 GB and OOM the 64GB box.
        # TruncatedSVD to 512 keeps almost all variance and shrinks X 75×.
        # The SVD is fit on every stimulus, including those only shown in
        # held-out runs (transductive); recorded on the score.
        self._feature_projection = 'none'
        if features.shape[1] > FEATURE_DIM_CAP:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=FEATURE_DIM_CAP, random_state=0)
            features = svd.fit_transform(features).astype(np.float32)
            self._feature_projection = TRANSDUCTIVE_SVD

        return features, stimulus_ids

    def _events_by_run_key(self) -> dict:
        """Group events by (subject, session, task, run), keeping row order."""
        ev = self.events
        key = list(EVENT_KEY_BY_VERSION[2])
        missing = [column for column in key if column not in ev.columns]
        if missing:
            raise ValueError(
                f"Events table lacks {missing}; version {self.version} selects "
                f"events by {tuple(key)}. Use the versioned events sidecar.")
        key_frame = ev[key].astype(str)
        return {
            group_key: ev.iloc[positions]
            for group_key, positions in key_frame.groupby(key, sort=False).indices.items()
        }

    def _events_for_run(self, participant, session, task, run, groups: Optional[dict] = None):
        """Events of one (subject, session, task, run).

        Raises if the run has no events, or if the selection holds more than
        one task (two tasks must never feed one run's design).
        """
        if groups is None:
            groups = self._events_by_run_key()
        key = (str(participant), str(session), str(task), str(run))
        if key not in groups:
            raise ValueError(
                f"No events for run (subject, session, task, run)={key}. "
                f"Every assembly run needs events; check the assembly and the "
                f"events sidecar come from the same build.")
        selected = groups[key]
        tasks = set(selected['task'].astype(str))
        if tasks != {str(task)}:
            raise ValueError(
                f"Events for run {key} hold tasks {sorted(tasks)}; expected "
                f"only {task!r}.")
        return selected

    def _events_for_run_without_task(self, participant, session, run):
        """Version 1 selector (defect F6): (subject, session, run), no task.

        Kept unchanged so version 1 reproduces historical scores. The two
        BOLDMoments tasks reuse these keys, so it can return both tasks'
        events. Do not use for new scores.
        """
        ev = self.events
        mask = ((ev['subject'] == participant) &
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
          3. Per-voxel z-score the residual BOLD. Removes participant-specific DC
             that would otherwise leak into the ridge intercept across folds.

        Returns:
            X_flat: (n_obs, n_features)
            Y_flat: (n_obs, n_voxels)
            run_idx_per_obs: (n_obs,) int — index into assembly's presentation
                axis. Two presentations may differ in participant; same run_idx ⇒
                same run.
            participant_per_obs: (n_obs,) object — participant string per observation.
                Threaded through so within-participant CV can split runs WITHIN
                each participant independently.
        """
        n_valid = assembly['n_valid_TR'].values.astype(int)
        bold = assembly.transpose('presentation', 'time_bin', 'neuroid').values
        participants_per_run = assembly['subject'].values
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

        X_chunks, Y_chunks, run_chunks, participant_chunks = [], [], [], []
        for run_idx, n_TR in enumerate(n_valid):
            X_chunks.append(feature_ts_convolved[run_idx, :n_TR, :])
            Y_run = bold[run_idx, :n_TR, :].astype(np.float32)

            if motion_lookup is not None:
                key = (str(participants_per_run[run_idx]),
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
            participant_chunks.append(np.full(n_TR, str(participants_per_run[run_idx]), dtype=object))

        X_flat = np.concatenate(X_chunks, axis=0).astype(np.float32)
        Y_flat = np.concatenate(Y_chunks, axis=0).astype(np.float32)
        run_idx_per_obs = np.concatenate(run_chunks, axis=0)
        participant_per_obs = np.concatenate(participant_chunks, axis=0)
        return X_flat, Y_flat, run_idx_per_obs, participant_per_obs

    def _cross_validated_ridge(self, X_flat, Y_flat, run_idx_per_obs, n_held_out: int):
        """Hold out contiguous blocks of ``n_held_out`` runs (assembly order);
        ridge per voxel; return (n_voxels,) Pearson array.

        Voxel-batched: holding the full held_out_preds (n_obs × 20484 × 4B = 10 GB)
        plus sklearn's internal Y copies during fit blew past 64GB. We batch the
        voxel axis so peak memory is bounded by VOXEL_BATCH × n_obs × 4B (~1 GB).
        """
        n_runs = int(run_idx_per_obs.max() + 1)
        fold_masks = [
            (np.isin(run_idx_per_obs, train_runs), np.isin(run_idx_per_obs, test_runs))
            for train_runs, test_runs in contiguous_block_cv(
                n_samples=n_runs, block_size_samples=n_held_out)
        ]
        per_voxel_r, _ = cross_validated_r(
            X_flat, Y_flat, fold_masks, ridge(1.0, SklearnRidge()),
            target_batch=VOXEL_BATCH)
        return per_voxel_r

    def _cross_validated_ridge_within_participant(self, X_flat, Y_flat,
                                              run_idx_per_obs, participant_per_obs,
                                              n_held_out_per_participant: int):
        """Per-participant run-block ridge; average per-voxel r across participants.

        For each participant independently: hold out contiguous blocks of K runs across that
        participant's runs only (block_size_samples=n_held_out_per_participant), fit
        ridge on participant-internal train, predict participant-internal held-out.
        Compute per-voxel pearson r on held-out predictions stitched across
        the participant's CV folds. Final per-voxel r = mean across participants
        (NaN-aware).

        This matches standard fMRI encoding practice (Conwell, Allen, NSD)
        where train/test splits are within-participant, and avoids the cross-
        participant mean-leakage that plagues subject_out CV.
        """
        n_obs, n_voxels = Y_flat.shape
        unique_participants = np.unique(participant_per_obs)
        per_participant_r = np.full((len(unique_participants), n_voxels), np.nan,
                                    dtype=np.float32)

        for s_idx, participant in enumerate(unique_participants):
            sub_obs = (participant_per_obs == participant)
            X_sub = X_flat[sub_obs]
            Y_sub_full = Y_flat[sub_obs]
            run_sub = run_idx_per_obs[sub_obs]
            sub_unique_runs = np.unique(run_sub)
            n_sub_runs = len(sub_unique_runs)

            fold_masks = []
            for train_local, test_local in contiguous_block_cv(
                    n_samples=n_sub_runs, block_size_samples=n_held_out_per_participant):
                train_runs = sub_unique_runs[train_local]
                test_runs  = sub_unique_runs[test_local]
                fold_masks.append((np.isin(run_sub, train_runs),
                                   np.isin(run_sub, test_runs)))

            per_participant_r[s_idx], _ = cross_validated_r(
                X_sub, Y_sub_full, fold_masks, ridge(1.0, SklearnRidge()),
                target_batch=VOXEL_BATCH)

        # Mean per-voxel r across participants, ignoring NaNs.
        return np.nanmean(per_participant_r, axis=0).astype(np.float32)



def _blocks_hold_out_whole_participants(participants_per_run: List[str], block_runs: int) -> bool:
    """True if every contiguous block of ``block_runs`` runs is one participant
    whose runs are all inside that block (a true leave-one-participant-out)."""
    for train_runs, test_runs in contiguous_block_cv(
            n_samples=len(participants_per_run), block_size_samples=block_runs):
        test_participants = {participants_per_run[i] for i in test_runs}
        train_participants = {participants_per_run[i] for i in train_runs}
        if len(test_participants) != 1 or test_participants & train_participants:
            return False
    return True
