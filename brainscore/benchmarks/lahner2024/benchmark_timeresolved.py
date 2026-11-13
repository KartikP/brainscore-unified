"""
Lahner 2024 BOLDMoments — TR-resolved fMRI variant (continuous-time encoding).

Same stimuli as the existing GLM-beta variant (`Lahner2024BOLDMoments`), but
predicts the per-TR BOLD time-series of each scanning run rather than a
per-clip GLM beta. M12-lite — the first benchmark that exercises the
temporal kit (`hrf_convolve`, `contiguous_block_cv`) end-to-end.

## Design: clip-mean onset-impulse HRF model

For each (participant, session, task, run), the brain produces a continuous BOLD
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

## Pipeline (per subject)

    1. Discover unique stimuli from events (1102 unique videos).
    2. Record per-stimulus features ONCE through participate (neural_response):
       - a subject that declares ``vision:video``: whole clips, its native
         time bins mean-pooled to one vector
       - any other subject: the mean of N still frames per clip
       - with the ``audio`` channel, also ``neural:A1`` for the sound track of
         each clip, joined column-wise after the visual stream
       If a stream is wider than FEATURE_DIM_CAP, a TruncatedSVD is fit on
       ALL stimuli (including those in held-out runs) — a transductive
       projection, recorded on the score.
    3. For each run (participant, session, task, run):
       a. Place each clip's vector at floor(onset_sec / TR) within the run.
       b. HRF-convolve along the time axis (canonical SPM double-gamma).
       c. Crop to the run's valid TR length.
    4. Concatenate the runs' feature time-series into the design matrix X and
       the runs' z-scored BOLD into Y.
    5. Per-voxel ridge (alpha=1.0) with whole-run held-out CV, by the fold rule:
       - ``RunBlocks(52)`` (default): contiguous blocks of 52 runs in assembly
         order; this holds out one participant per fold only if the assembly
         lists each participant's runs together (checked and recorded).
       - ``ParticipantRunBlocks(10)``: per participant, contiguous blocks of
         10 runs; per-voxel r averaged over participants.
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

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

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
from brainscore import load_dataset, load_stimulus_set
from brainscore.metrics.encoding import SklearnRidge, cross_validated_r, ridge
from brainscore.benchmarks._score import benchmark_score

from .benchmark import (
    BIBTEX,
    _VISUAL,
    _ceiling_measured,
    _split_half_reliability,
    _select_voxels,
)
from ._util import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_SAMPLE_TIMES_MS,
    align,
    channel_regions,
    clip_table,
    frame_table,
    read_stimulus_ids,
    record_clips,
    record_visual,
    resolve_frames_dir,
    takes_audiovisual,
    time_mean,
)


# Confirmed scanner / paradigm parameters (from EC2 recon, ds005165 v1.0.4)
TR_SEC = 1.75
SOA_SEC = 4.0

# Voxels fitted together. It bounds memory (held-out predictions for all 20484
# voxels are about 10 GB).
VOXEL_BATCH = 2000

# Cap on per-stimulus feature dim before ridge — guards a 64GB OOM on g5.4xlarge
# when n_TR_obs ≈ 127k and flattened ViT features ≈ 38k.
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


# ── Fold rules ────────────────────────────────────────────────────────

class RunBlocks:
    """Hold out contiguous blocks of ``n_runs`` runs, in assembly order.

    One fold holds out one participant only if the assembly lists each
    participant's runs together; the score records whether it does.
    """

    def __init__(self, n_runs: int = 52):
        self.n_runs = n_runs

    def per_voxel_r(self, X, Y, run_idx, participant_per_obs) -> np.ndarray:
        """Held-out Pearson r per voxel; ridge at penalty 1, voxels in batches."""
        n_runs = int(run_idx.max() + 1)
        folds = [(np.isin(run_idx, train_runs), np.isin(run_idx, test_runs))
                 for train_runs, test_runs in contiguous_block_cv(
                     n_samples=n_runs, block_size_samples=self.n_runs)]
        r, _ = cross_validated_r(X, Y, folds, ridge(1.0, SklearnRidge()),
                                 target_batch=VOXEL_BATCH)
        return r

    def attrs(self, assembly) -> dict:
        # The label values are the historical ones ('subject_out'); they name participants.
        participants = [str(p) for p in assembly['subject'].values]
        return {
            'cv_mode': 'subject_out',
            'cv_n_held_out_runs': self.n_runs,
            'cv_policy': 'contiguous_run_blocks_in_assembly_order',
            'cv_block_runs': int(self.n_runs),
            'cv_blocks_hold_out_whole_subjects': _blocks_hold_out_whole_participants(
                participants, self.n_runs),
        }


class ParticipantRunBlocks:
    """Within each participant, hold out contiguous blocks of ``n_runs`` of its
    runs; the per-voxel r is averaged over participants (NaN-aware).

    Standard fMRI encoding practice (Conwell, Allen, NSD); it avoids the
    cross-participant mean leakage of ``RunBlocks``.
    """

    def __init__(self, n_runs: int = 10):
        self.n_runs = n_runs

    def per_voxel_r(self, X, Y, run_idx, participant_per_obs) -> np.ndarray:
        participants = np.unique(participant_per_obs)
        r = np.full((len(participants), Y.shape[1]), np.nan, dtype=np.float32)
        for index, participant in enumerate(participants):
            rows = participant_per_obs == participant
            runs = run_idx[rows]
            own_runs = np.unique(runs)
            folds = [(np.isin(runs, own_runs[train]), np.isin(runs, own_runs[test]))
                     for train, test in contiguous_block_cv(
                         n_samples=len(own_runs), block_size_samples=self.n_runs)]
            r[index], _ = cross_validated_r(X[rows], Y[rows], folds, ridge(1.0, SklearnRidge()),
                                            target_batch=VOXEL_BATCH)
        return np.nanmean(r, axis=0).astype(np.float32)

    def attrs(self, assembly) -> dict:
        return {
            'cv_mode': 'within_subject',
            'cv_n_held_out_runs': self.n_runs,
            'cv_policy': 'within_subject_contiguous_run_blocks',
            'cv_block_runs': int(self.n_runs),
        }


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


def _cap(features):
    """``(features, projected)``: a stream wider than ``FEATURE_DIM_CAP`` projected
    by a TruncatedSVD fit on every stimulus (transductive)."""
    if features.shape[1] <= FEATURE_DIM_CAP:
        return features, False
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=FEATURE_DIM_CAP, random_state=0)
    return svd.fit_transform(features).astype(np.float32), True


# ── Benchmark class ───────────────────────────────────────────────────

class Lahner2024BOLDMoments_timeresolved(BenchmarkBase):
    """Predict per-TR BOLD time-series for each BOLDMoments scanning run.

    Same stimuli as ``Lahner2024BOLDMoments`` (GLM betas); the neural target
    and the regression differ.

    Args:
        channels: what the benchmark sends, as for ``Lahner2024BOLDMoments``.
        folds: the fold rule, ``RunBlocks(52)`` (default) or ``ParticipantRunBlocks(10)``.
        confounds: columns of the motion sidecar regressed out of each run's
            BOLD before the z-score (``MOTION_COLUMNS``); none by default.
        ceiling, reliability_threshold, voxel_mask_fn, audio_dir: as for
            ``Lahner2024BOLDMoments``. The reliability is the GLM-beta one: both
            assemblies share the fsaverage5 voxels.
        identifier_suffix: appended to ``Lahner2024-fMRI-naturalistic``.
        version: the event key (see the module docstring).
    """

    trial_memory = 'none'   # each clip is an independent trial

    def __init__(
        self,
        channels: Sequence[str] = _VISUAL,
        folds=RunBlocks(52),
        confounds: Sequence[str] = (),
        ceiling: Optional[float] = None,
        reliability_threshold: Optional[float] = None,
        voxel_mask_fn=None,
        audio_dir: Optional[str] = None,
        identifier_suffix: str = '-timeresolved',
        version: int = CURRENT_VERSION,
    ):
        if version not in EVENT_KEY_BY_VERSION:
            raise ValueError(
                f"version must be one of {sorted(EVENT_KEY_BY_VERSION)}, got {version!r}")
        if reliability_threshold is not None and voxel_mask_fn is not None:
            raise ValueError('Give reliability_threshold or voxel_mask_fn, not both.')
        channels, regions = channel_regions(channels)
        self.required_input_channels = channels
        self.requested_output_channels = frozenset(f'neural:{region}' for region in regions)
        self._audio = 'audio' in channels
        self._folds = folds
        self._confounds = list(confounds)
        self._reliability_threshold = reliability_threshold
        self._voxel_mask_fn = voxel_mask_fn
        self._audio_dir = Path(audio_dir or DEFAULT_AUDIO_DIR).expanduser()
        self._frames_dir = resolve_frames_dir()
        self._voxel_mask: Optional[np.ndarray] = None
        self._assembly: Optional[NeuronRecordingAssembly] = None
        self._events = None
        self._motion = None
        self._stimulus_set = None
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
            self._events = load_dataset('Lahner2024-fMRI-timeresolved-events')
        return self._events

    @property
    def motion(self):
        """The motion sidecar; ``None`` without confounds."""
        if not self._confounds:
            return None
        if self._motion is None:
            self._motion = load_dataset('Lahner2024-fMRI-timeresolved-motion')
        return self._motion

    @property
    def stimulus_set(self):
        if self._stimulus_set is None:
            self._stimulus_set = load_stimulus_set('BOLDMoments')
        return self._stimulus_set

    @property
    def voxel_mask(self) -> Optional[np.ndarray]:
        """The scored voxels (``None``: all 20484); computed once."""
        if self._voxel_mask is None:
            self._voxel_mask = _select_voxels(
                self._reliability_threshold, self._voxel_mask_fn,
                lambda: _split_half_reliability(load_dataset('Lahner2024-fMRI')))
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
        # 1. One feature vector per unique stimulus.
        features, stimulus_ids, stream_attrs = self._features(subject)

        # 2. Place each clip's vector at its onset TR, HRF-convolve within each run.
        assembly = self.assembly
        design = self._build_design(features, stimulus_ids, assembly)

        # 3. Flatten (run, TR) into rows; regress the confounds out of the BOLD and
        #    z-score it per run and voxel.
        X_flat, Y_flat, run_idx_per_obs, participant_per_obs = self._concatenate_with_mask(
            design, assembly)

        # 4. Per-voxel ridge under the fold rule; 5. keep the scored voxels.
        per_voxel_r = self._folds.per_voxel_r(X_flat, Y_flat, run_idx_per_obs,
                                              participant_per_obs)
        mask = self.voxel_mask
        if mask is not None:
            per_voxel_r = per_voxel_r[mask]
        per_voxel_r_finite = per_voxel_r[~np.isnan(per_voxel_r)]
        median_r = float(np.median(per_voxel_r_finite))
        mean_r = float(np.mean(per_voxel_r_finite))

        score = benchmark_score(
            self, median_r / float(self.ceiling), raw=median_r,
            metric='median_per_voxel_pearson_r', normalized=_ceiling_measured(self))
        score.attrs['mean_r'] = mean_r
        score.attrs['n_voxels_scored'] = int(len(per_voxel_r_finite))
        score.attrs['n_runs'] = assembly.sizes['presentation']
        score.attrs['n_observations'] = int(len(Y_flat))
        score.attrs['motion_regressed'] = bool(self._confounds)
        score.attrs['pipeline'] = 'continuous_time_encoding'
        score.attrs.update(self._folds.attrs(assembly))
        score.attrs.update(self._method_attrs(stream_attrs.pop('feature_projection_protocol')))
        score.attrs['ridge_alpha'] = 1.0
        score.attrs['inner_validation'] = 'none_fixed_alpha'
        score.attrs.update(stream_attrs)
        return score

    # ── Score labels ───────────────────────────────────────────────

    def _method_attrs(self, feature_projection: str) -> dict:
        """Version, event key, known defects and method description (3A)."""
        # benchmark_version is a common attribute, written by benchmark_score.
        attrs = dict(METHOD_ATTRS)
        attrs['event_key'] = EVENT_KEY_BY_VERSION[self.version]
        attrs['known_defects'] = KNOWN_DEFECTS_BY_VERSION[self.version]
        attrs['feature_projection_protocol'] = feature_projection
        attrs['feature_dim_cap'] = FEATURE_DIM_CAP
        return attrs

    # ── Features ───────────────────────────────────────────────────

    def _features(self, subject):
        """``(features (n_stimuli, n_features), stimulus_ids, attrs)``.

        One forward pass per unique stimulus of the events (at most 1102 videos).
        """
        stimuli = self.stimulus_set
        unique_ids = sorted(set(self.events['stimulus_id'].unique().tolist())
                            & set(stimuli['stimulus_id'].astype(str).tolist()))
        frames = lambda: (lambda table: table[table['clip_id'].isin(unique_ids)])(  # noqa: E731
            frame_table(stimuli, list(DEFAULT_SAMPLE_TIMES_MS), self._frames_dir))
        # A subject that takes no audio, or records no A1, is scored on the visual
        # stream alone, also with the audio channel (the GLM-beta benchmark refuses
        # it instead; kept as it was).
        if not (self._audio and takes_audiovisual(subject)):
            def clips():
                table = clip_table(stimuli, f'{stimuli.identifier}-videos')
                return table[table['stimulus_id'].isin(unique_ids)]
            visual = record_visual(subject, clips, frames)
            features, projected = _cap(time_mean(visual))
            attrs = {'feature_projection_protocol': TRANSDUCTIVE_SVD if projected else 'none'}
            if self._audio:
                attrs.update(mode='concat', pipeline='tr_resolved_multimodal_concat')
            return features, read_stimulus_ids(visual), attrs
        # Two streams. Differences from the one-stream route above, kept so no score
        # moves: the clip table is built in sorted stimulus order and named
        # '-timeresolved-multimodal-video' (the one-stream table keeps the table
        # order), each stream is projected on its own, and the joined features are
        # cast to float32.
        visual = record_visual(
            subject, lambda: clip_table(stimuli, f'{stimuli.identifier}-timeresolved-multimodal-video',
                                        ids=unique_ids), frames)
        sound = record_clips(subject, clip_table(
            stimuli, f'{stimuli.identifier}-timeresolved-multimodal-audio', ids=unique_ids,
            audio_dir=self._audio_dir), 'A1')
        stimulus_ids = read_stimulus_ids(visual)
        v_features, v_projected = _cap(time_mean(visual))
        a_features, a_projected = _cap(align(time_mean(sound), read_stimulus_ids(sound),
                                             stimulus_ids))
        features = np.concatenate([v_features, a_features], axis=1).astype(np.float32)
        return features, stimulus_ids, {
            'feature_projection_protocol': (TRANSDUCTIVE_SVD + '_per_modality'
                                            if v_projected or a_projected else 'none'),
            'n_features_video': int(v_features.shape[1]),
            'n_features_audio': int(a_features.shape[1]),
            'mode': 'concat',
            'pipeline': 'tr_resolved_multimodal_concat',
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
                    continue   # oddball or stimulus the subject was not shown
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

    def _events_for_run(self, participant, session, task, run, groups: dict):
        """Events of one run (participant, session, task, run).

        Raises if the run has no events, or if the selection holds more than
        one task (two tasks must never feed one run's design).
        """
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

        Per run:
          1. Slice to n_valid_TR (drop padding).
          2. With confounds: OLS of the BOLD on [1, confounds] per voxel, keep the
             residuals (Power 2014, Ciric 2017).
          3. Per-voxel z-score. Removes participant-specific DC that would
             otherwise leak into the ridge intercept across folds.

        Returns ``(X_flat, Y_flat, run_idx_per_obs, participant_per_obs)``;
        ``run_idx_per_obs`` indexes the assembly's presentation axis.
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
                (s, ses, t, r): g[self._confounds].values.astype(np.float32)
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
