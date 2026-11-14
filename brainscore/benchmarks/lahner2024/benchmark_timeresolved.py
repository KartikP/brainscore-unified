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
       the runs' z-scored BOLD of the scored voxels into Y.
    5. Per-voxel ridge (alpha=1.0) with whole-run held-out CV, by the fold rule:
       - ``RunBlocks(52)`` (default): contiguous blocks of 52 runs in assembly
         order; this holds out one participant per fold only if the assembly
         lists each participant's runs together (checked and recorded).
       - ``ParticipantRunBlocks(10)``: per participant, contiguous blocks of
         10 runs; per-voxel r averaged over participants.
       Every held-out stimulus is also shown in training runs, so the score
       measures generalization to new runs of seen stimuli, not to new
       stimuli.
    6. Per-voxel Pearson on held-out predictions, median across the voxels
       that count (``voxel_score``, the rule of the GLM-beta benchmark).

## Versions

Version 2 is the only version. Compared with version 1 it changes:

- Events are selected by the full run key (subject, session, task, run).
  Version 1 left out the task; the two BOLDMoments tasks ('train', 'test')
  reuse the other keys, so 240 of the 520 runs also received the other
  task's events.
- The voxel mask is applied before the fit. Version 1 fitted all 20484
  voxels and kept the region after. Every step of the fit treats a voxel on
  its own (confound regression, z-score, ridge at a fixed penalty, Pearson),
  so in exact arithmetic the r of a kept voxel does not depend on the other
  voxels. The confound regression and the z-score run in float64 per voxel and
  round alike in any selection; only scikit-learn's float32 ridge rounds a
  voxel's fit by the voxels in its batch, so a region score moves in the last
  digits. Against commit eb6a4d5 (float32 preprocessing) scores move by rounding
  (at most 8e-7 on synthetic data). A region needs its share of memory and time.
- The score keeps ``per_voxel_r`` (one value per scored voxel, NaN where a
  voxel has no correlation) and counts the left-out voxels in
  ``n_voxels_excluded``.
- The fold labels name participants: ``cv_mode`` is ``'participant_out'`` or
  ``'within_participant'``, and ``cv_blocks_hold_out_whole_participants``
  replaces ``cv_blocks_hold_out_whole_subjects``. The data keep their own
  coordinate name ``subject`` (the participant of a run).

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
    _split_half_reliability,
    _select_voxels,
    voxel_score,
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
    taken_inputs,
    time_mean,
)


# Confirmed scanner / paradigm parameters (from EC2 recon, ds005165 v1.0.4)
TR_SEC = 1.75
SOA_SEC = 4.0

# Voxels fitted together. It bounds memory (held-out predictions for all 20484
# voxels are about 10 GB). At a fixed penalty the batch changes a voxel's r only by
# float32 rounding.
VOXEL_BATCH = 2000

# Cap on per-stimulus feature dim before ridge — guards a 64GB OOM on g5.4xlarge
# when n_TR_obs ≈ 127k and flattened ViT features ≈ 38k.
FEATURE_DIM_CAP = 512
CLIP_DURATION_SEC = 3.0

# The columns that name one run in the assembly, the events and the motion
# sidecar. They are the data's own names: ``subject`` is the participant.
RUN_KEY = ('subject', 'session', 'task', 'run')

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
    'event_key': RUN_KEY,
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
        participants = [str(p) for p in assembly['subject'].values]
        return {
            'cv_mode': 'participant_out',
            'cv_n_held_out_runs': self.n_runs,
            'cv_policy': 'contiguous_run_blocks_in_assembly_order',
            'cv_block_runs': int(self.n_runs),
            'cv_blocks_hold_out_whole_participants': _blocks_hold_out_whole_participants(
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
            'cv_mode': 'within_participant',
            'cv_n_held_out_runs': self.n_runs,
            'cv_policy': 'within_participant_contiguous_run_blocks',
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
    ):
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
            version=2,
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
        assert 'task' in presentation_levels, presentation_levels   # part of RUN_KEY
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

        # 3. Flatten (run, TR) into rows, the scored voxels only; regress the
        #    confounds out of the BOLD and z-score it per run and voxel.
        X_flat, Y_flat, run_idx_per_obs, participant_per_obs = self._concatenate_with_mask(
            design, assembly, self.voxel_mask)

        # 4. Per-voxel ridge under the fold rule.
        per_voxel_r = self._folds.per_voxel_r(X_flat, Y_flat, run_idx_per_obs,
                                              participant_per_obs)

        score = voxel_score(self, per_voxel_r)
        score.attrs['n_runs'] = assembly.sizes['presentation']
        score.attrs['n_observations'] = int(len(Y_flat))
        score.attrs['motion_regressed'] = bool(self._confounds)
        score.attrs['pipeline'] = 'continuous_time_encoding'
        score.attrs.update(self._folds.attrs(assembly))
        score.attrs.update(METHOD_ATTRS)
        score.attrs['feature_projection_protocol'] = stream_attrs.pop('feature_projection_protocol')
        score.attrs['feature_dim_cap'] = FEATURE_DIM_CAP
        score.attrs['ridge_alpha'] = 1.0
        score.attrs['inner_validation'] = 'none_fixed_alpha'
        score.attrs.update(stream_attrs)
        return score

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
        # The streams are the benchmark inputs the subject takes in, matched as
        # pre-flight matches them. A subject that takes one stream of the sound
        # variant is scored on it alone; 'scored_inputs' names it (rules 5.4, 5.5).
        taken = taken_inputs(subject, self.required_input_channels)
        if 'audio' not in taken:
            def clips():
                table = clip_table(stimuli, f'{stimuli.identifier}-videos')
                return table[table['stimulus_id'].isin(unique_ids)]
            visual = record_visual(subject, clips, frames)
            features, projected = _cap(time_mean(visual))
            attrs = {'feature_projection_protocol': TRANSDUCTIVE_SVD if projected else 'none'}
            if self._audio:
                attrs['scored_inputs'] = sorted(taken)
            return features, read_stimulus_ids(visual), attrs
        # The sound, with the visual stream when the subject takes it. Differences
        # from the one-stream route above, kept so no score moves: the clip table is
        # built in sorted stimulus order and named '-timeresolved-multimodal-video'
        # (the one-stream table keeps the table order), each stream is projected on
        # its own, and the joined features are cast to float32.
        streams = []
        if taken & set(_VISUAL):
            streams.append(('video', record_visual(subject, lambda: clip_table(
                stimuli, f'{stimuli.identifier}-timeresolved-multimodal-video', ids=unique_ids),
                frames)))
        streams.append(('audio', record_clips(subject, clip_table(
            stimuli, f'{stimuli.identifier}-timeresolved-multimodal-audio', ids=unique_ids,
            audio_dir=self._audio_dir), 'A1')))
        stimulus_ids = read_stimulus_ids(streams[0][1])
        capped = [(name, *_cap(align(time_mean(response), read_stimulus_ids(response), stimulus_ids)))
                  for name, response in streams]
        features = np.concatenate([block for _, block, _ in capped], axis=1).astype(np.float32)
        attrs = {f'n_features_{name}': int(block.shape[1]) for name, block, _ in capped}
        attrs.update(
            feature_projection_protocol=(TRANSDUCTIVE_SVD + '_per_modality'
                                         if any(projected for *_, projected in capped) else 'none'),
            scored_inputs=sorted(taken))
        if len(streams) == 2:
            attrs.update(mode='concat', pipeline='tr_resolved_multimodal_concat')
        return features, stimulus_ids, attrs

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
        """Yield each assembly run's events, selected by ``RUN_KEY``."""
        keys = zip(*(assembly[column].values for column in RUN_KEY))
        groups = self._events_by_run_key()
        for participant, session, task, run in keys:
            yield self._events_for_run(
                str(participant), str(session), str(task), str(run), groups=groups)

    def _events_by_run_key(self) -> dict:
        """Group events by ``RUN_KEY``, keeping row order."""
        ev = self.events
        key = list(RUN_KEY)
        missing = [column for column in key if column not in ev.columns]
        if missing:
            raise ValueError(
                f"Events table lacks {missing}; events are selected by "
                f"{RUN_KEY}. Use the versioned events sidecar.")
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

    def _concatenate_with_mask(self, feature_ts_convolved, assembly, voxels=None):
        """Flatten (run, TR, ...) into long (n_obs, ...) tables, masking padding.

        ``voxels`` (a boolean mask, ``None`` for all) keeps the scored voxels
        before any other step. Per run:
          1. Slice to n_valid_TR (drop padding).
          2. With confounds: OLS of the BOLD on [1, confounds] per voxel, keep the
             residuals (Power 2014, Ciric 2017).
          3. Per-voxel z-score. Removes participant-specific DC that would
             otherwise leak into the ridge intercept across folds. A voxel's
             values in the run stay zero when its input is constant in the run
             (before step 2) or when the standard deviation after step 2 is at
             most 1e-12 times its largest absolute input value in the run (the
             float64 rounding of step 2, not signal). Both tests read only that
             voxel.

        Returns ``(X_flat, Y_flat, run_idx_per_obs, participant_per_obs)``;
        ``run_idx_per_obs`` indexes the assembly's presentation axis.
        """
        n_valid = assembly['n_valid_TR'].values.astype(int)
        bold = assembly.transpose('presentation', 'time_bin', 'neuroid').values
        participants_per_run = assembly['subject'].values
        sessions_per_run = assembly['session'].values
        runs_per_run     = assembly['run'].values
        tasks_per_run    = assembly['task'].values

        # Pre-index motion sidecar by RUN_KEY for O(1) lookup.
        motion_df = self.motion
        motion_lookup = None
        if motion_df is not None:
            motion_lookup = {
                (s, ses, t, r): g[self._confounds].values.astype(np.float32)
                for (s, ses, t, r), g in motion_df.groupby(list(RUN_KEY), sort=False)
            }

        X_chunks, Y_chunks, run_chunks, participant_chunks = [], [], [], []
        for run_idx, n_TR in enumerate(n_valid):
            X_chunks.append(feature_ts_convolved[run_idx, :n_TR, :])
            Y_run = bold[run_idx, :n_TR, :]
            if voxels is not None:
                Y_run = Y_run[:, voxels]
            # float64, one contiguous row per voxel, sums along a row only: no voxel affects another.
            Y_run = np.array(Y_run.T, dtype=np.float64, order='C')
            varies = np.nanstd(Y_run, axis=1, keepdims=True) > 0     # an all-NaN voxel: False
            floor = 1e-12 * np.nanmax(np.abs(Y_run), axis=1, keepdims=True)

            if motion_lookup is not None:
                key = (str(participants_per_run[run_idx]),
                       str(sessions_per_run[run_idx]),
                       str(tasks_per_run[run_idx]),
                       str(runs_per_run[run_idx]))
                M = motion_lookup.get(key)
                if M is not None and len(M) >= n_TR:
                    # OLS with intercept: residuals of Y on [1, M].
                    M_design = np.concatenate([np.ones((n_TR, 1)), M[:n_TR]], axis=1)
                    filled = np.nan_to_num(Y_run)
                    for p, m in zip(np.linalg.pinv(M_design), M_design.T):
                        Y_run = Y_run - (filled * p).sum(axis=1, keepdims=True) * m

            mu = np.nanmean(Y_run, axis=1, keepdims=True)
            sd = np.nanstd(Y_run, axis=1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                Y_run = np.where(varies & (sd > floor), (Y_run - mu) / sd, 0.0).T.astype(np.float32)
            Y_chunks.append(Y_run)
            run_chunks.append(np.full(n_TR, run_idx, dtype=np.int32))
            participant_chunks.append(np.full(n_TR, str(participants_per_run[run_idx]), dtype=object))

        X_flat = np.concatenate(X_chunks, axis=0).astype(np.float32)
        Y_flat = np.concatenate(Y_chunks, axis=0).astype(np.float32)
        run_idx_per_obs = np.concatenate(run_chunks, axis=0)
        participant_per_obs = np.concatenate(participant_chunks, axis=0)
        return X_flat, Y_flat, run_idx_per_obs, participant_per_obs
