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

## What these benchmarks are, and what they are not

- They regress the continuous scanner time series of each run on the model's
  features placed at the video onsets and convolved with a response shape (the
  double-gamma HRF). The registered variants report the raw correlation; none
  passes a noise ceiling.
- The folds hold out whole runs, not videos. Every participant saw every
  video (each training-set video 3 times, each test-set video 10 times: 4020
  events a participant). Under the participant-out rule (every variant
  without ``-improved``) every scored video therefore also occurs in the
  training runs. Under the within-participant rule (the ``-improved``
  variants) a scored video also occurs in the training runs unless all its
  showings fall in the held-out block of runs.
- The studies of this data set, the data set paper included, analyze responses
  locked to each video, fitted on the training-set videos and tested on the
  test-set videos. These benchmarks do not follow that protocol, so their values
  are not comparable with published values. The GLM benchmark
  (``Lahner2024BOLDMoments``) follows it.
- The events name 1102 videos (1000 training-set, 102 test-set); the hosted
  stimulus set holds 1026 of them. The 67 training-set videos and 9 test-set
  videos that it lacks get no features: their events add nothing to the design,
  while the scanner responses to them stay in the scored time series.
- The voxel mask of the ``-visualROI`` variants is reliability of at least 0.3
  on the 93 test-set videos of the GLM assembly (7209 voxels on the hosted
  data). The GLM class selects on its training-set videos (3958 voxels), so the
  two classes' visual-ROI scores cover different voxels.

## Pipeline (per subject)

    1. Discover the unique stimuli of the events of the scored runs (the stimulus set holds 1026 of the 1102 videos).
    2. Record per-stimulus features ONCE through participate (neural_response):
       - a subject that declares ``vision:video``: whole clips, its native
         time bins mean-pooled to one vector
       - any other subject: the mean of N still frames per clip
       - with the ``audio`` channel, also ``neural:A1`` for the sound track of
         each clip, joined column-wise after the visual stream
       If a stream is wider than FEATURE_DIM_CAP, it is projected by a
       TruncatedSVD fitted, in each fold, on the stimuli shown in the training
       runs of that fold (step 5); the same projection then transforms every
       stimulus. The protocol is recorded on the score.
    3. For each fold, with the features of the fold (a stream wider than the cap
       is projected by the SVD of the fold, as above), and for each run
       (participant, session, task, run):
       a. Place each clip's vector at floor(onset_sec / TR) within the run.
       b. HRF-convolve along the time axis (canonical SPM double-gamma).
       c. Crop to the run's valid TR length.
    4. Concatenate the runs' feature time-series into the design matrix X of
       the fold and the runs' z-scored BOLD of the scored voxels into Y.
    5. Per-voxel ridge (alpha=1.0) with whole-run held-out CV, by the fold rule:
       - ``RunBlocks(52)`` (default): contiguous blocks of 52 runs in assembly
         order; this holds out one participant per fold only if the assembly
         lists each participant's runs together (checked and recorded).
       - ``ParticipantRunBlocks(10)``: per participant, contiguous blocks of
         10 runs; per-voxel r averaged over participants.
       Under the participant-out rule every held-out video is also shown in
       training runs, so the score measures generalization to new runs of seen
       videos, not to new videos (see "What these benchmarks are").
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

Same models and stimulus set, but the target (single-run BOLD time series
vs. GLM betas averaged over participants), the protocol (whole runs held out vs.
one fit on the training-set videos and the score on the test-set videos), the
visual-ROI voxels, the voxel noise level and the feature design all differ. A score difference between this variant and
`Lahner2024-fMRI-naturalistic` therefore does not isolate the contribution
of temporal information.
"""

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
from brainscore.benchmarks._score import benchmark_score, not_estimated

from .benchmark import (
    BIBTEX,
    _VISUAL,
    _select_voxels,
    presented_stimuli,
    reliability_on_test_set,
    voxel_score,
)
from brainscore.data.lahner2024 import DEFAULT_SAMPLE_TIMES_MS
from ._util import (
    align,
    channel_regions,
    clip_table,
    frame_table,
    read_stimulus_ids,
    record_clips,
    record_visual,
    stream_regions,
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
TRAINING_SVD = 'truncated_svd_fit_on_training_stimuli_of_each_fold'


# ── Fold rules ────────────────────────────────────────────────────────

class RunBlocks:
    """Hold out contiguous blocks of ``n_runs`` runs, in assembly order.

    One fold holds out one participant only if the assembly lists each
    participant's runs together; the score records whether it does.
    """

    def __init__(self, n_runs: int = 52):
        self.n_runs = n_runs

    def per_voxel_r(self, features_for, Y, run_idx, participant_per_obs) -> np.ndarray:
        """Held-out Pearson r per voxel; ridge at penalty 1, voxels in batches.

        ``features_for(train_runs, scope_runs)`` returns the design rows of the
        scope runs, prepared from the training runs of the fold.
        """
        n_runs = int(run_idx.max() + 1)
        scope = np.arange(n_runs)
        splits = list(contiguous_block_cv(n_samples=n_runs, block_size_samples=self.n_runs))
        folds = [(np.isin(run_idx, train_runs), np.isin(run_idx, test_runs))
                 for train_runs, test_runs in splits]
        r, _ = cross_validated_r(lambda i: features_for(splits[i][0], scope), Y, folds,
                                 ridge(1.0, SklearnRidge()), target_batch=VOXEL_BATCH)
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

    def per_voxel_r(self, features_for, Y, run_idx, participant_per_obs) -> np.ndarray:
        participants = np.unique(participant_per_obs)
        r = np.full((len(participants), Y.shape[1]), np.nan, dtype=np.float32)
        for index, participant in enumerate(participants):
            rows = participant_per_obs == participant
            runs = run_idx[rows]
            own_runs = np.unique(runs)
            splits = list(contiguous_block_cv(
                n_samples=len(own_runs), block_size_samples=self.n_runs))
            folds = [(np.isin(runs, own_runs[train]), np.isin(runs, own_runs[test]))
                     for train, test in splits]
            r[index], _ = cross_validated_r(
                lambda i: features_for(own_runs[splits[i][0]], own_runs), Y[rows], folds,
                ridge(1.0, SklearnRidge()), target_batch=VOXEL_BATCH)
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


def _reduce(features, train):
    """``features`` projected by a TruncatedSVD fitted on the rows ``train`` (a
    boolean mask), in their order in ``features``; every row is transformed. A
    stream at or below ``FEATURE_DIM_CAP`` is returned as it is.

    The training rows take the output of ``fit_transform``, the others the
    ``transform``: with every row in ``train`` the result is the old fit on all
    stimuli, to the last digit. With fewer training stimuli than components,
    scikit-learn returns one component per training stimulus (no error).
    """
    if features.shape[1] <= FEATURE_DIM_CAP:
        return features
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=FEATURE_DIM_CAP, random_state=0)
    fitted = svd.fit_transform(features[train])
    out = np.empty((len(features), fitted.shape[1]), dtype=np.float32)
    out[train] = fitted
    if not train.all():
        out[~train] = svd.transform(features[~train])
    return out


# ── Benchmark class ───────────────────────────────────────────────────

class Lahner2024BOLDMoments_timeresolved(BenchmarkBase):
    """Predict per-TR BOLD time-series for each BOLDMoments scanning run.

    Same stimuli as ``Lahner2024BOLDMoments`` (GLM betas); the neural target
    and the regression differ.

    What is scored. Every participant has 52 runs: 40 ``train`` task runs (the
    training-set videos) and 12 ``test`` task runs (the 102 test-set videos). A
    variant without a reliability mask scores all 52 runs. A variant with
    ``reliability_threshold`` scores the ``train`` runs only (``scored_assembly``),
    because its voxel mask comes from the GLM reliability of the test-set videos
    (``reliability_on_test_set``; the hosted GLM assembly holds 1026 of the 1102
    videos, 93 of them in the test set): a ``test`` run is in no fold, and the
    videos shown only in it are not presented to the subject. The run filter comes
    before the folds, the reduction and the design. With 40 runs a participant, the
    default rule holds out blocks of 40 (``RunBlocks(40)``, set in the registry
    line), which is still one participant per block.

    Args:
        channels: what the benchmark sends, as for ``Lahner2024BOLDMoments``.
        folds: the fold rule, ``RunBlocks(52)`` (default) or ``ParticipantRunBlocks(10)``.
        confounds: columns of the motion sidecar regressed out of each run's
            BOLD before the z-score (``MOTION_COLUMNS``); none by default.
        ceiling: a scalar; when given, the raw median is divided by it. No
            registered variant passes one: the values are raw correlations.
        reliability_threshold: score only the voxels whose GLM-beta reliability
            on the 93 test-set videos (``reliability_on_test_set``) is at least
            this value; both assemblies share the fsaverage5 voxels. This differs
            from ``Lahner2024BOLDMoments``, which selects on its training-set
            videos. With a threshold only the ``train`` task runs are scored;
            state the fold rule for 40 runs a participant in that case.
        voxel_mask_fn: score only the voxels of this mask (for example
            ``build_auditory_mask``). Give at most one of the two.
        identifier: the benchmark identifier (required; the registry key of the variant).
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
        *,
        identifier: str,
    ):
        if reliability_threshold is not None and voxel_mask_fn is not None:
            raise ValueError('Give reliability_threshold or voxel_mask_fn, not both.')
        channels, regions = channel_regions(channels)
        self.required_input_channels = channels
        self.requested_output_channels = frozenset(f'neural:{region}' for region in regions)
        self._regions = regions
        self._audio = 'audio' in channels
        self._folds = folds
        self._confounds = list(confounds)
        self._reliability_threshold = reliability_threshold
        self._voxel_mask_fn = voxel_mask_fn
        self._voxel_mask: Optional[np.ndarray] = None
        self._assembly: Optional[NeuronRecordingAssembly] = None
        self._scored_assembly: Optional[NeuronRecordingAssembly] = None
        self._events = None
        self._motion = None
        self._stimulus_set = None
        super().__init__(
            identifier=identifier,
            version=2,
            parent='naturalistic',
            ceiling=not_estimated() if ceiling is None else Score(float(ceiling)),
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
                lambda: reliability_on_test_set(load_dataset('Lahner2024-fMRI')))
        return self._voxel_mask

    @property
    def scored_assembly(self) -> NeuronRecordingAssembly:
        """The runs that are scored. With a reliability mask, the ``train`` task runs only:
        the ``test`` task runs show the videos whose betas chose the voxels, so they are in
        no fold and their videos are not presented. Otherwise every run."""
        if self._scored_assembly is None:
            assembly = self.assembly
            if self._reliability_threshold is not None:
                assembly = assembly.isel(presentation=np.flatnonzero(
                    np.asarray(assembly['task'].values).astype(str) == 'train'))
            self._scored_assembly = assembly
        return self._scored_assembly

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
        # 1. One feature vector per unique stimulus; ``prepare(train)`` reduces
        #    the streams with the SVD fitted on the stimuli ``train`` (a mask).
        assembly = self.scored_assembly
        prepare, stimulus_ids, stream_attrs = self._features(subject, assembly)

        # 2. The BOLD of the scored voxels, flattened to (run, TR) rows; the
        #    confounds regressed out and the BOLD z-scored per run and voxel.
        Y_flat, run_idx_per_obs, participant_per_obs = self._concatenate_with_mask(
            assembly, self.voxel_mask)

        # 3. The design of a fold: the features reduced on the stimuli shown in the
        #    training runs of the fold, placed at the onsets, HRF-convolved. One
        #    fit for each distinct set of training stimuli.
        run_events = list(self._events_per_run(assembly))
        n_valid = assembly['n_valid_TR'].values.astype(int)
        shown = [frozenset(events['stimulus_id'].astype(str)) for events in run_events]
        projections, designs = {}, {}

        def features_for(train_runs, scope_runs):
            trained = frozenset().union(*(shown[run] for run in train_runs))
            if trained not in projections:
                projections[trained] = prepare(
                    np.array([stimulus in trained for stimulus in stimulus_ids]))
            key = (trained, tuple(scope_runs))
            if key not in designs:
                design = self._build_design(projections[trained], stimulus_ids, assembly,
                                            run_events, scope_runs)
                designs[key] = np.concatenate(
                    [design[position, :n_valid[run]] for position, run in enumerate(scope_runs)],
                    axis=0).astype(np.float32)
            return designs[key]

        # 4. Per-voxel ridge under the fold rule.
        per_voxel_r = self._folds.per_voxel_r(features_for, Y_flat, run_idx_per_obs,
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

    def _features(self, subject, assembly):
        """``(prepare, stimulus_ids, attrs)``.

        ``prepare(train)`` gives the features ``(n_stimuli, n_features)``, a stream
        wider than the cap reduced by an SVD fitted on the stimuli of the boolean
        mask ``train``. One forward pass per unique stimulus of the events (at
        most 1102 videos).
        """
        # The videos shown in the scored runs that the stimulus set holds; the
        # table presented carries an identifier that names this set.
        known = set(self.stimulus_set['stimulus_id'].astype(str))
        shown = {str(stimulus_id) for events in self._events_per_run(assembly)
                 for stimulus_id in events['stimulus_id']}
        unique_ids = sorted(shown & known)
        stimuli = presented_stimuli(self.stimulus_set, unique_ids)
        frames = lambda: frame_table(stimuli, list(DEFAULT_SAMPLE_TIMES_MS))  # noqa: E731
        # Every region the benchmark names is requested (rule 5.1a). The streams
        # shown are the benchmark inputs the subject takes in, matched as pre-flight
        # matches them; each region is recorded from a stream the subject can answer
        # it from (stream_regions). 'scored_inputs' names the inputs shown (rules
        # 5.4, 5.5); the two-stream labels are set only when both are shown.
        plan = stream_regions(subject, taken_inputs(subject, self.required_input_channels),
                              self._regions)
        scored = sorted(channel for _, channel, _ in plan)
        if [stream for stream, _, _ in plan] == ['video']:
            responses = record_visual(
                subject, lambda: clip_table(stimuli, f'{stimuli.identifier}-videos'),
                frames, plan[0][2])
            stimulus_ids = read_stimulus_ids(responses[0])
            features = np.concatenate(
                [align(time_mean(response), read_stimulus_ids(response), stimulus_ids)
                 for response in responses], axis=1)
            attrs = {'feature_projection_protocol':
                     TRAINING_SVD if features.shape[1] > FEATURE_DIM_CAP else 'none'}
            if self._audio:
                attrs['scored_inputs'] = scored
            return (lambda train: _reduce(features, train)), stimulus_ids, attrs
        # The sound, with the visual stream when the subject takes it. Differences
        # from the one-stream route above, kept so no score moves: the clip table is
        # built in sorted stimulus order and named '-timeresolved-multimodal-video'
        # (the one-stream table keeps the table order), each stream is projected on
        # its own, and the joined features are cast to float32.
        # The sound tracks are checked before any stream is recorded.
        sound = clip_table(stimuli, f'{stimuli.identifier}-timeresolved-multimodal-audio', ids=unique_ids,
                           sound=True)
        streams = []
        for stream, _, regions in plan:
            if stream == 'video':
                responses = record_visual(subject, lambda: clip_table(
                    stimuli, f'{stimuli.identifier}-timeresolved-multimodal-video', ids=unique_ids),
                    frames, regions)
            else:
                responses = record_clips(subject, sound, regions)
            streams.append((stream, responses))
        stimulus_ids = read_stimulus_ids(streams[0][1][0])
        blocks = [(stream, np.concatenate(
            [align(time_mean(response), read_stimulus_ids(response), stimulus_ids)
             for response in responses], axis=1)) for stream, responses in streams]

        def prepare(train):
            return np.concatenate([_reduce(block, train) for _, block in blocks],
                                  axis=1).astype(np.float32)
        # The width of a stream after the reduction, with every stimulus in the fit
        # (a fold with fewer training stimuli than components has fewer columns).
        attrs = {f'n_features_{stream}': int(min(block.shape[1], FEATURE_DIM_CAP, len(block)))
                 if block.shape[1] > FEATURE_DIM_CAP else int(block.shape[1])
                 for stream, block in blocks}
        attrs.update(
            feature_projection_protocol=(TRAINING_SVD + '_per_modality'
                                         if any(block.shape[1] > FEATURE_DIM_CAP
                                                for _, block in blocks) else 'none'),
            scored_inputs=scored)
        if len(streams) == 2:
            attrs.update(mode='concat', pipeline='tr_resolved_multimodal_concat')
        return prepare, stimulus_ids, attrs

    # ── Design matrix ──────────────────────────────────────────────

    def _build_design(self, per_stim_features, stimulus_ids, assembly, run_events,
                      runs) -> np.ndarray:
        """Clip-mean onset-impulse design, HRF-convolved within each run.

        ``run_events`` lists the events of every assembly run; ``runs`` are the
        positions of the runs built. Returns ``(len(runs), n_TR_max, n_features)``.
        Each event adds its clip's feature vector at ``floor(onset_sec / TR)``;
        the sub-TR part of the onset and the event duration are not used.
        """
        stim_idx = {sid: i for i, sid in enumerate(stimulus_ids)}
        n_TR_max = assembly.sizes['time_bin']
        n_features = per_stim_features.shape[1]

        feature_ts = np.zeros((len(runs), n_TR_max, n_features), dtype=np.float32)
        for position, run in enumerate(runs):
            for _, ev in run_events[run].iterrows():
                if ev['stimulus_id'] not in stim_idx:
                    continue   # oddball or stimulus the subject was not shown
                onset_TR = int(np.floor(ev['onset_sec'] / TR_SEC))
                if 0 <= onset_TR < n_TR_max:
                    feature_ts[position, onset_TR, :] += per_stim_features[stim_idx[ev['stimulus_id']]]

        # HRF-convolve along time axis. hrf_convolve expects (n_time, n_features).
        hrf_kernel = double_gamma_hrf(duration_sec=32.0, sampling_rate_hz=1.0/TR_SEC)
        feature_ts_convolved = np.zeros_like(feature_ts)
        for position in range(len(runs)):
            feature_ts_convolved[position] = hrf_convolve(
                feature_ts[position], sampling_rate_hz=1.0/TR_SEC, hrf=hrf_kernel)
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

    def _concatenate_with_mask(self, assembly, voxels=None):
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

        Returns ``(Y_flat, run_idx_per_obs, participant_per_obs)``;
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

        Y_chunks, run_chunks, participant_chunks = [], [], []
        for run_idx, n_TR in enumerate(n_valid):
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

        Y_flat = np.concatenate(Y_chunks, axis=0).astype(np.float32)
        run_idx_per_obs = np.concatenate(run_chunks, axis=0)
        participant_per_obs = np.concatenate(participant_chunks, axis=0)
        return Y_flat, run_idx_per_obs, participant_per_obs
