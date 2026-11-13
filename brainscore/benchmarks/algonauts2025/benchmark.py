"""The Courtois NeuroMod Friends benchmark of the Algonauts 2025 challenge.

What it uses. The registered identifiers score the ``friends`` split: all the
fMRI the challenge released for one participant (Friends seasons 1 to 6 and the
four Movie10 films; 1000 Schaefer parcels, TR = 1.49 s). The data set has four
participants (1, 2, 3, 5):

- ``Algonauts2025-friends-sub{01,02,03,05}``: frames; the full released data of
  that participant.
- ``Algonauts2025-friends-sub01-video`` and ``-video-audio``: clips, and clips
  with sound; the full released data of participant 1 only.

Friends season 7 (``split='friends_s7'``) and the out-of-distribution films
(``split='ood'``) have no released fMRI: they are not registered and not scored
(``brainscore.tools.algonauts2025_predictions`` predicts them). ``movies`` limits
a benchmark to named films, a part of the data; the score lists them.

What it shows. One table row per scan, from the files the data plugin prepared
(``brainscore.data.algonauts2025``): the frame at the middle of the scan on
``vision:image`` (default), the 1.49 s clip of the scan on ``vision:video``, or
the clip and its sound on ``vision:video`` and ``audio``. With ``streaming=True``
the subject gets one scan at a time; otherwise the whole table at once.

What it requests. ``neural:V1``, ``neural:V4`` and ``neural:IT``, and
``neural:A1`` with sound (``CHANNEL_REGIONS``), in one session. Each answer is
averaged over its time bins; each region is one band.

How features meet scans. Each row goes to the assembly scan of the same film
and scan number. Per run, ``stimulus_window`` scans are stacked and shifted by
``hrf_delay`` scans; the first ``excluded_samples_start`` and the last
``excluded_samples_end`` scans are left out.

The fit and the score. 5 folds over stimuli (half-episodes and Movie10
segments; all runs of one stimulus stay on one side). Per fold, each band is
centred, reduced to at most ``LAYER_DIM_CAP`` components and scaled on the
training stimuli only; a banded ridge with one penalty tuple for all parcels,
chosen on held-out training stimuli, predicts the test stimuli (``scoring.py``). The score is the raw
median per-parcel Pearson r. No ceiling is measured.
"""
import logging
import math
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.execution_plan import ExecutionPlan
from brainscore_core.metrics import Score
from brainscore import load_dataset, load_stimulus_set
from brainscore.data.algonauts2025 import (
    CLIP_FRAMES,
    SOUND_RATE_HZ,
    TR_SEC,
    assembly_identifier,
    default_data_root,
    per_scan_stimuli,
    probe_stimuli,
    scans_per_stimulus,
    stimulus_set_identifier,
)
from brainscore.metrics.encoding import summarize

from brainscore.benchmarks._score import benchmark_score, not_estimated
from .scoring import PENALTY_PROTOCOL, PROJECTION_PROTOCOL, score_banded

logger = logging.getLogger(__name__)

BIBTEX = """@article{gifford2025algonauts,
  title={The Algonauts Project 2025 Challenge: How the Human Brain Makes Sense of Multimodal Movies},
  author={Gifford, Alessandro T and Bersch, Domenic and St-Laurent, Marie and Pinsard, Basile
          and Boyle, Julie and Bellec, Pierre and Oliva, Aude and Roig, Gemma and Cichy, Radoslaw M},
  journal={arXiv preprint arXiv:2501.00504},
  year={2025},
}"""


def _as_features(assembly):
    """``(values (n_presentations, n_units) float32, neuroid ids or None)``."""
    if 'time_bin' in assembly.dims:
        assembly = assembly.mean(dim='time_bin')
    feats = assembly.values.astype(np.float32)
    try:
        feature_ids = assembly['neuroid_id'].values
    except KeyError:
        feature_ids = None
    return feats, feature_ids


def _features_in_order(response, stimuli):
    """``_as_features`` of ``response`` with its rows in the order of ``stimuli``.

    The response must answer each row of ``stimuli`` exactly once; its rows
    are matched by ``stimulus_id``.
    """
    from brainscore.presentation import response_stimulus_ids
    wanted = [str(s) for s in stimuli['stimulus_id'].values]
    answered = [str(s) for s in response_stimulus_ids(response, wanted)]
    if answered != wanted:
        position = {sid: index for index, sid in enumerate(answered)}
        response = response.isel(presentation=[position[sid] for sid in wanted])
    return _as_features(response)


class Algonauts2025(BenchmarkBase):
    """Algonauts 2025 encoding benchmark for one participant and one split."""

    # One penalty per decade from 1e-2 to 1e8; the bands are standardized.
    BANDED_ALPHA_GRID = tuple(float(10.0 ** e) for e in range(-2, 9))
    # Components kept for each band, and penalty tuples tried by the random search
    # when the full grid (11 ** number of bands: 11 penalties per band) is larger.
    LAYER_DIM_CAP = 100
    BANDED_CANDIDATES = 60

    # The benchmark stacks the TRs itself, so the subject keeps no history across TRs.
    _IMAGE = frozenset({'vision:image'})
    _VIDEO = frozenset({'vision:video'})
    _VIDEO_AUDIO = frozenset({'vision:video', 'audio'})
    # The regions the benchmark names for each choice of channels. Each is one band.
    CHANNEL_REGIONS = {
        _IMAGE: ('IT', 'V1', 'V4'),
        _VIDEO: ('IT', 'V1', 'V4'),
        _VIDEO_AUDIO: ('A1', 'IT', 'V1', 'V4'),
    }
    _CHANNEL_SUFFIX = {_IMAGE: '', _VIDEO: '-video', _VIDEO_AUDIO: '-video-audio'}
    # The prepared files of one row (see brainscore.data.algonauts2025.MEDIA).
    _CHANNEL_MEDIA = {_IMAGE: ['frames'], _VIDEO: ['clips'], _VIDEO_AUDIO: ['clips', 'sound']}
    # The identifier ends in a suffix for the split.
    _SPLIT_SUFFIX = {'friends': '', 'friends_s7': '-test', 'ood': '-ood'}
    required_input_channels = _IMAGE
    requested_output_channels = frozenset(f'neural:{r}' for r in CHANNEL_REGIONS[_IMAGE])
    trial_memory = 'none'
    _N_FOLDS = 5   # 5-fold scoring over stimuli

    def __init__(
        self,
        participant: int,
        split: str = 'friends',
        stimulus_window: int = 5,
        hrf_delay: int = 3,
        excluded_samples_start: Optional[int] = None,
        excluded_samples_end: int = 5,
        assembly_root: Optional[Path] = None,
        channels: Sequence[str] = ('vision:image',),
        streaming: bool = False,
        movies: Optional[Sequence[str]] = None,
    ):
        if participant not in (1, 2, 3, 5):
            raise ValueError(
                f"Algonauts participants are {{1, 2, 3, 5}}; got {participant}.")
        if split not in self._SPLIT_SUFFIX:
            raise ValueError(
                f"split must be one of {tuple(self._SPLIT_SUFFIX)}; got {split!r}.")
        channels = frozenset(channels)
        if channels not in self.CHANNEL_REGIONS:
            raise ValueError(
                f"channels must be one of "
                f"{[sorted(c) for c in self.CHANNEL_REGIONS]}; got {sorted(channels)}.")
        # A kept sample needs every lag of its window inside its run.
        first_full = hrf_delay + stimulus_window - 1
        if excluded_samples_start is None:
            excluded_samples_start = first_full
        if excluded_samples_start < first_full:
            raise ValueError(
                f"excluded_samples_start={excluded_samples_start} keeps samples whose "
                f"window starts before the run; use at least hrf_delay + stimulus_window "
                f"- 1 = {first_full}.")
        if movies is not None and (isinstance(movies, str) or not len(movies)):
            raise ValueError(
                "movies must be a non-empty list of stimulus ids, for example "
                "['friends_s01e01a', 'friends_s01e01b'].")

        self._channels = channels
        self.required_input_channels = channels
        self.requested_output_channels = frozenset(
            f'neural:{region}' for region in self.CHANNEL_REGIONS[channels])
        self._streaming = bool(streaming)
        self._movies = None if movies is None else [str(movie) for movie in movies]
        self._participant = participant
        self._split = split
        self._stimulus_window = stimulus_window
        self._hrf_delay = hrf_delay
        self._excluded_samples_start = excluded_samples_start
        self._excluded_samples_end = excluded_samples_end
        self._assembly_root = (Path(assembly_root)
                               if assembly_root else default_data_root())
        self._assembly = None
        self._stimulus_set = None

        super().__init__(
            identifier=(f'Algonauts2025-{split}-sub{participant:02d}'
                        f'{self._SPLIT_SUFFIX[split]}{self._CHANNEL_SUFFIX[channels]}'),
            version=2,
            parent='naturalistic',
            ceiling=not_estimated(),   # the score is the raw correlation
            bibtex=BIBTEX,
        )

    @property
    def settings(self) -> MappingProxyType:
        """What a held-out prediction shares with its training split, read only."""
        return MappingProxyType(dict(participant=self._participant, split=self._split, channels=self._channels,
                                     streaming=self._streaming, stimulus_window=self._stimulus_window,
                                     hrf_delay=self._hrf_delay))

    @property
    def assembly(self):
        if self._assembly is None:
            self._assembly = load_dataset(
                assembly_identifier(self._split, self._participant),
                root=self._assembly_root,
            )
            if self._movies is not None:
                self._assembly = self._only_movies(self._assembly)
        return self._assembly

    def _only_movies(self, assembly):
        """The TRs of ``assembly`` that belong to ``self._movies``.

        Raises ``ValueError`` when a named movie is not in the assembly, or when
        the movies are too few for 5-fold scoring over stimuli with a 5-fold
        choice of penalties inside every training fold.
        """
        stim = np.asarray(assembly['stimulus_id'].values).astype(str)
        unknown = sorted(set(self._movies) - set(stim))
        if unknown:
            raise ValueError(
                f"movies {unknown} are not in the assembly. Use stimulus ids such as "
                f"{sorted(set(stim))[:4]}.")
        keep = np.isin(stim, self._movies)
        n_stimuli = len(set(stim[keep]))
        # A fold trains on n - ceil(n / 5) stimuli and splits them 5 ways again.
        needed = self._N_FOLDS
        while needed - math.ceil(needed / self._N_FOLDS) < self._N_FOLDS:
            needed += 1
        # Only the training split is cross-validated; a held-out split needs no split.
        if self._split == 'friends' and n_stimuli < needed:
            raise ValueError(
                f"movies give {n_stimuli} stimuli; the benchmark needs at least "
                f"{needed} for 5-fold scoring over stimuli. Name more movies.")
        return assembly.isel(presentation=np.flatnonzero(keep))

    @property
    def stimulus_set(self):
        if self._stimulus_set is None:
            self._stimulus_set = load_stimulus_set(
                stimulus_set_identifier(self._split),
                root=self._assembly_root,
            )
        return self._stimulus_set

    # ── What is shown ─────────────────────────────────────────────

    def _stimuli(self):
        """The table of what one scan presents: one row per (film, scan), from
        the files the data plugin prepared."""
        return per_scan_stimuli(self.stimulus_set, scans_per_stimulus(self.assembly),
                                self._CHANNEL_MEDIA[self._channels], self._split,
                                self._participant, self._assembly_root)

    # ── What is requested ─────────────────────────────────────────

    def _record_bands(self, subject, stimuli, regions) -> Dict[str, Tuple[np.ndarray, object]]:
        """``{region: (features, neuroid ids)}`` with one row per row of ``stimuli``.

        One session requests every region on its own channel ``neural:<region>``,
        so the subject sees each row once and the request is what pre-flight
        checked. Each answer is averaged over its time bins and put in the row
        order of ``stimuli``. Nothing is reduced here: ``score_banded`` fits the
        reduction per fold.
        """
        from brainscore_core.streaming_helpers import stimulus_session
        channels = [f'neural:{region}' for region in regions]
        session = stimulus_session(stimuli, channels, None,
                                   time_bins=[(0, int(TR_SEC * 1000))],
                                   streaming=self._streaming)
        subject.participate(session)
        bands = {}
        for region, channel in zip(regions, channels):
            features, feature_ids = _features_in_order(session.collect(channel), stimuli)
            if features.shape[0] != len(stimuli):
                raise ValueError(
                    f"the stimuli have {len(stimuli)} rows but {channel} answered "
                    f"{features.shape[0]}; cannot trust input order.")
            bands[region] = (features, feature_ids)
        return bands

    def _assembly_bands(self, subject, regions):
        """The bands of ``_record_bands``, aligned to the TR rows of the assembly."""
        table = self._stimuli()
        logger.debug('recording %s for %d rows (%s)', ['neural:' + region for region in regions],
                     len(table), 'stream' if self._streaming else 'table')
        frame_ids = list(table['stimulus_id'])
        return {region: (self._align_features_to_assembly(features, frame_ids, table), ids)
                for region, (features, ids) in self._record_bands(subject, table, regions).items()}

    # ── How features meet scans ───────────────────────────────────

    def _align_features_to_assembly(
        self, features: np.ndarray, frame_ids: List[str],
        frame_stim_set,
    ) -> np.ndarray:
        """The per-row features on the assembly scans: row (film, scan) goes to the
        scan with the same ``stimulus_id`` and ``t_within_run``. Returns X with
        shape (n_TRs_in_assembly, n_features)."""
        feat_idx = {fid: i for i, fid in enumerate(frame_ids)}
        # The table's 'stimulus_id' is the unique row id; the film is 'clip_id'.
        sf = frame_stim_set
        stim_t_to_frame_id = {(str(clip), int(t)): frame for clip, t, frame in
                              zip(sf['clip_id'], sf['t_within_run'], sf['frame_id'])}

        n_TRs = self.assembly.sizes['presentation']
        X = np.full((n_TRs, features.shape[1]), np.nan, dtype=np.float32)
        a_stim = list(self.assembly['stimulus_id'].values)
        a_t = list(self.assembly['t_within_run'].values)
        missing_fid = missing_row = 0
        for i in range(n_TRs):
            fid = stim_t_to_frame_id.get((str(a_stim[i]), int(a_t[i])))
            if fid is None:
                missing_fid += 1
                continue
            row = feat_idx.get(fid)
            if row is None:
                missing_row += 1
                continue
            X[i] = features[row]
        missing = missing_fid + missing_row
        if missing > 0:
            # Fill a scan without a row from the nearest earlier scan of the same
            # run (the first valid one at a run start), never across runs.
            logger.warning('%d of %d TRs had no frame (%d not in the stimulus table, %d not '
                           'recorded); filling each from the same run.',
                           missing, n_TRs, missing_fid, missing_row)
            run_idx, _ = self._runs_and_keep(drop_excluded=False)
            for run in np.unique(run_idx):
                rows = np.flatnonzero(run_idx == run)
                valid = rows[~np.isnan(X[rows, 0])]
                if not len(valid):
                    raise ValueError(f'Run {run} of the assembly has no TR with features.')
                last = X[valid[0]].copy()
                for i in rows:
                    if np.isnan(X[i, 0]):
                        X[i] = last
                    else:
                        last = X[i]
        return X

    def _apply_stimulus_window_and_hrf(
        self, X: np.ndarray, run_idx_per_obs: np.ndarray,
    ) -> np.ndarray:
        """Stack stimulus_window TRs of context per fMRI sample, then
        shift the whole thing by hrf_delay so feature(t) predicts
        BOLD(t + hrf_delay).

        run_idx_per_obs is used to ensure stacking doesn't cross run
        boundaries (i.e., sample t in run R only stacks with samples
        from the same run).
        """
        n_TRs, n_feat = X.shape
        W = self._stimulus_window
        D = self._hrf_delay
        # Output: (n_TRs, W * n_feat), the concat of W stacked feature vectors,
        # the leftmost W-1 TRs in the past. Sample t gets features at
        # t-D-(W-1), ..., t-D from its own run; a lag before the run start is zero.
        X_stacked = np.zeros((n_TRs, W * n_feat), dtype=X.dtype)
        rows = np.arange(n_TRs)
        for offset in range(W):
            src = rows - (D + (W - 1 - offset))
            inside = src >= 0
            inside[inside] = run_idx_per_obs[src[inside]] == run_idx_per_obs[rows[inside]]
            X_stacked[rows[inside], offset * n_feat:(offset + 1) * n_feat] = X[src[inside]]
        return X_stacked

    def _runs_and_keep(self, drop_excluded: bool = True):
        """``(run index per TR, keep mask)`` over the assembly's TR rows.

        Raises when the rows of a run are not in ``t_within_run`` order.

        A run is one unique ``(stimulus_id, run)`` block, numbered in order of
        first appearance. Held-out stubs (S7, OOD) carry no ``run`` coordinate, so
        each clip is its own run there. ``keep`` is False for the first
        ``excluded_samples_start`` and the last ``excluded_samples_end`` TRs of
        each run when ``drop_excluded`` is True.
        """
        a_stim = list(self.assembly['stimulus_id'].values)
        a_run = (list(self.assembly['run'].values)
                 if 'run' in self.assembly.coords else a_stim)
        run_idx = np.empty(len(a_stim), dtype=np.int64)
        seen: Dict[Tuple[str, str], int] = {}
        for i, (s, r) in enumerate(zip(a_stim, a_run)):
            run_idx[i] = seen.setdefault((str(s), str(r)), len(seen))
        # The stack reads earlier rows as earlier scans: each run's rows must be
        # in t_within_run order.
        steps = np.asarray(self.assembly['t_within_run'].values)
        for ri in range(len(seen)):
            run_steps = steps[run_idx == ri]
            if np.any(np.diff(run_steps.astype(np.int64)) <= 0):
                raise ValueError(
                    f'The TRs of run {list(seen)[ri]} are not in t_within_run order.')
        keep = np.ones(len(run_idx), dtype=bool)
        if drop_excluded:
            for ri in range(len(seen)):
                run_idxs = np.where(run_idx == ri)[0]
                keep[run_idxs[:self._excluded_samples_start]] = False
                if self._excluded_samples_end > 0:
                    keep[run_idxs[-self._excluded_samples_end:]] = False
        return run_idx, keep

    def _stacker(self, run_idx):
        """Stack the stimulus window and shift by the HRF delay within ``run_idx``'s runs."""
        return lambda X: self._apply_stimulus_window_and_hrf(X, run_idx)

    # ── The fit and the score ─────────────────────────────────────

    def _stimulus_groups(self) -> np.ndarray:
        """The stimulus of each TR row: the unit that the folds keep on one side.

        Movie10 repeats some clips in a second run with the same ``stimulus_id``;
        both runs of a clip land in the same fold.
        """
        return np.asarray(self.assembly['stimulus_id'].values).astype(str)

    def regression_inputs(self, subject, drop_excluded: bool = True):
        """What the banded ridge needs from this benchmark and ``subject``.

        Returns ``(bands, Y, groups, keep, stack)``: the recorded bands aligned to
        the assembly scans, the parcel responses, the film of each scan (the fold
        groups), the scans kept after the exclusions at the start and the end of
        each run (every scan when ``drop_excluded`` is False), and the stack of
        the stimulus window and the HRF delay inside each run. The score uses
        it, and so does ``brainscore.tools.algonauts2025_predictions``.
        """
        regions = list(self.CHANNEL_REGIONS[self._channels])
        bands = self._assembly_bands(subject, regions)
        run_idx, keep = self._runs_and_keep(drop_excluded)
        return bands, self.assembly.values, self._stimulus_groups(), keep, self._stacker(run_idx)

    def __call__(self, subject) -> Score:
        if self._split != 'friends':
            raise ValueError(
                f"The {self._split!r} benchmark is a prediction target, not a score: its "
                f"fMRI is withheld. Call `brainscore.tools.algonauts2025_predictions."
                f"generate_predictions(subject, benchmark, out_dir)` to write the predictions. "
                f"For a score, use `Algonauts2025-friends-sub{self._participant:02d}`.")
        # The same pre-flight as score(): N/A before any recording.
        from brainscore_core.compatibility import check_compatibility
        check_compatibility(subject, self)
        bands, Y, groups, keep, stack = self.regression_inputs(subject)
        logger.debug('%d stimuli; %d of %d TRs after exclusions; banded ridge over bands %s '
                     '(5 folds over stimuli)', len(np.unique(groups)), int(keep.sum()),
                     len(keep), list(bands))
        per_parcel_r, info = score_banded(
            bands, Y, groups, keep, stack,
            dim_cap=self.LAYER_DIM_CAP, alpha_grid=self.BANDED_ALPHA_GRID,
            n_candidates=self.BANDED_CANDIDATES)
        finite, median_r, mean_r = summarize(per_parcel_r)
        # No ceiling is measured: the score is the raw median correlation.
        score = benchmark_score(
            self, median_r, raw=median_r, normalized=False,
            metric='median_per_parcel_pearson_r',
            mean_r=mean_r,
            n_parcels_scored=int(len(finite)),
            # The per-parcel scores in Schaefer order, for a cortical surface. A
            # parcel that could not be scored stays NaN, unpainted.
            per_parcel_r=per_parcel_r,
            n_TRs=int(keep.sum()),
            stimulus_window=self._stimulus_window,
            hrf_delay=self._hrf_delay,
            excluded_samples_start=self._excluded_samples_start,
            excluded_samples_end=self._excluded_samples_end,
            cv_groups='stimulus_id',
            input_channels=sorted(self._channels),
            presentation='stream' if self._streaming else 'table',
            bands=info['bands'],
            band_channels={band: f'neural:{band}' for band in info['bands']},
            components_per_band=info['components_per_band'],
            alpha_grid=info['alpha_grid'],
            alpha_candidates=info['alpha_candidates'],
            penalty_protocol=PENALTY_PROTOCOL,
            penalties_per_fold=info['penalties_per_fold'],
            feature_projection_protocol=PROJECTION_PROTOCOL,
        )
        if self._channels != self._IMAGE:
            score.attrs['clip_seconds'] = TR_SEC
            score.attrs['frames_per_clip'] = CLIP_FRAMES
        if self._channels == self._VIDEO_AUDIO:
            score.attrs['sound_rate_hz'] = SOUND_RATE_HZ
        if self._movies is not None:
            score.attrs['movies'] = list(self._movies)
        return score

    @property
    def execution_plan(self) -> ExecutionPlan:
        """The memory plan: one extracted row per TR; the metric rows are the TRs
        after the exclusions of each run; the metric width is one reduced band per
        region, stacked over ``stimulus_window``. The pre-flight probes the raw
        width with one blank frame or clip. No ceiling metric runs."""
        n_trs = len(self.assembly['stimulus_id'])
        try:
            stim = self.assembly['stimulus_id'].values
            run = self.assembly['run'].values
            run_lengths = Counter((str(s), str(r)) for s, r in zip(stim, run))
            excluded = self._excluded_samples_start + self._excluded_samples_end
            # clamp per run: a run shorter than the excluded span retains 0, not a
            # negative count (subtracting excluded*n_runs globally would under-count).
            metric_obs = max(sum(max(L - excluded, 0)
                                 for L in run_lengths.values()), 1)
        except Exception:
            metric_obs = int(n_trs)  # no run metadata (e.g. a stub): safe over-count
        return ExecutionPlan(
            n_extraction_presentations=int(n_trs),
            metric_observations=metric_obs,
            metric_feature_width=(len(self.CHANNEL_REGIONS[self._channels])
                                  * self._stimulus_window * self.LAYER_DIM_CAP),
            metric_category='ridge',
            runs_ceiling_metric=False,
            probe_stimuli=probe_stimuli(self._CHANNEL_MEDIA[self._channels][0],
                                        self._assembly_root, self.identifier),
        )
