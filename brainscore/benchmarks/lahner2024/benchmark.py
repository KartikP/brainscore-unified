"""
Lahner 2024 BOLDMoments fMRI naturalistic benchmark (unified interface).

Ten participants watched short (3s) video clips while being scanned with fMRI;
a GLM gives one BOLD estimate per (video, participant, voxel). The assembly
holds 1026 of the 1102 BOLDMoments videos and has shape:

    (time_bin=1, neuroid=20484, presentation=10260)

where ``time_bin`` spans the full 3-second clip (single pre-computed
estimate; no time-series regression required), ``neuroid`` is cortical
vertices across L/R hemispheres, and ``presentation`` is 1026 videos × 10
participants. The coordinate that tells the participants apart is named
``repetition``: its 10 values are the 10 participants, not repeated showings
to one person. The benchmark averages the participants to get one response per video.

The benchmark follows the protocol of the data set. BOLDMoments has 1000
training-set videos, each shown 3 times to a participant, and 102 test-set
videos (id above 1000; ``boldmoments_test_set``), each shown 10 times, so the
test-set responses are much less noisy. The hosted assembly holds 933 of the
training-set videos and 93 of the test-set videos. Every variant presents all
of them, fits one regression on the training-set videos (no cross-validation)
and computes the correlation of each voxel on the test-set videos only; no
test-set video is in the fit.

The reliability of a voxel is the agreement between two halves of the 10
participants (``_split_half_reliability``). The voxel mask of the ``-visualROI``
variants is the set of voxels with a reliability of at least the threshold on
the training-set videos, the videos that the fit sees; no test-set response
reaches the selection. The per-voxel ceilings are sqrt(reliability) on the
scored videos, the test-set videos (``reliability_on_test_set``): the ceiling is
the reliability of the data that is predicted, it is not a parameter of the
fit, and the voxels were selected on other videos. Both ``-visualROI`` variants
divide by these ceilings; whole cortex, ``-multimodal`` and
``-multimodal-auditoryROI`` report the raw correlation.

The Lahner2024 family has two visual-ROI voxel sets. This class selects on the
training-set videos (3958 voxels on the hosted data). The time-resolved class
(``Lahner2024BOLDMoments_timeresolved``) selects on the test-set videos (7209
voxels), with the same threshold. A ``-visualROI`` score of one class and of the
other covers different voxels.

Taken from brain-score/vision PR #1249 (YingtianDt):
    - BOLDMoments as the stimulus-set identifier
    - Citation

Scoring pipeline:
    1. Record ``neural:IT`` through ``participate`` (``neural_response``): a
       subject that declares ``vision:video`` sees whole clips; any other
       subject sees N frames sampled from each clip, pooled per clip
       (``temporal_bin``). With the ``audio`` channel, ``neural:A1`` is also
       recorded for the sound track of each clip, and the two are joined
       column-wise. A subject that takes one stream is shown that stream and
       answers both regions from it (its ``output_inputs`` permitting).
    2. Average the response over its time bins → one vector per video.
    3. Average the neural assembly across the 10 participants → one BOLD
       vector per video per voxel; keep the voxels of the voxel mask.
    4. One ridge fit on the training-set videos (penalty 1,
       ``brainscore.metrics.encoding``) → predictions for the test-set videos
       → Pearson per voxel over the test-set videos → median over the voxels
       that count (``voxel_score``).

## Versions

- version 2 (current): the score keeps one Pearson r per scored voxel, in
  voxel order (``per_voxel_r``, NaN where a voxel has none). A voxel counts
  when its r is finite and, with a measured ceiling, its ceiling is finite
  and above zero; the other voxels are left out of every median and mean and
  counted in ``n_voxels_excluded``. The time-resolved benchmark uses the same
  rule.
- version 1: the NaN entries were removed first and the shorter vector was
  then divided by the ceiling of every voxel. With one NaN voxel this failed
  (the lengths differ) or divided one value by every ceiling, and the
  ``per_voxel_r`` attribute lost the voxel positions. Removed: when no voxel
  is NaN, version 2 gives the same value.

## IMPORTANT: what the frame route does NOT measure

For a stills-only subject the benchmark is **frame-aggregated**: for each
3-second video the subject sees N static frames (default 3: at 0.5s / 1.5s /
2.5s), each as an independent image, and the N responses are **mean-pooled**
before regression. The subject does NOT see motion, temporal ordering or
event dynamics; two videos that differ only in temporal order give the same
features. A subject that declares ``vision:video`` sees the whole clip.

The score attrs include ``pipeline`` (``frame_aggregation`` or
``video_native``) and ``input_adaptation`` so the route is explicit on every
one-stream result.
"""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score, per_unit_pearson
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuronRecordingAssembly,
)
from brainscore import load_dataset, load_stimulus_set
from brainscore.benchmarks._score import benchmark_score, not_estimated
from brainscore.metrics.encoding import SklearnRidge, cross_validated_r, ridge
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


BIBTEX = """@article{lahner2024modeling,
  title={Modeling short visual events through the BOLD moments video fMRI dataset and metadata},
  author={Lahner, Benjamin and Dwivedi, Kshitij and Iamshchinina, Polina and
          Graumann, Monika and Lascelles, Alex and Roig, Gemma and
          Gifford, Alessandro T and Pan, Bowen and Jin, Sunny-Yu and
          Ratan Murty, N Apurva and others},
  journal={Nature Communications},
  volume={15},
  pages={6241},
  year={2024},
}"""

_VISUAL = ('vision:image', 'vision:video')


def _ceiling_measured(benchmark) -> bool:
    """Whether a Lahner2024 benchmark normalizes by a measured ceiling.

    A benchmark built without a ceiling holds ``not_estimated()`` (NaN) and
    reports the raw correlation; only one given a measured ceiling normalizes.
    """
    return not np.isnan(float(benchmark.ceiling))


def boldmoments_test_set(stimulus_ids) -> List[str]:
    """The BOLDMoments test-set videos among ``stimulus_ids``: id above 1000
    (``stimulus_1001`` to ``stimulus_1102``), sorted, without repeats.

    The one split rule of the Lahner2024 benchmarks. The GLM benchmark follows the
    protocol of the data set with it: one fit on the other videos (the training
    set), the correlation and the ceilings on the test-set videos, and the voxel
    mask from the training-set videos. The time-resolved benchmark uses it for its
    voxel mask and its presented subset.
    """
    return sorted({str(stimulus_id) for stimulus_id in stimulus_ids
                   if int(str(stimulus_id).rsplit('_', 1)[1]) > 1000})


def _split_half_reliability(assembly, stimuli: Sequence[str], n_splits: int = 20,
                            random_state: int = 0) -> np.ndarray:
    """Per-voxel split-half reliability of the GLM-beta assembly across participants.

    The ``repetition`` entries of a video are the participants. For each random
    split of each video's participants into two halves, average within each half
    to get (n_videos, n_voxels), then Pearson-correlate the two halves across
    videos for each voxel. Average across ``n_splits`` random splits and apply
    the Spearman-Brown correction.

    ``stimuli`` names the videos to compute on; there is no default, so that no
    caller computes on every video of the assembly by accident.

    Returns a (n_voxels,) array. High-reliability voxels are stimulus-driven
    (signal); low-reliability voxels are noise-dominated.
    """
    a = assembly.squeeze('time_bin', drop=True)
    stim_ids = np.asarray(a['stimulus_id'].values)
    keep = np.isin(stim_ids.astype(str), np.asarray(list(stimuli), dtype=str))
    if not keep.any():
        raise ValueError('No presentation of the assembly shows one of the given stimuli.')
    a = a.isel(presentation=keep)
    stim_ids = stim_ids[keep]
    data = a.values  # (n_voxels, n_presentations)
    n_voxels = data.shape[0]
    unique_stims = sorted(set(stim_ids))

    # Organize: per-stim, list of (rep, data_col_idx)
    stim_to_cols: dict = {s: [] for s in unique_stims}
    for i in range(len(stim_ids)):
        stim_to_cols[stim_ids[i]].append(i)

    rng = np.random.default_rng(random_state)
    reliability_sum = np.zeros(n_voxels)
    for _ in range(n_splits):
        half_a_cols = []
        half_b_cols = []
        for s in unique_stims:
            cols = stim_to_cols[s]
            shuffled = rng.permutation(cols)
            mid = len(shuffled) // 2
            half_a_cols.append(shuffled[:mid])
            half_b_cols.append(shuffled[mid:mid * 2])

        def _mean_of_cols(cols_per_stim):
            out = np.zeros((len(unique_stims), n_voxels))
            for si, cols in enumerate(cols_per_stim):
                out[si] = data[:, cols].mean(axis=1)
            return out
        A = _mean_of_cols(half_a_cols)
        B = _mean_of_cols(half_b_cols)
        # Per-voxel Pearson across stimuli; a constant voxel counts as 0.
        reliability_sum += np.nan_to_num(per_unit_pearson(A, B), nan=0.0)
    r_half = reliability_sum / n_splits
    # Spearman-Brown: full-set reliability from half-set reliability
    return 2 * r_half / (1 + r_half)


def reliability_on_test_set(assembly) -> np.ndarray:
    """``_split_half_reliability`` on the test-set videos of the GLM assembly. The GLM
    benchmark takes its per-voxel ceilings from it (the test-set videos are the scored
    ones). The time-resolved benchmark selects its voxel mask with it (7209 voxels at
    the threshold 0.3 on the hosted data), so its visual-ROI voxels differ from the
    GLM ones (selected on the training-set videos, 3958 voxels)."""
    test = boldmoments_test_set(assembly['stimulus_id'].values)
    if not test:
        raise ValueError('The assembly holds no test-set video (id above 1000): '
                         'no voxel mask or ceiling can be measured.')
    return _split_half_reliability(assembly, stimuli=test)


def presented_stimuli(stimuli, stimulus_ids):
    """The rows of ``stimuli`` named in ``stimulus_ids``, in table order, with an
    identifier that tells the set apart: the activation store keys a response by
    the identifier. Every row: ``stimuli`` itself; every row outside the test set:
    ``<identifier>-train``; any other subset: ``<identifier>-<digest of its ids>``.
    Used by the time-resolved benchmark; the GLM benchmark presents every row."""
    import hashlib
    wanted = {str(stimulus_id) for stimulus_id in stimulus_ids}
    ids = stimuli['stimulus_id'].astype(str)
    kept = stimuli[ids.isin(wanted).values]
    if len(kept) == len(stimuli):
        return stimuli
    train = set(ids) - set(boldmoments_test_set(ids))
    kept_ids = set(kept['stimulus_id'].astype(str))
    kept.identifier = (f'{stimuli.identifier}-train' if kept_ids == train else
                       f'{stimuli.identifier}-' + hashlib.sha256(
                           '|'.join(sorted(kept_ids)).encode()).hexdigest()[:10])
    kept.stimulus_paths = {stimulus_id: stimuli.get_stimulus(stimulus_id)
                           for stimulus_id in kept['stimulus_id']}
    return kept


def _select_voxels(reliability_threshold: Optional[float], voxel_mask_fn: Optional[Callable],
                   reliability: Callable[[], np.ndarray]) -> Optional[np.ndarray]:
    """The voxels a Lahner2024 benchmark scores: every voxel (``None``), the voxels
    whose ``reliability()`` is at least ``reliability_threshold``, or the mask
    ``voxel_mask_fn()`` returns (for example ``build_auditory_mask``)."""
    if voxel_mask_fn is not None:
        return np.asarray(voxel_mask_fn(), dtype=bool)
    if reliability_threshold is not None:
        return reliability() >= reliability_threshold
    return None


def voxel_score(benchmark, per_voxel_r, ceiling_per_voxel: Optional[np.ndarray] = None) -> Score:
    """The score of a Lahner2024 benchmark from the held-out r of each scored voxel.

    ``per_voxel_r`` holds one value per voxel of ``benchmark.voxel_mask`` (every
    voxel without a mask), in voxel order. One rule for every variant: a voxel
    counts when its r is finite and, given ``ceiling_per_voxel``, its ceiling is
    finite and above zero. A voxel with a constant response has no correlation;
    a voxel with a reliability at or below zero has no ceiling. Such a voxel is
    left out of the median and the mean and counted in ``n_voxels_excluded``,
    as the Algonauts2025 benchmark leaves out a constant parcel. The attribute
    ``per_voxel_r`` keeps every scored voxel (NaN where r is undefined), so
    ``voxel_mask_indices`` places it back on the surface.

    With ``ceiling_per_voxel`` the score is ``median(r_i / c_i)``, not
    ``median(r_i) / median(c_i)``: the repository convention. Without it the
    raw median is divided by ``benchmark.ceiling`` when it is measured.
    """
    r = np.asarray(per_voxel_r)
    counts = np.isfinite(r)
    if ceiling_per_voxel is not None:
        ceiling_per_voxel = np.asarray(ceiling_per_voxel, dtype=float)
        counts &= np.isfinite(ceiling_per_voxel) & (ceiling_per_voxel > 0)
    kept = r[counts]
    median_r = float(np.median(kept))
    if ceiling_per_voxel is not None:
        score = benchmark_score(
            benchmark, value=float(np.median(np.asarray(kept, dtype=float) / ceiling_per_voxel[counts])),
            raw=median_r, metric='median_per_voxel_pearson_r',
            ceiling=Score(float(np.median(ceiling_per_voxel[counts]))),
            ceiling_aggregation='per_voxel')
    else:
        score = benchmark_score(
            benchmark, raw=median_r, metric='median_per_voxel_pearson_r',
            ceiling_aggregation='scalar')
    score.attrs['mean_r'] = float(np.mean(kept))
    score.attrs['n_voxels_scored'] = int(counts.sum())
    score.attrs['n_voxels_excluded'] = int(r.size - counts.sum())
    score.attrs['per_voxel_r'] = r   # exposed for bootstrap CIs and surface maps
    mask = benchmark.voxel_mask
    if mask is not None:
        score.attrs['voxel_mask_n_total'] = int(mask.size)
        score.attrs['voxel_mask_n_kept'] = int(mask.sum())
        score.attrs['voxel_mask_indices'] = np.flatnonzero(mask)
        if benchmark._reliability_threshold is not None:
            score.attrs['reliability_threshold'] = float(benchmark._reliability_threshold)
        if benchmark._voxel_mask_fn is not None:
            score.attrs['voxel_mask_source'] = benchmark._voxel_mask_fn.__name__
    return score


class Lahner2024BOLDMoments(BenchmarkBase):
    """Naturalistic fMRI benchmark: predict per-video BOLD responses (GLM betas).

    Args:
        channels: what the benchmark sends. ``('vision:image', 'vision:video')``
            (default) records ``neural:IT``; adding ``'audio'`` also sends the
            sound track of each clip and records ``neural:A1``. Both regions are
            always requested; a subject that takes one of the two streams is shown
            that stream alone (a partial match) and answers both regions from it,
            as its ``output_inputs`` allow. ``attrs['scored_inputs']`` names the
            inputs shown.
        sample_times_ms: the frames a stills-only subject sees from each clip.
        ceiling: a measured noise ceiling; the score is then normalized per voxel
            by sqrt(reliability) of each voxel on the scored test-set videos.
        reliability_threshold: score only the voxels whose split-half
            reliability on the training-set videos is at least this value.
            Every variant fits on the training-set videos and scores the
            test-set videos (see the module docstring). The time-resolved
            class selects on the test-set videos instead.
        voxel_mask_fn: score only the voxels of this mask (for example
            ``build_auditory_mask``). Give at most one of the two.
        identifier: the benchmark identifier (required; the registry key of the variant).
    """

    trial_memory = 'none'   # each clip is an independent trial

    def __init__(
        self,
        channels: Sequence[str] = _VISUAL,
        sample_times_ms: List[float] = list(DEFAULT_SAMPLE_TIMES_MS),
        ceiling: Optional[float] = None,
        reliability_threshold: Optional[float] = None,
        voxel_mask_fn: Optional[Callable[[], np.ndarray]] = None,
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
        self._sample_times_ms = list(sample_times_ms)
        self._reliability_threshold = reliability_threshold
        self._voxel_mask_fn = voxel_mask_fn
        self._assembly: Optional[NeuronRecordingAssembly] = None
        self._stimulus_set = None
        self._mask_rel: Optional[np.ndarray] = None
        self._ceiling_rel: Optional[np.ndarray] = None
        self._voxel_mask: Optional[np.ndarray] = None
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
            self._assembly = load_dataset('Lahner2024-fMRI')
        return self._assembly

    @property
    def stimulus_set(self):
        if self._stimulus_set is None:
            self._stimulus_set = load_stimulus_set('BOLDMoments')
        return self._stimulus_set

    def _mask_reliability(self) -> np.ndarray:
        """The reliability that selects the voxels: ``_split_half_reliability`` on the
        training-set videos of the assembly (the videos the fit sees), computed once."""
        if self._mask_rel is None:
            self._mask_rel = _split_half_reliability(self.assembly, stimuli=self._split()[0])
        return self._mask_rel

    def _ceiling_reliability(self) -> np.ndarray:
        """The reliability that gives the ceilings: ``reliability_on_test_set``, the
        reliability of the scored videos, computed once. It is not a parameter of the
        fit, and the voxels were selected on the training-set videos."""
        if self._ceiling_rel is None:
            self._ceiling_rel = reliability_on_test_set(self.assembly)
        return self._ceiling_rel

    def _split(self) -> Tuple[List[str], List[str]]:
        """``(training-set ids, test-set ids)`` of the assembly. The presented table
        and the assembly must hold the same videos (a video of the assembly that is not
        presented would enter the mask and the ceilings but not the fit or the score),
        and both sets must be present; otherwise this raises, before any subject runs."""
        ids = {str(i) for i in self.assembly['stimulus_id'].values}
        table = {str(i) for i in self.stimulus_set['stimulus_id']}
        no_bold, not_presented = sorted(table - ids), sorted(ids - table)
        if no_bold or not_presented:
            raise ValueError(
                'The stimulus table and the assembly must hold the same videos: '
                f'{len(no_bold)} stimuli of the table have no BOLD in the assembly '
                f'(for example {no_bold[:3]}), and {len(not_presented)} videos of the '
                f'assembly are not in the table (for example {not_presented[:3]}).')
        test = boldmoments_test_set(ids)
        if not test:
            raise ValueError('The assembly holds no test-set video (id above 1000): '
                             'there is nothing to score.')
        train = sorted(ids - set(test))
        if not train:
            raise ValueError('The assembly holds no training-set video: there is nothing to fit.')
        return train, test

    @property
    def voxel_mask(self) -> Optional[np.ndarray]:
        """The scored voxels (``None``: all of them); computed once."""
        if self._voxel_mask is None:
            self._voxel_mask = _select_voxels(self._reliability_threshold, self._voxel_mask_fn,
                                              self._mask_reliability)
        return self._voxel_mask

    def _frames(self):
        return frame_table(self.stimulus_set, self._sample_times_ms)

    def _per_voxel_ceiling(self, mask):
        """sqrt(reliability on the scored test-set videos) per scored voxel; ``None``
        without a measured ceiling.

        Whole cortex and every variant without a measured ceiling hold
        ``not_estimated()`` and take the scalar path. A voxel of the mask whose
        reliability on the scored videos is at or below zero gets a ceiling of zero:
        ``voxel_score`` leaves it out of the median and counts it in
        ``n_voxels_excluded``.
        """
        if not _ceiling_measured(self):
            return None
        rel = np.asarray(self._ceiling_reliability(), dtype=float)
        if mask is not None:
            rel = rel[np.asarray(mask, dtype=bool)]
        return np.sqrt(np.clip(rel, 0.0, 1.0))

    def _features(self, subject):
        """``(features (n_videos, n_features), clip_ids, attrs)`` recorded through ``participate``.

        Every region the benchmark names is requested (rule 5.1a). The streams
        shown are the benchmark inputs the subject takes in (``taken_inputs``, the
        rule of pre-flight), and each region is recorded from a stream the subject
        can answer it from (``stream_regions``). ``attrs['scored_inputs']`` names
        the inputs shown (Contract Rules 5.4, 5.5); the two-stream labels are set
        only when both streams are shown.
        """
        stimuli = self.stimulus_set
        plan = stream_regions(subject, taken_inputs(subject, self.required_input_channels),
                              self._regions)
        scored = sorted(channel for _, channel, _ in plan)
        if [stream for stream, _, _ in plan] == ['video']:
            responses = record_visual(
                subject, lambda: clip_table(stimuli, f'{stimuli.identifier}-videos'),
                self._frames, plan[0][2])
            clip_ids = read_stimulus_ids(responses[0])
            features = np.concatenate(
                [align(time_mean(response), read_stimulus_ids(response), clip_ids)
                 for response in responses], axis=1)
            attrs = self._visual_attrs(subject)
            if self._audio:
                attrs['scored_inputs'] = scored
            return features, clip_ids, attrs
        # The sound, with the visual stream when the subject takes it. The clip table
        # is the one of the one-stream route, so the video activations are read once
        # for both. The name needs no file signature: both routes build it from the
        # same stimuli, ids and order, so it names the same files, and only a subject
        # that declares vision:video gets it (an old image model gets frames). A
        # difference from that route, kept so no score moves: the time bins are averaged with xarray's NaN-skipping mean (the one-stream route and
        # the time-resolved benchmark use numpy's mean; the two differ only when a
        # response holds NaN).
        # The sound tracks are checked before any stream is recorded.
        sound = clip_table(stimuli, f'{stimuli.identifier}-multimodal-audio', sound=True)
        streams = []
        for stream, _, regions in plan:
            if stream == 'video':
                responses = record_visual(
                    subject, lambda: clip_table(stimuli, f'{stimuli.identifier}-videos'),
                    self._frames, regions)
            else:
                responses = record_clips(subject, sound, regions)
            streams.append((stream, [response.mean(dim='time_bin') if 'time_bin' in response.dims
                                     else response for response in responses]))
        clip_ids = read_stimulus_ids(streams[0][1][0])
        blocks = [np.concatenate([align(response.values, read_stimulus_ids(response), clip_ids)
                                  for response in responses], axis=1)
                  for _, responses in streams]
        attrs = {f'n_features_{stream}': int(block.shape[1])
                 for (stream, _), block in zip(streams, blocks)}
        attrs['scored_inputs'] = scored
        if len(streams) == 2:
            attrs.update(pipeline='multimodal_av_concat', mode='concat')
        return np.concatenate(blocks, axis=1), clip_ids, attrs

    def _visual_attrs(self, subject) -> dict:
        """The route of a one-stream score: whole clips, or frames sampled from them."""
        from brainscore.presentation import frames_from_clip, takes_clips
        if takes_clips(subject):
            return {'pipeline': 'video_native', 'input_adaptation': None, 'note': (
                'Video-native model: full temporal dynamics preserved '
                'through the encoder; features mean-pooled across output '
                'time steps for regression against the single Lahner2024 '
                'BOLD estimate per video.')}
        return {
            'pipeline': 'frame_aggregation',
            'input_adaptation': frames_from_clip(self._sample_times_ms),
            'sample_times_ms': self._sample_times_ms,
            'n_frames_per_video': len(self._sample_times_ms),
            'note': (f'{len(self._sample_times_ms)} static frames per video, '
                     f'mean-pooled before regression. Temporal dynamics discarded. '
                     f'See benchmark docstring.'),
        }

    def __call__(self, subject) -> Score:
        self._split()   # every stimulus has BOLD and both sets exist, or this fails before the subject runs
        features, clip_ids, route_attrs = self._features(subject)

        # Average the participants; align to the response order; keep the scored voxels.
        neural = self.assembly.squeeze('time_bin', drop=True).groupby('stimulus_id').mean('presentation')
        neural_mat = neural.sel(stimulus_id=list(clip_ids)).transpose('stimulus_id', 'neuroid').values
        mask = self.voxel_mask
        if mask is not None:
            neural_mat = neural_mat[:, mask]

        # One fit on the training-set videos, the correlation on the test-set videos:
        # one fixed fold, so held_out_r correlates over the test-set rows only. Ridge
        # (scikit-learn, penalty 1) on the raw features: it centres through its
        # intercept; a StandardScaler over-rescales features of unequal variance and
        # lowers the fit.
        test = np.isin(np.asarray(clip_ids, dtype=str), self._split()[1])
        per_voxel_r, _ = cross_validated_r(
            features, neural_mat, [(~test, test)], ridge(1.0, SklearnRidge()))
        score = voxel_score(self, per_voxel_r, self._per_voxel_ceiling(mask))
        score.attrs['n_videos_fitted'] = int((~test).sum())
        score.attrs['n_videos_scored'] = int(test.sum())
        score.attrs.update(route_attrs)
        return score

# Noise ceiling for the visual-ROI voxel set. For a predictivity correlation the ceiling
# is sqrt(reliability): the highest r a perfect model could reach given measurement
# noise. Under the protocol of the data set the constant is the median of
# sqrt(reliability on the scored videos, the 93 test-set videos) over the voxels of the
# mask (reliability of at least 0.3 on the 933 training-set videos) that have a
# positive ceiling. Measured on the hosted data: the mask has 3958 voxels, every one
# with a positive ceiling, and this is their median. It is what ``benchmark.ceiling``
# reports for both ``-visualROI`` variants; the score divides each voxel by its own
# ceiling and does not read this value.
VISUAL_ROI_CEILING = 0.8515666971964203
