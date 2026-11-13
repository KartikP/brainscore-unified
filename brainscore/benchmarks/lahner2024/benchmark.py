"""
Lahner 2024 BOLDMoments fMRI naturalistic benchmark (unified interface).

Participants watched 1026 short (3s) video clips while being scanned with fMRI;
responses were averaged per TR and fit with GLM to give one BOLD estimate
per (stimulus, repetition, voxel). The assembly has shape:

    (time_bin=1, neuroid=20484, presentation=10260)

where ``time_bin`` spans the full 3-second clip (single pre-computed
estimate; no time-series regression required), ``neuroid`` is cortical
vertices across L/R hemispheres, and ``presentation`` is 1026 videos × 10
repetitions. We average repetitions to get one clean response per video.

Taken from brain-score/vision PR #1249 (YingtianDt):
    - BOLDMoments as the stimulus-set identifier
    - Citation

Scoring pipeline:
    1. Record ``neural:IT`` through ``participate`` (``neural_response``): a
       subject that declares ``vision:video`` sees whole clips; any other
       subject sees N frames sampled from each clip, pooled per clip
       (``temporal_bin``). With the ``audio`` channel, ``neural:A1`` is also
       recorded for the sound track of each clip, and the two are joined
       column-wise.
    2. Average the response over its time bins → one vector per video.
    3. Average the neural assembly across the 10 repetitions → one BOLD
       vector per video per voxel; keep the voxels of the voxel mask.
    4. 5-fold ridge (penalty 1, ``brainscore.metrics.encoding``) → Pearson
       per voxel → median over the voxels that count (``voxel_score``).

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

from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score, per_unit_pearson
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuronRecordingAssembly,
)
from brainscore import load_dataset, load_stimulus_set
from brainscore.benchmarks._score import benchmark_score
from brainscore.metrics.encoding import SklearnRidge, cross_validated_r, ridge, row_kfold
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

    A benchmark built without a ceiling keeps ``Score(1.0)`` and reports the raw
    correlation; only one given a measured ceiling normalizes.
    """
    return float(benchmark.ceiling) != 1.0


def _split_half_reliability(assembly, n_splits: int = 20, random_state: int = 0) -> np.ndarray:
    """Per-voxel split-half reliability of the GLM-beta assembly across repetitions.

    For each random split of each video's repetitions into two halves, average
    within each half to get (n_videos, n_voxels), then Pearson-correlate the
    two halves across videos for each voxel. Average across ``n_splits``
    random splits and apply the Spearman-Brown correction.

    Returns a (n_voxels,) array. High-reliability voxels are stimulus-driven
    (signal); low-reliability voxels are noise-dominated.
    """
    a = assembly.squeeze('time_bin', drop=True)
    stim_ids = np.asarray(a['stimulus_id'].values)
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
    raw median is divided by ``benchmark.ceiling``.
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
            benchmark, float(np.median(np.asarray(kept, dtype=float) / ceiling_per_voxel[counts])),
            raw=median_r, metric='median_per_voxel_pearson_r', normalized=True,
            ceiling=Score(float(np.median(ceiling_per_voxel[counts]))),
            ceiling_aggregation='per_voxel')
    else:
        score = benchmark_score(
            benchmark, median_r / float(benchmark.ceiling), raw=median_r,
            metric='median_per_voxel_pearson_r', normalized=_ceiling_measured(benchmark),
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
            sound track of each clip and records ``neural:A1``, and the subject
            must take both.
        sample_times_ms: the frames a stills-only subject sees from each clip.
        frames_dir: where extracted frames are kept (see ``resolve_frames_dir``).
        audio_dir: where the sound tracks are (16 kHz WAV, one per clip).
        ceiling: a measured noise ceiling; the score is then normalized per voxel.
        reliability_threshold: score only the voxels whose split-half
            reliability is at least this value.
        voxel_mask_fn: score only the voxels of this mask (for example
            ``build_auditory_mask``). Give at most one of the two.
        identifier_suffix: appended to ``Lahner2024-fMRI-naturalistic``.
    """

    trial_memory = 'none'   # each clip is an independent trial

    def __init__(
        self,
        channels: Sequence[str] = _VISUAL,
        sample_times_ms: List[float] = list(DEFAULT_SAMPLE_TIMES_MS),
        frames_dir: Optional[Path] = None,
        audio_dir: Optional[str] = None,
        ceiling: Optional[float] = None,
        reliability_threshold: Optional[float] = None,
        voxel_mask_fn: Optional[Callable[[], np.ndarray]] = None,
        identifier_suffix: str = '',
    ):
        if reliability_threshold is not None and voxel_mask_fn is not None:
            raise ValueError('Give reliability_threshold or voxel_mask_fn, not both.')
        channels, regions = channel_regions(channels)
        self.required_input_channels = channels
        self.requested_output_channels = frozenset(f'neural:{region}' for region in regions)
        self._audio = 'audio' in channels
        self._sample_times_ms = list(sample_times_ms)
        self._frames_dir = resolve_frames_dir(frames_dir)
        self._audio_dir = Path(audio_dir or DEFAULT_AUDIO_DIR).expanduser()
        self._reliability_threshold = reliability_threshold
        self._voxel_mask_fn = voxel_mask_fn
        self._assembly: Optional[NeuronRecordingAssembly] = None
        self._stimulus_set = None
        self._reliability: Optional[np.ndarray] = None
        self._voxel_mask: Optional[np.ndarray] = None
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
            self._assembly = load_dataset('Lahner2024-fMRI')
        return self._assembly

    @property
    def stimulus_set(self):
        if self._stimulus_set is None:
            self._stimulus_set = load_stimulus_set('BOLDMoments')
        return self._stimulus_set

    def _reliability_once(self) -> np.ndarray:
        """``_split_half_reliability`` of this benchmark's assembly, computed once."""
        if self._reliability is None:
            self._reliability = _split_half_reliability(self.assembly)
        return self._reliability

    @property
    def voxel_mask(self) -> Optional[np.ndarray]:
        """The scored voxels (``None``: all of them); computed once."""
        if self._voxel_mask is None:
            self._voxel_mask = _select_voxels(self._reliability_threshold, self._voxel_mask_fn,
                                              self._reliability_once)
        return self._voxel_mask

    def _frames(self):
        return frame_table(self.stimulus_set, self._sample_times_ms, self._frames_dir)

    def _per_voxel_ceiling(self, mask):
        """sqrt(reliability) per scored voxel; ``None`` without a measured ceiling.

        Whole cortex and every variant without a measured ceiling keep
        ``ceiling=1.0`` and take the scalar path. A ceiling of zero (reliability
        at or below zero) leaves the voxel out (``voxel_score``).
        """
        if not _ceiling_measured(self):
            return None
        rel = np.asarray(self._reliability_once(), dtype=float)
        if mask is not None:
            rel = rel[np.asarray(mask, dtype=bool)]
        return np.sqrt(np.clip(rel, 0.0, 1.0))

    def _features(self, subject):
        """``(features (n_videos, n_features), clip_ids, attrs)`` recorded through ``participate``."""
        stimuli = self.stimulus_set
        if not self._audio:
            visual = record_visual(subject, lambda: clip_table(stimuli, f'{stimuli.identifier}-videos'),
                                   self._frames)
            return time_mean(visual), read_stimulus_ids(visual), self._visual_attrs(subject)
        # Two streams. Differences from the one-stream route above, kept so no score
        # moves: the clip table is named '-multimodal-video', and the time bins are
        # averaged with xarray's NaN-skipping mean (the one-stream route and the
        # time-resolved benchmark use numpy's mean; the two differ only when a
        # response holds NaN).
        visual = record_visual(
            subject, lambda: clip_table(stimuli, f'{stimuli.identifier}-multimodal-video'),
            self._frames)
        sound = record_clips(subject, clip_table(stimuli, f'{stimuli.identifier}-multimodal-audio',
                                                 audio_dir=self._audio_dir), 'A1')
        visual, sound = (response.mean(dim='time_bin') if 'time_bin' in response.dims
                         else response for response in (visual, sound))
        clip_ids = read_stimulus_ids(visual)
        v_feats = visual.values
        a_feats = align(sound.values, read_stimulus_ids(sound), clip_ids)
        attrs = {'pipeline': 'multimodal_av_concat', 'mode': 'concat',
                 'n_features_video': int(v_feats.shape[1]),
                 'n_features_audio': int(a_feats.shape[1])}
        return np.concatenate([v_feats, a_feats], axis=1), clip_ids, attrs

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
        # Route on declarations only (Contract Rule 2.2), before any compute.
        if self._audio and not takes_audiovisual(subject):
            raise ValueError(
                f"{self.identifier} requires the subject to take both "
                f"a visual input ('vision:image' or 'vision:video') AND "
                f"'audio', and to record neural:IT and neural:A1. Got "
                f"in_channels={sorted(subject.in_channels)}, "
                f"out_channels={sorted(subject.out_channels)}. "
                f"Use Lahner2024-fMRI-naturalistic for video-only models."
            )
        features, clip_ids, route_attrs = self._features(subject)

        # Average the repetitions; align to the response order; keep the scored voxels.
        neural = self.assembly.squeeze('time_bin', drop=True).groupby('stimulus_id').mean('presentation')
        neural_mat = neural.sel(stimulus_id=list(clip_ids)).transpose('stimulus_id', 'neuroid').values
        mask = self.voxel_mask
        if mask is not None:
            neural_mat = neural_mat[:, mask]

        # Ridge (scikit-learn, penalty 1) on the raw features: it centres through its
        # intercept; a StandardScaler over-rescales features of unequal variance and
        # cost about 0.18 raw r here (2026-04-24 replication note).
        per_voxel_r, _ = cross_validated_r(
            features, neural_mat, row_kfold(len(features), n_splits=5, random_state=0),
            ridge(1.0, SklearnRidge()))
        score = voxel_score(self, per_voxel_r, self._per_voxel_ceiling(mask))
        score.attrs['n_videos'] = int(features.shape[0])
        score.attrs.update(route_attrs)
        return score


# Noise ceiling for the visual-ROI voxel set, measured on EC2 2026-07-31.
# For a predictivity correlation the ceiling is sqrt(reliability): the highest r a
# perfect model could reach given measurement noise. This is the median over the 4042
# ROI voxels of sqrt(_split_half_reliability(assembly, n_splits=20, random_state=0)),
# the same Spearman-Brown-corrected reliability that selects those voxels. The
# constant only marks the variant that divides by a ceiling: the score recomputes
# the per-voxel ceilings from the assembly and does not read this value.
VISUAL_ROI_CEILING = 0.7304512193654528
