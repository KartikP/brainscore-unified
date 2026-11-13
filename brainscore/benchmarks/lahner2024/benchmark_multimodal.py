"""Lahner 2024 BOLDMoments — multimodal (audio + video) variant.

The first benchmark in the unified interface that exercises BOTH the
multi-region recording API and per-modality dispatch on real model
forward passes against real fMRI data. Sister to
``Lahner2024-fMRI-naturalistic`` (video-only) and
``...-naturalistic-visualROI`` (reliability-thresholded ROI).

Pipeline:
1. Build TWO single-modality views of the stimulus_set:
   - one with ``video_path`` (BOLDMoments MP4)
   - one with ``audio_path`` (16 kHz mono WAV demuxed from the MP4 via
     ``prepare_audio_tracks.py``)
2. For each modality, record through ``participate`` (``neural_response``):
   ``neural:IT`` for the video view, ``neural:A1`` for the audio view → 3D
   ``(presentation, time_bin, neuroid)`` assembly → mean-pool the
   time_bin axis to get a single feature vector per video.
3. Concatenate features along the neuroid axis with a per-neuroid
   ``modality`` coord. The model is now represented by one
   ``(n_videos, n_features_video + n_features_audio)`` matrix.
4. Standard cross-validated Ridge regression against per-video BOLD
   betas, per-voxel Pearson, median across voxels.

Why two stim_sets and NOT ``multi_modality=True``? Each wrapper
returns time-resolved features at its native rate (V-JEPA v1: 8
temporal tubelets per 3-second clip; Wav2Vec2-base: ~150 audio frames
per 3-second clip @ 50 Hz output). xarray cannot concat along the
neuroid axis when the time_bin axes differ in size. Per-modality
extraction + per-modality time-mean + concat sidesteps the alignment
issue cleanly. The new ``multi_modality=True`` API still applies —
just for benchmarks where wrappers can be configured to share a time
grid (e.g., a continuous-naturalistic benchmark using ``temporal_bin``
to align both modalities to the brain's TR grid).

Subjects that do not declare a visual input, ``audio``, ``neural:IT`` and
``neural:A1`` cannot be scored on this variant — the benchmark raises before
any compute so misuse is caught early. Use the
non-multimodal variants for unimodal models.
"""

from functools import partial
from typing import Optional

import numpy as np
import pandas as pd

from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from brainscore.metrics.encoding import (
    NormalEquations,
    SklearnRidge,
    cross_validated_r,
    each_band,
    grid_search,
    penalty_grid,
    random_row_split,
    ridge,
    row_kfold,
    summarize,
)
from ._util import (
    read_stimulus_ids, record_clips, record_frames_per_clip, takes_audiovisual,
)

from brainscore.benchmarks._score import benchmark_score
from .benchmark import (
    BIBTEX,
    DEFAULT_SAMPLE_TIMES_MS,
    Lahner2024BOLDMoments,
    VIDEO_DURATION_MS,
    _ceiling_measured,
)


# Default location of pre-extracted audio files. Override via constructor.
DEFAULT_AUDIO_DIR = '~/.brainio/lahner2024_audio_16k'



def multimodal_fit(mode: str, alpha_grid):
    """The fit of a multimodal scoring mode; the time-resolved variant uses it too.

    ``banded`` is one joint ridge with one penalty per feature group (video,
    audio), chosen from ``alpha_grid`` on a random 20 % of the training rows,
    with nothing centred. That inner split leaks on autocorrelated rows (issue
    #86); a fix moves the scores, so it waits for a new benchmark version. Every
    other mode fits one scikit-learn ridge at penalty 1 per feature group and
    adds the predictions (one group for ``concat``, ``video_only`` and
    ``audio_only``).
    """
    if mode == 'banded':
        return ridge(grid_search(penalty_grid(alpha_grid, 2),
                                 partial(random_row_split, fraction=0.2, seed=0)),
                     NormalEquations(centre='none'))
    return each_band(ridge(1.0, SklearnRidge()))

class Lahner2024BOLDMoments_multimodal(Lahner2024BOLDMoments):
    """Audio + video variant of the Lahner2024 GLM-beta benchmark.

    Inherits the assembly + ridge regression logic from the parent;
    overrides ``__call__`` to drive a per-modality extraction.
    """

    # Sends the visual stream and the audio stream; asks for the visual tower
    # (`neural:IT`) and the audio tower (`neural:A1`). Each clip is an independent trial.
    required_input_channels = frozenset({'vision:image', 'vision:video', 'audio'})
    requested_output_channels = frozenset({'neural:IT', 'neural:A1'})
    trial_memory = 'none'

    VALID_MODES = ('concat', 'per_modality', 'video_only', 'audio_only',
                   'banded')
    # Banded-ridge α grid. Per-modality α is selected from this grid via
    # held-out validation inside each outer CV fold. Logarithmic spread
    # gives banded ridge enough range to push uninformative modalities
    # to ~zero contribution (the audio-on-visual-cortex case) without
    # over-shrinking informative ones.
    BANDED_ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)

    def __init__(
        self,
        audio_dir: Optional[str] = None,
        ceiling: Optional[float] = None,
        reliability_threshold: Optional[float] = None,
        identifier_suffix: str = '-multimodal',
        mode: str = 'concat',
        voxel_mask_fn: Optional[callable] = None,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}; got {mode!r}.")
        super().__init__(
            ceiling=ceiling,
            reliability_threshold=reliability_threshold,
            identifier_suffix=identifier_suffix,
        )
        from pathlib import Path
        self._audio_dir = Path(audio_dir or DEFAULT_AUDIO_DIR).expanduser()
        # Optional custom voxel mask. When set, overrides the
        # reliability-threshold mask from the parent class — the
        # auditory-ROI variant uses this with a Destrieux-based mask
        # over auditory cortex vertices on fsaverage5.
        self._voxel_mask_fn = voxel_mask_fn
        # Scoring mode:
        # - 'concat': fit one Ridge on [video|audio] concat (default; what
        #   we shipped first). Sensitive to dilution when one modality
        #   carries little target-aligned signal.
        # - 'per_modality': fit Ridge on video features and Ridge on
        #   audio features separately, sum the held-out predictions.
        #   Equivalent to fitting two independent encoders and adding
        #   their outputs — eliminates cross-modality regularization
        #   competition.
        # - 'video_only' / 'audio_only': single-tower upper-bounds for
        #   isolating each modality's contribution.
        self._mode = mode

    def _get_voxel_mask(self):
        """Return the voxel mask for this benchmark variant.

        - If ``voxel_mask_fn`` was passed, call it (one-time) and cache.
        - Otherwise fall back to the parent's split-half-reliability mask.
        """
        if self._voxel_mask_fn is None:
            return super()._get_voxel_mask()
        if not hasattr(self, '_custom_mask_cache'):
            mask = self._voxel_mask_fn()
            mask = np.asarray(mask, dtype=bool)
            self._custom_mask_cache = mask
        return self._custom_mask_cache

    # ── Per-modality stim sets ─────────────────────────────────────

    def _video_stim_set(self) -> StimulusSet:
        stim = self.stimulus_set
        rows = []
        for _, row in stim.iterrows():
            rows.append({
                'stimulus_id': row['stimulus_id'],
                'video_path': str(stim.get_stimulus(row['stimulus_id'])),
            })
        df = pd.DataFrame(rows)
        out = StimulusSet(df)
        out.identifier = f'{stim.identifier}-multimodal-video'
        out.stimulus_paths = dict(zip(df['stimulus_id'], df['video_path']))
        return out

    def _audio_stim_set(self) -> StimulusSet:
        from pathlib import Path
        stim = self.stimulus_set
        rows = []
        missing = []
        for _, row in stim.iterrows():
            video_path = Path(stim.get_stimulus(row['stimulus_id']))
            audio_path = self._audio_dir / f'{video_path.stem}.wav'
            if not audio_path.exists():
                missing.append(str(audio_path))
            rows.append({
                'stimulus_id': row['stimulus_id'],
                'audio_path': str(audio_path),
            })
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} audio files missing under {self._audio_dir}. "
                f"Run `python -m brainscore.data.lahner2024."
                f"prepare_audio_tracks --audio-dir {self._audio_dir}` first. "
                f"First missing: {missing[0]}"
            )
        df = pd.DataFrame(rows)
        out = StimulusSet(df)
        out.identifier = f'{stim.identifier}-multimodal-audio'
        out.stimulus_paths = dict(zip(df['stimulus_id'], df['audio_path']))
        return out

    # ── Per-modality extraction + concat ──────────────────────────

    _read_stimulus_ids = staticmethod(read_stimulus_ids)

    @staticmethod
    def _to_2d(assembly):
        """Mean-pool the time_bin axis so each presentation is one
        feature vector — required because cross-modality time grids do
        NOT align (V-JEPA's 8 tubelets vs Wav2Vec2's 150 frames over a
        3-second clip)."""
        if 'time_bin' in assembly.dims:
            return assembly.mean(dim='time_bin')
        return assembly

    def _multimodal_features(self, subject):
        """Extract visual + audio features from the subject.

        Visual tower routing:
        - subject declares 'vision:video' → use video stim set, native
          temporal extraction (V-JEPA / VideoMAE / etc.)
        - any other visual subject (stills only) → use
          frame-aggregation: extract N frames per clip, run CLIP/BLIP-2/
          Qwen-VL on each as a still image, mean-pool over frames

        The first branch produces richer temporal features but is only
        available for native-video models. The second branch lets us
        score CLIP-class VLMs on the same multimodal benchmark by
        treating them as frame extractors.

        Returns:
            features: (n_videos, n_features) numpy array, visual first,
                audio second.
            clip_ids: list of stimulus_ids in the same order.
            modality_per_neuroid: ndarray of {'video', 'audio'} strings,
                one per output feature column.
        """
        time_bins = [(0, VIDEO_DURATION_MS)]
        # Clips natively when the subject declares vision:video, frames otherwise.
        if Lahner2024BOLDMoments._is_native_video(subject):
            video_assembly = record_clips(
                subject, self._video_stim_set(), 'IT', time_bins)
        else:
            # Frame-aggregation path: expand each clip into N frames, show
            # them as stills, then mean-pool per clip.
            video_assembly = record_frames_per_clip(
                subject, self._expand_videos(), 'IT', time_bins)
        video_2d = self._to_2d(video_assembly)
        video_clip_ids = self._read_stimulus_ids(video_2d)
        v_feats = video_2d.values

        audio_assembly = record_clips(
            subject, self._audio_stim_set(), 'A1', time_bins)
        audio_2d = self._to_2d(audio_assembly)
        audio_clip_ids = self._read_stimulus_ids(audio_2d)
        a_feats = audio_2d.values

        if video_clip_ids != audio_clip_ids:
            # Different ordering — re-align audio to the video order
            order = {cid: i for i, cid in enumerate(audio_clip_ids)}
            perm = [order[cid] for cid in video_clip_ids]
            a_feats = a_feats[perm]

        features = np.concatenate([v_feats, a_feats], axis=1)
        modality = np.concatenate([
            np.array(['video'] * v_feats.shape[1]),
            np.array(['audio'] * a_feats.shape[1]),
        ])
        return features, video_clip_ids, modality

    # ── Scoring ────────────────────────────────────────────────────

    def __call__(self, subject) -> Score:
        # Route on declarations only (Contract Rule 2.2), before any compute.
        if not takes_audiovisual(subject):
            raise ValueError(
                f"{self.identifier} requires the subject to take both "
                f"a visual input ('vision:image' or 'vision:video') AND "
                f"'audio', and to record neural:IT and neural:A1. Got "
                f"in_channels={sorted(subject.in_channels)}, "
                f"out_channels={sorted(subject.out_channels)}. "
                f"Use Lahner2024-fMRI-naturalistic for video-only models."
            )

        features, clip_ids, modality_per_neuroid = (
            self._multimodal_features(subject))

        # Slice features per scoring mode. modality_per_neuroid is a
        # parallel array tagging each column 'video' or 'audio'.
        v_mask = modality_per_neuroid == 'video'
        a_mask = modality_per_neuroid == 'audio'
        if self._mode == 'video_only':
            features_groups = [features[:, v_mask]]
        elif self._mode == 'audio_only':
            features_groups = [features[:, a_mask]]
        elif self._mode == 'per_modality':
            features_groups = [features[:, v_mask], features[:, a_mask]]
        elif self._mode == 'banded':
            # Banded ridge solves jointly with a per-modality penalty.
            # Pass the v/a partition through to the fold loop so each
            # fold can tune α_v / α_a on inner held-out data.
            features_groups = [features[:, v_mask], features[:, a_mask]]
        else:  # 'concat'
            features_groups = [features]

        # Align neural to model order; mask voxels per ROI threshold.
        neural = self._average_repetitions()
        neural_aligned = neural.sel(
            stimulus_id=list(clip_ids)
        ).transpose('stimulus_id', 'neuroid')
        neural_mat = neural_aligned.values

        mask = self._get_voxel_mask()
        if mask is not None:
            neural_mat = neural_mat[:, mask]

        n = features.shape[0]
        per_voxel_r, chosen = cross_validated_r(
            features_groups, neural_mat, row_kfold(n, n_splits=5, random_state=0),
            multimodal_fit(self._mode, self.BANDED_ALPHA_GRID))
        chosen_alphas = ([fold.shared for fold in chosen]
                         if self._mode == 'banded' else [])
        per_voxel_r, median_r, mean_r = summarize(per_voxel_r)

        score = benchmark_score(
            self, median_r / float(self.ceiling), raw=median_r,
            metric='median_per_voxel_pearson_r', normalized=_ceiling_measured(self))
        score.attrs['mean_r'] = mean_r
        score.attrs['n_voxels_scored'] = int(len(per_voxel_r))
        # Same contract as the single-modality benchmark: expose the per-voxel scores
        # and WHICH vertices they belong to, so they can be rendered on a cortex.
        score.attrs['per_voxel_r'] = per_voxel_r
        mask = self._get_voxel_mask()
        if mask is not None:
            import numpy as _np
            score.attrs['voxel_mask_n_total'] = int(mask.size)
            score.attrs['voxel_mask_indices'] = _np.flatnonzero(mask)
        score.attrs['n_videos'] = int(n)
        score.attrs['pipeline'] = f'multimodal_av_{self._mode}'
        score.attrs['mode'] = self._mode
        if chosen_alphas:
            score.attrs['banded_alpha_video_per_fold'] = [
                float(av) for av, _ in chosen_alphas]
            score.attrs['banded_alpha_audio_per_fold'] = [
                float(aa) for _, aa in chosen_alphas]
        score.attrs['n_features_video'] = int(
            (modality_per_neuroid == 'video').sum())
        score.attrs['n_features_audio'] = int(
            (modality_per_neuroid == 'audio').sum())
        if mask is not None:
            score.attrs['voxel_mask_n_total'] = int(mask.size)
            score.attrs['voxel_mask_n_kept'] = int(mask.sum())
            if self._reliability_threshold is not None:
                score.attrs['reliability_threshold'] = float(
                    self._reliability_threshold)
            if self._voxel_mask_fn is not None:
                score.attrs['voxel_mask_source'] = (
                    self._voxel_mask_fn.__name__)
        return score
