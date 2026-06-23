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
2. For each modality: ``candidate.process(stim)`` → 3D
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

Models without an ``audio`` preprocessor cannot be scored on this
variant — the benchmark raises so misuse is caught early. Use the
non-multimodal variants for unimodal models.
"""

from typing import Optional

import numpy as np
import pandas as pd

from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from brainscore.tools.banded_ridge import banded_ridge_fit_predict
from ._util import read_stimulus_ids

from .benchmark import (
    BIBTEX,
    DEFAULT_SAMPLE_TIMES_MS,
    Lahner2024BOLDMoments,
    VIDEO_DURATION_MS,
    load_stimulus_set,
)


# Default location of pre-extracted audio files. Override via constructor.
DEFAULT_AUDIO_DIR = '~/.brainio/lahner2024_audio_16k'


class Lahner2024BOLDMoments_multimodal(Lahner2024BOLDMoments):
    """Audio + video variant of the Lahner2024 GLM-beta benchmark.

    Inherits the assembly + ridge regression logic from the parent;
    overrides ``__call__`` to drive a per-modality extraction.
    """

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
                f"Run `python -m brainscore.benchmarks.lahner2024."
                f"prepare_audio_tracks --audio-dir {self._audio_dir}` first. "
                f"First missing: {missing[0]}"
            )
        df = pd.DataFrame(rows)
        out = StimulusSet(df)
        out.identifier = f'{stim.identifier}-multimodal-audio'
        out.stimulus_paths = dict(zip(df['stimulus_id'], df['audio_path']))
        return out

    # ── Per-modality extraction + concat ──────────────────────────

    def _banded_ridge_fit_predict(self, X_train_groups, X_test_groups,
                                  Y_train):
        """Banded ridge with per-group α — see ``tools.banded_ridge``.

        Returns ``(Y_test_pred, (alpha_v, alpha_a))``.
        """
        return banded_ridge_fit_predict(
            X_train_groups, X_test_groups, Y_train, self.BANDED_ALPHA_GRID)

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

    def _multimodal_features(self, candidate):
        """Extract visual + audio features from the candidate.

        Visual tower routing:
        - candidate has 'video' modality → use video stim set, native
          temporal extraction (V-JEPA / VideoMAE / etc.)
        - candidate has 'vision' modality (still-image only) → use
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
        supports = getattr(candidate, 'supported_modalities', set())
        candidate.start_recording('IT', time_bins=[(0, VIDEO_DURATION_MS)])
        if 'video' in supports:
            video_assembly = candidate.process(self._video_stim_set())
        else:
            # Frame-aggregation path: expand each clip into N frames,
            # process them as a still-image stim set, then mean-pool.
            from brainscore_core.temporal import temporal_bin
            frame_stim = self._expand_videos()
            per_frame = candidate.process(frame_stim)
            video_assembly = temporal_bin(
                per_frame, time_bins=[(0, VIDEO_DURATION_MS)])
        video_2d = self._to_2d(video_assembly)
        video_clip_ids = self._read_stimulus_ids(video_2d)
        v_feats = video_2d.values

        candidate.start_recording('A1', time_bins=[(0, VIDEO_DURATION_MS)])
        audio_assembly = candidate.process(self._audio_stim_set())
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

    def __call__(self, candidate) -> Score:
        supports = getattr(candidate, 'supported_modalities', set())
        if 'audio' not in supports:
            raise ValueError(
                f"{self.identifier} requires the candidate to support both "
                f"a visual modality ('video' or 'vision') AND 'audio'. Got "
                f"supported_modalities={supports}. "
                f"Use Lahner2024-fMRI-naturalistic for video-only models."
            )
        if 'video' not in supports and 'vision' not in supports:
            raise ValueError(
                f"{self.identifier} requires the candidate to support both "
                f"a visual modality ('video' or 'vision') AND 'audio'. Got "
                f"supported_modalities={candidate.supported_modalities}."
            )

        features, clip_ids, modality_per_neuroid = (
            self._multimodal_features(candidate))

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

        from sklearn.model_selection import KFold
        from sklearn.linear_model import Ridge

        n = features.shape[0]
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        fold_preds = np.zeros_like(neural_mat)
        chosen_alphas = []
        for train_idx, test_idx in kf.split(np.arange(n)):
            if self._mode == 'banded':
                X_train = [g[train_idx] for g in features_groups]
                X_test = [g[test_idx] for g in features_groups]
                Y_train = neural_mat[train_idx]
                preds, alpha_pair = self._banded_ridge_fit_predict(
                    X_train, X_test, Y_train)
                fold_preds[test_idx] = preds
                chosen_alphas.append(alpha_pair)
            else:
                # Per-modality ridge (sum of independents) when
                # features_groups has 2 entries; concat / video_only /
                # audio_only when it has 1. Each group gets its own
                # Ridge with α=1.0 and predictions are summed.
                for X in features_groups:
                    reg = Ridge(alpha=1.0).fit(
                        X[train_idx], neural_mat[train_idx])
                    fold_preds[test_idx] += reg.predict(X[test_idx])

        n_voxels = neural_mat.shape[1]
        per_voxel_r = np.zeros(n_voxels)
        for j in range(n_voxels):
            yt = neural_mat[:, j]
            yp = fold_preds[:, j]
            if yt.std() > 0 and yp.std() > 0:
                per_voxel_r[j] = np.corrcoef(yt, yp)[0, 1]
            else:
                per_voxel_r[j] = np.nan
        per_voxel_r = per_voxel_r[~np.isnan(per_voxel_r)]
        median_r = float(np.median(per_voxel_r))
        mean_r = float(np.mean(per_voxel_r))

        score = Score(median_r / float(self.ceiling))
        score.attrs['raw'] = Score(median_r)
        score.attrs['mean_r'] = mean_r
        score.attrs['n_voxels_scored'] = int(len(per_voxel_r))
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


def Lahner2024BOLDMoments_multimodal_visualROI(
        audio_dir: Optional[str] = None):
    """Reliability-thresholded multimodal variant. Same threshold as the
    single-modality ROI variant (0.3) so the scores compare directly."""
    return Lahner2024BOLDMoments_multimodal(
        audio_dir=audio_dir,
        reliability_threshold=0.3,
        identifier_suffix='-multimodal-visualROI',
    )


def Lahner2024BOLDMoments_multimodal_auditoryROI(
        audio_dir: Optional[str] = None,
        mode: str = 'concat'):
    """Auditory-cortex variant. Uses the Destrieux 2009 surface atlas
    on fsaverage5 to select early auditory + planum voxels (HG, lateral
    STG, planum polare/tempo, transverse temporal sulcus). ~526 voxels.

    This is the variant where audio features SHOULD predict and video
    features SHOULD NOT — fair test of whether the multimodal pipeline
    detects asymmetric utility in either direction.
    """
    from .auditory_roi import build_auditory_mask
    return Lahner2024BOLDMoments_multimodal(
        audio_dir=audio_dir,
        mode=mode,
        identifier_suffix='-multimodal-auditoryROI',
        voxel_mask_fn=build_auditory_mask,
    )
