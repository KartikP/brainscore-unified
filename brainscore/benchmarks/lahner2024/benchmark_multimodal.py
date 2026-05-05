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

    VALID_MODES = ('concat', 'per_modality', 'video_only', 'audio_only')

    def __init__(
        self,
        audio_dir: Optional[str] = None,
        ceiling: Optional[float] = None,
        reliability_threshold: Optional[float] = None,
        identifier_suffix: str = '-multimodal',
        mode: str = 'concat',
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

    @staticmethod
    def _read_stimulus_ids(assembly):
        """Read stimulus_id from the presentation axis whether it's a
        MultiIndex level or a plain coord. AudioWrapper resets the
        presentation MultiIndex during meta attachment (to avoid level-
        name collisions); VideoWrapper preserves it. Both paths land on
        the same canonical stimulus_id values."""
        if 'presentation' in assembly.indexes:
            idx = assembly.indexes['presentation']
            if hasattr(idx, 'get_level_values'):
                return list(idx.get_level_values('stimulus_id'))
        return list(assembly['stimulus_id'].values)

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
        """Extract video + audio features from the candidate.

        Returns:
            features: (n_videos, n_features) numpy array, video features
                first, audio features second.
            clip_ids: list of stimulus_ids in the same order as features.
            modality_per_neuroid: ndarray of {'video', 'audio'} strings,
                one per output feature column. Stored on the score as
                ``modality_split`` for downstream introspection.
        """
        candidate.start_recording('IT', time_bins=[(0, VIDEO_DURATION_MS)])
        video_assembly = candidate.process(self._video_stim_set())
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
        if 'audio' not in getattr(candidate, 'supported_modalities', set()):
            raise ValueError(
                f"{self.identifier} requires the candidate to support both "
                f"'video' and 'audio' modalities. Got "
                f"supported_modalities={candidate.supported_modalities}. "
                f"Use Lahner2024-fMRI-naturalistic for video-only models."
            )
        if 'video' not in candidate.supported_modalities:
            raise ValueError(
                f"{self.identifier} requires the candidate to support both "
                f"'video' and 'audio' modalities. Got "
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
        # Per-modality (or single-group) ridge: fit one Ridge per group on
        # train, predict on test, sum predictions across groups. With one
        # group ('concat' / 'video_only' / 'audio_only') this collapses to
        # the simple ridge call. Summing predictions across separately-fit
        # models is equivalent to fitting independent encoders and adding
        # their outputs — no cross-modality regularization competition.
        fold_preds = np.zeros_like(neural_mat)
        for train_idx, test_idx in kf.split(np.arange(n)):
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
        score.attrs['n_features_video'] = int(
            (modality_per_neuroid == 'video').sum())
        score.attrs['n_features_audio'] = int(
            (modality_per_neuroid == 'audio').sum())
        if mask is not None:
            score.attrs['voxel_mask_n_total'] = int(mask.size)
            score.attrs['voxel_mask_n_kept'] = int(mask.sum())
            score.attrs['reliability_threshold'] = float(
                self._reliability_threshold)
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
