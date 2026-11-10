"""TR-resolved multimodal Lahner2024 — first benchmark in the unified
interface where two-modality features are aligned to the brain's TR
grid via onset placement / HRF / per-TR ridge.

What this validates that the GLM-beta multimodal variant cannot:
- Per-modality clip-mean features placed as impulses at each clip's onset
  TR (``floor(onset_sec / TR)``), HRF-convolved within each run. Native
  time bins are averaged first, so within-clip dynamics are not modelled
  (a clip-mean onset-impulse HRF model; see ``benchmark_timeresolved``).
- Both modalities share the SAME TR axis after alignment
- Continuous-time encoding still keeps per-modality feature columns

Pipeline:
1. For each unique stimulus (3-s clip): extract video features +
   audio features separately.
   - Video: V-JEPA (or other video backbone) with `time_series` mode →
     mean over the 8-tubelet axis to get one feature vector per clip.
   - Audio: Wav2Vec2 (or other audio backbone) with `time_series` mode →
     mean over the ~150-frame axis to get one feature vector per clip.
   - Concatenate to get one (n_features_v + n_features_a) vector per clip.
2. Place per-clip features at clip-onset TRs via the events sidecar
   (already used by the unimodal TR-resolved benchmark).
3. HRF-convolve per run.
4. Per-voxel Ridge (or banded ridge with per-group α) at TR grain.

Mode parameter (matches the GLM-beta multimodal benchmark):
- 'concat': one ridge on [video|audio] flat features, α=1.0; uses the
  parent's outer CV and honours ``cv_mode`` ('subject_out' default, or
  'within_subject')
- 'per_modality': two independent ridges, predictions summed
- 'video_only' / 'audio_only': single-tower upper bounds
- 'banded': per-group α chosen on a random 20% of the outer-training ROWS
  (rows of one run fall on both sides of this inner split)

The four non-concat modes use one outer CV: shuffled K-fold over whole runs
with all subjects pooled (``cv_mode='pooled_run_kfold'``). They do not
implement 'subject_out' or 'within_subject' and refuse those values.

Banded ridge is the recommended default for any multimodal scoring —
flat α across asymmetric-utility modalities consistently underperforms
(see CLAUDE.md milestone for the GLM-beta visual-ROI evidence).

For naturalistic CONTINUOUS stimuli (Sherlock / Forrest / NNDb), this
same machinery generalizes: replace clip-onset placement with
per-TR-onset features extracted via sliding-window inference. That's
the M12-full extension. This benchmark is the validation step before
that work.
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from brainscore_core.metrics import Score

from brainscore.benchmarks._scoring_utils import (
    run_kfold_masks,
    valid_prediction_per_unit_pearson,
)
from brainscore.tools.banded_ridge import (
    banded_ridge_fit_predict,
    ridge_fit_predict,
)
from ._util import (
    read_stimulus_ids, record_clips, record_frames_per_clip, takes_audiovisual,
)

from .benchmark_timeresolved import (
    Lahner2024BOLDMoments_timeresolved,
    FEATURE_DIM_CAP,
    CURRENT_VERSION,
    TRANSDUCTIVE_SVD,
)
from .benchmark_multimodal import DEFAULT_AUDIO_DIR


class Lahner2024BOLDMoments_timeresolved_multimodal(
        Lahner2024BOLDMoments_timeresolved):
    """TR-resolved variant that extracts video AND audio per stimulus
    and places both at clip-onset TRs."""

    VALID_MODES = ('concat', 'per_modality', 'video_only', 'audio_only',
                   'banded')
    # Outer CV of the non-concat modes: shuffled K-fold over whole runs,
    # subjects pooled. It is the only policy those modes implement.
    POOLED_RUN_KFOLD = 'pooled_run_kfold'
    BANDED_ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)

    def __init__(
        self,
        audio_dir: Optional[str] = None,
        ceiling: Optional[float] = None,
        cv_n_held_out_runs: int = 52,
        reliability_threshold: Optional[float] = None,
        identifier_suffix: str = '-timeresolved-multimodal',
        apply_motion_regression: bool = False,
        cv_mode: Optional[str] = None,
        within_subject_n_held_out: int = 10,
        mode: str = 'concat',
        voxel_mask_fn: Optional[callable] = None,
        version: int = CURRENT_VERSION,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}; got {mode!r}.")
        # cv_mode=None picks the policy the mode implements. Non-concat modes
        # implement only pooled run K-fold; asking them for subject_out or
        # within_subject is refused here, before any compute (F1).
        if mode == 'concat':
            parent_cv_mode = cv_mode or 'subject_out'
        elif cv_mode in (None, self.POOLED_RUN_KFOLD):
            parent_cv_mode = 'subject_out'   # unused by the non-concat fit
        else:
            raise ValueError(
                f"mode={mode!r} uses shuffled K-fold over whole runs with all "
                f"subjects pooled; it does not implement cv_mode={cv_mode!r}. "
                f"Use mode='concat' for 'subject_out' / 'within_subject', or "
                f"pass cv_mode=None or {self.POOLED_RUN_KFOLD!r}.")
        super().__init__(
            ceiling=ceiling,
            cv_n_held_out_runs=cv_n_held_out_runs,
            reliability_threshold=reliability_threshold,
            identifier_suffix=identifier_suffix,
            apply_motion_regression=apply_motion_regression,
            cv_mode=parent_cv_mode,
            within_subject_n_held_out=within_subject_n_held_out,
            version=version,
        )
        if mode != 'concat':
            self._cv_mode = self.POOLED_RUN_KFOLD
        # Multimodal A+V: audio hard-required; visual tower accepts video OR
        # vision (accepted_modalities inherited from the timeresolved base).
        self.required_modalities = {'audio'}
        from pathlib import Path
        self._audio_dir = Path(audio_dir or DEFAULT_AUDIO_DIR).expanduser()
        self._mode = mode
        self._voxel_mask_fn = voxel_mask_fn
        # Modality-tagged feature columns — populated during extraction;
        # used by banded / per_modality / single-tower modes to slice.
        self._modality_per_feature: Optional[np.ndarray] = None

    # ── Voxel mask: support custom function (auditory-ROI etc.) ──────

    @property
    def voxel_mask(self) -> Optional[np.ndarray]:
        if self._voxel_mask_fn is None:
            return super().voxel_mask
        if self._voxel_mask is None:
            mask = self._voxel_mask_fn()
            self._voxel_mask = np.asarray(mask, dtype=bool)
        return self._voxel_mask

    # ── Per-modality stim sets (mirrors GLM-beta variant) ───────────

    def _video_stim_set_for(self, unique_ids):
        """One row per stimulus_id with ``video_path`` column."""
        from brainscore_core.supported_data_standards.brainio.stimuli import (
            StimulusSet)
        full_stim = self._stim_helper.stimulus_set
        rows = []
        for sid in unique_ids:
            video_path = full_stim.get_stimulus(sid)
            rows.append({
                'stimulus_id': sid,
                'video_path': str(video_path),
            })
        df = pd.DataFrame(rows)
        out = StimulusSet(df)
        out.identifier = (
            f'{full_stim.identifier}-timeresolved-multimodal-video')
        out.stimulus_paths = dict(zip(df['stimulus_id'], df['video_path']))
        return out

    def _audio_stim_set_for(self, unique_ids):
        """One row per stimulus_id with ``audio_path`` column. Locates
        WAV files by stimulus number under self._audio_dir."""
        from pathlib import Path
        from brainscore_core.supported_data_standards.brainio.stimuli import (
            StimulusSet)
        full_stim = self._stim_helper.stimulus_set
        rows = []
        missing = []
        for sid in unique_ids:
            video_path = Path(full_stim.get_stimulus(sid))
            audio_path = self._audio_dir / f'{video_path.stem}.wav'
            if not audio_path.exists():
                missing.append(str(audio_path))
            rows.append({
                'stimulus_id': sid,
                'audio_path': str(audio_path),
            })
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} audio files missing under {self._audio_dir}. "
                f"Run `python -m brainscore.data.lahner2024."
                f"prepare_audio_tracks --target-rate 16000` first. "
                f"First missing: {missing[0]}")
        df = pd.DataFrame(rows)
        out = StimulusSet(df)
        out.identifier = (
            f'{full_stim.identifier}-timeresolved-multimodal-audio')
        out.stimulus_paths = dict(zip(df['stimulus_id'], df['audio_path']))
        return out

    # ── Override per-stim feature extraction for dual modality ──────

    def _extract_per_stimulus_features(self, candidate):
        """Extract per-modality features per unique stimulus, concat
        along feature axis, and tag each column with its modality so
        downstream modes (banded / per_modality / single-tower) can
        slice."""
        # Determine which stimulus IDs we need
        unique_event_ids = set(self.events['stimulus_id'].unique().tolist())
        full_stim = self._stim_helper.stimulus_set
        stim_set_ids = set(full_stim['stimulus_id'].astype(str).tolist())
        unique_ids = sorted(unique_event_ids & stim_set_ids)

        # Accept audio-visual subjects, routed on declared channels only
        # (Contract Rule 2.2): a visual input (clips natively, or stills shown
        # as frames), audio, and the neural:IT and neural:A1 outputs. Any
        # other subject falls back to the parent's single-tower extraction.
        if not takes_audiovisual(candidate):
            self._modality_per_feature = None
            return super()._extract_per_stimulus_features(candidate)

        # ── Visual tower ───────────────────────────────────────────
        from .benchmark import VIDEO_DURATION_MS
        time_bins = [(0, VIDEO_DURATION_MS)]
        # Clips natively when the subject declares vision:video, frames otherwise.
        from .benchmark import Lahner2024BOLDMoments
        if Lahner2024BOLDMoments._is_native_video(candidate):
            video_stim = self._video_stim_set_for(unique_ids)
            video_assembly = record_clips(candidate, video_stim, 'IT', time_bins)
        else:
            # Frame-aggregation: expand into N frames, show them as stills,
            # then mean-pool over time bins.
            frame_stim = self._stim_helper._expand_videos()
            frame_stim = frame_stim[frame_stim['clip_id'].isin(unique_ids)]
            video_assembly = record_frames_per_clip(
                candidate, frame_stim, 'IT', time_bins)
        v_data = video_assembly.values
        if v_data.ndim == 3:    # (presentation, time_bin, neuroid)
            v_features = v_data.mean(axis=1)
        else:
            v_features = v_data
        v_stim_ids = self._read_stimulus_ids(video_assembly)

        # ── Audio tower ────────────────────────────────────────────
        audio_stim = self._audio_stim_set_for(unique_ids)
        audio_assembly = record_clips(candidate, audio_stim, 'A1', time_bins)
        a_data = audio_assembly.values
        if a_data.ndim == 3:
            a_features = a_data.mean(axis=1)
        else:
            a_features = a_data
        a_stim_ids = self._read_stimulus_ids(audio_assembly)

        # Align audio order to video order
        if v_stim_ids != a_stim_ids:
            order = {s: i for i, s in enumerate(a_stim_ids)}
            perm = [order[s] for s in v_stim_ids]
            a_features = a_features[perm]

        # Cap each modality independently before concat (the parent's
        # FEATURE_DIM_CAP would otherwise mix modalities under SVD). Each SVD
        # is fit on all stimuli (transductive); recorded on the score.
        self._feature_projection = 'none'
        if max(v_features.shape[1], a_features.shape[1]) > FEATURE_DIM_CAP:
            self._feature_projection = TRANSDUCTIVE_SVD + '_per_modality'
        if v_features.shape[1] > FEATURE_DIM_CAP:
            from sklearn.decomposition import TruncatedSVD
            v_features = TruncatedSVD(
                n_components=FEATURE_DIM_CAP,
                random_state=0).fit_transform(v_features).astype(np.float32)
        if a_features.shape[1] > FEATURE_DIM_CAP:
            from sklearn.decomposition import TruncatedSVD
            a_features = TruncatedSVD(
                n_components=FEATURE_DIM_CAP,
                random_state=0).fit_transform(a_features).astype(np.float32)

        # Concat + tag
        features = np.concatenate([v_features, a_features], axis=1)
        self._modality_per_feature = np.concatenate([
            np.array(['video'] * v_features.shape[1]),
            np.array(['audio'] * a_features.shape[1]),
        ])
        return features.astype(np.float32), v_stim_ids

    _read_stimulus_ids = staticmethod(read_stimulus_ids)

    # ── Override scoring to support per-modality / banded modes ─────

    def __call__(self, candidate) -> Score:
        # Defer to the parent for the single-tower path; only the
        # multimodal path with mode != 'concat' needs custom handling.
        # For 'concat' (parent default), the parent's ridge fits the
        # full concat matrix with α=1.0 — which is what we want.
        if self._mode == 'concat':
            score = super().__call__(candidate)
            self._tag_score(score, candidate)
            return score

        # For other modes we replicate the parent's pipeline but slice
        # / fit ridge differently. To avoid duplicating ~200 lines, we
        # reuse the parent's feature placement + HRF + flatten path.
        return self._call_with_mode(candidate)

    def _tag_score(self, score, candidate):
        """Attach modality breakdown attrs once the score is computed."""
        if self._modality_per_feature is not None:
            score.attrs['n_features_video'] = int(
                (self._modality_per_feature == 'video').sum())
            score.attrs['n_features_audio'] = int(
                (self._modality_per_feature == 'audio').sum())
        score.attrs['mode'] = self._mode
        score.attrs['pipeline'] = (
            f'tr_resolved_multimodal_{self._mode}')
        if self._mode == 'banded':
            score.attrs['inner_validation'] = (
                'random_rows_20pct_runs_shared_with_inner_training')
            score.attrs['banded_alpha_grid'] = tuple(self.BANDED_ALPHA_GRID)
        return score

    def _call_with_mode(self, candidate) -> Score:
        # Same as parent.__call__ up to the X_flat / Y_flat point
        per_stim_features, stimulus_ids = (
            self._extract_per_stimulus_features(candidate))

        assembly = self.assembly
        n_runs = assembly.sizes['presentation']
        feature_ts_convolved = self._build_design(
            per_stim_features, stimulus_ids, assembly)

        X_flat, Y_flat, run_idx_per_obs, subject_per_obs = (
            self._concatenate_with_mask(feature_ts_convolved, assembly))

        # Slice features per mode
        v_mask_feat = self._modality_per_feature == 'video'
        a_mask_feat = self._modality_per_feature == 'audio'
        if self._mode == 'video_only':
            X_flat = X_flat[:, v_mask_feat]
            feature_groups = [X_flat]
        elif self._mode == 'audio_only':
            X_flat = X_flat[:, a_mask_feat]
            feature_groups = [X_flat]
        elif self._mode == 'per_modality':
            feature_groups = [X_flat[:, v_mask_feat], X_flat[:, a_mask_feat]]
        elif self._mode == 'banded':
            feature_groups = [X_flat[:, v_mask_feat], X_flat[:, a_mask_feat]]
        else:
            feature_groups = [X_flat]

        # Fit per mode. For 'banded' we use the BG-mod ridge from the
        # GLM-beta variant. For 'per_modality' we sum separate fits.
        per_voxel_r = self._fit_score_with_groups(
            feature_groups, Y_flat, run_idx_per_obs, subject_per_obs)

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
        # CV labels for the policy this path actually applies (F1).
        score.attrs['n_runs'] = n_runs
        score.attrs['n_observations'] = int(len(Y_flat))
        score.attrs['cv_mode'] = self._cv_mode
        score.attrs['cv_policy'] = 'shuffled_kfold_over_pooled_runs'
        score.attrs['cv_n_held_out_runs'] = self._cv_n_held_out_runs
        score.attrs['cv_n_splits'] = self._n_pooled_splits(run_idx_per_obs)
        score.attrs['cv_blocks_hold_out_whole_subjects'] = False
        score.attrs['motion_regressed'] = bool(self._apply_motion_regression)
        score.attrs.update(self._method_attrs())
        score.attrs['ridge_alpha'] = (
            'per_group_from_grid' if self._mode == 'banded' else 1.0)
        score.attrs['inner_validation'] = 'none_fixed_alpha'
        self._tag_score(score, candidate)
        return score

    def _n_pooled_splits(self, run_idx_per_obs) -> int:
        n_runs = len(np.unique(run_idx_per_obs))
        return max(2, n_runs // self._cv_n_held_out_runs)

    def _fit_score_with_groups(self, feature_groups, Y_flat,
                               run_idx_per_obs, subject_per_obs):
        """Mode-specific ridge fit. Returns per-voxel Pearson r.

        Voxel-batched (VOXEL_BATCH=2000) to keep peak memory bounded —
        same pattern as the parent class's _cross_validated_ridge.
        Without this, scoring on the 4042-voxel visualROI variant OOMs
        around the held_out_preds shadow array + sklearn's per-fold
        Y copies (CLAUDE.md note from M12-lite).
        """
        n_obs, n_voxels = Y_flat.shape
        # Outer CV: shuffled K-fold over whole runs, subjects pooled
        # (cv_mode='pooled_run_kfold'). subject_per_obs is not used.
        fold_masks = list(run_kfold_masks(
            run_idx_per_obs, n_splits=self._n_pooled_splits(run_idx_per_obs),
            random_state=0))

        per_voxel_r = np.full(n_voxels, np.nan, dtype=np.float32)
        VOXEL_BATCH = 2000
        for v_start in range(0, n_voxels, VOXEL_BATCH):
            v_end = min(v_start + VOXEL_BATCH, n_voxels)
            Y_sub = Y_flat[:, v_start:v_end]
            held_out_preds = np.full_like(Y_sub, np.nan, dtype=np.float32)

            for tr_mask, te_mask in fold_masks:
                Y_tr = Y_sub[tr_mask]
                if self._mode == 'banded':
                    preds = self._banded_fit_predict(
                        feature_groups, tr_mask, te_mask, Y_tr)
                    held_out_preds[te_mask] = preds.astype(np.float32)
                else:
                    preds = np.zeros((te_mask.sum(), v_end - v_start),
                                     dtype=np.float32)
                    for group in feature_groups:
                        preds += ridge_fit_predict(
                            group[tr_mask], Y_tr, group[te_mask], alpha=1.0)
                    held_out_preds[te_mask] = preds

            per_voxel_r[v_start:v_end] = valid_prediction_per_unit_pearson(
                Y_sub, held_out_preds)

        return per_voxel_r

    def _banded_fit_predict(self, feature_groups, tr_mask, te_mask, Y_tr):
        """Banded ridge over (video, audio) feature groups — see
        ``tools.banded_ridge``. Masks are applied here; row-indexing
        commutes with the column-concat done inside the shared helper, so
        this is equivalent to stacking-then-masking."""
        train_groups = [g[tr_mask] for g in feature_groups]
        test_groups = [g[te_mask] for g in feature_groups]
        preds, _ = banded_ridge_fit_predict(
            train_groups, test_groups, Y_tr, self.BANDED_ALPHA_GRID)
        return preds


# ── Factory functions for the registered variants ─────────────────


def Lahner2024BOLDMoments_timeresolved_multimodal_visualROI(
        audio_dir: Optional[str] = None, mode: str = 'concat',
        version: int = CURRENT_VERSION):
    return Lahner2024BOLDMoments_timeresolved_multimodal(
        audio_dir=audio_dir,
        reliability_threshold=0.3,
        identifier_suffix='-timeresolved-multimodal-visualROI',
        mode=mode,
        version=version,
    )


def Lahner2024BOLDMoments_timeresolved_multimodal_auditoryROI(
        audio_dir: Optional[str] = None, mode: str = 'concat',
        version: int = CURRENT_VERSION):
    from .auditory_roi import build_auditory_mask
    return Lahner2024BOLDMoments_timeresolved_multimodal(
        audio_dir=audio_dir,
        identifier_suffix='-timeresolved-multimodal-auditoryROI',
        mode=mode,
        voxel_mask_fn=build_auditory_mask,
        version=version,
    )
