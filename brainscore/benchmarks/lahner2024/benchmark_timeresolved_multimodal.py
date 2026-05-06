"""TR-resolved multimodal Lahner2024 — first benchmark in the unified
interface where two-modality features are aligned to the brain's TR
grid via temporal_bin / HRF / per-TR ridge.

What this validates that the GLM-beta multimodal variant cannot:
- Per-modality features placed at TR onsets (not at clip-mean)
- Both modalities share the SAME time_bin axis after alignment, so
  ``multi_modality=True`` would actually concat cleanly along neuroid
- Continuous-time encoding still respects per-modality features

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
- 'concat': one ridge on [video|audio] flat features, α=1.0
- 'per_modality': two independent ridges, predictions summed
- 'video_only' / 'audio_only': single-tower upper bounds
- 'banded': per-group α via 20% inner-fold hold-out grid search

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

from .benchmark_timeresolved import (
    Lahner2024BOLDMoments_timeresolved,
    FEATURE_DIM_CAP,
    TR_SEC,
)
from .benchmark_multimodal import DEFAULT_AUDIO_DIR


class Lahner2024BOLDMoments_timeresolved_multimodal(
        Lahner2024BOLDMoments_timeresolved):
    """TR-resolved variant that extracts video AND audio per stimulus
    and places both at clip-onset TRs."""

    VALID_MODES = ('concat', 'per_modality', 'video_only', 'audio_only',
                   'banded')
    BANDED_ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)

    def __init__(
        self,
        audio_dir: Optional[str] = None,
        ceiling: Optional[float] = None,
        cv_n_held_out_runs: int = 52,
        reliability_threshold: Optional[float] = None,
        identifier_suffix: str = '-timeresolved-multimodal',
        apply_motion_regression: bool = False,
        cv_mode: str = 'subject_out',
        within_subject_n_held_out: int = 10,
        mode: str = 'concat',
        voxel_mask_fn: Optional[callable] = None,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}; got {mode!r}.")
        super().__init__(
            ceiling=ceiling,
            cv_n_held_out_runs=cv_n_held_out_runs,
            reliability_threshold=reliability_threshold,
            identifier_suffix=identifier_suffix,
            apply_motion_regression=apply_motion_regression,
            cv_mode=cv_mode,
            within_subject_n_held_out=within_subject_n_held_out,
        )
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
                f"Run prepare_audio_tracks.py with --target-rate 16000 "
                f"first. First missing: {missing[0]}")
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

        # Accept dual-modality models — visual tower can be 'video'
        # (native-temporal) or 'vision' (frame-aggregation). For
        # unimodal models, fall back to the parent class's
        # single-modality extraction.
        supports = getattr(candidate, 'supported_modalities', set())
        has_visual = 'video' in supports or 'vision' in supports
        if not has_visual or 'audio' not in supports:
            self._modality_per_feature = None
            return super()._extract_per_stimulus_features(candidate)

        # ── Visual tower ───────────────────────────────────────────
        from .benchmark import VIDEO_DURATION_MS
        from brainscore_core.temporal import temporal_bin
        candidate.start_recording('IT', time_bins=[(0, VIDEO_DURATION_MS)])
        if 'video' in supports:
            video_stim = self._video_stim_set_for(unique_ids)
            video_assembly = candidate.process(video_stim)
        else:
            # Frame-aggregation: expand into N frames, run still-image
            # path, then mean-pool over time bins.
            frame_stim = self._stim_helper._expand_videos()
            frame_stim = frame_stim[frame_stim['clip_id'].isin(unique_ids)]
            per_frame = candidate.process(frame_stim)
            video_assembly = temporal_bin(
                per_frame, time_bins=[(0, VIDEO_DURATION_MS)])
        v_data = video_assembly.values
        if v_data.ndim == 3:    # (presentation, time_bin, neuroid)
            v_features = v_data.mean(axis=1)
        else:
            v_features = v_data
        v_stim_ids = self._read_stimulus_ids(video_assembly)

        # ── Audio tower ────────────────────────────────────────────
        candidate.start_recording('A1', time_bins=[(0, VIDEO_DURATION_MS)])
        audio_stim = self._audio_stim_set_for(unique_ids)
        audio_assembly = candidate.process(audio_stim)
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
        # FEATURE_DIM_CAP would otherwise mix modalities under SVD).
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

    @staticmethod
    def _read_stimulus_ids(assembly):
        """Same logic as the GLM-beta multimodal benchmark — handles
        both native-video assemblies (use 'stimulus_id') and frame-
        aggregation assemblies from temporal_bin (use 'clip_id')."""
        if 'presentation' in assembly.indexes:
            idx = assembly.indexes['presentation']
            if hasattr(idx, 'get_level_values'):
                names = list(idx.names) if hasattr(idx, 'names') else []
                if 'stimulus_id' in names:
                    return list(idx.get_level_values('stimulus_id'))
                if 'clip_id' in names:
                    return list(idx.get_level_values('clip_id'))
        for col in ('stimulus_id', 'clip_id'):
            if col in assembly.coords:
                return list(assembly[col].values)
        raise KeyError(
            f"assembly has neither 'stimulus_id' nor 'clip_id' on its "
            f"presentation axis; coords={list(assembly.coords)}")

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
        return score

    def _call_with_mode(self, candidate) -> Score:
        from brainscore_core.temporal import double_gamma_hrf, hrf_convolve

        # Same as parent.__call__ up to the X_flat / Y_flat point
        per_stim_features, stimulus_ids = (
            self._extract_per_stimulus_features(candidate))

        stim_idx = {sid: i for i, sid in enumerate(stimulus_ids)}
        assembly = self.assembly
        n_TR_max = assembly.sizes['time_bin']
        n_features = per_stim_features.shape[1]
        n_runs = assembly.sizes['presentation']

        feature_ts = np.zeros((n_runs, n_TR_max, n_features), dtype=np.float32)
        for run_idx in range(n_runs):
            run_id = (str(assembly['subject'].values[run_idx]),
                      str(assembly['session'].values[run_idx]),
                      str(assembly['run'].values[run_idx]))
            run_events = self._events_for_run(*run_id)
            for _, ev in run_events.iterrows():
                if ev['stimulus_id'] not in stim_idx:
                    continue
                onset_TR = int(np.floor(ev['onset_sec'] / TR_SEC))
                if 0 <= onset_TR < n_TR_max:
                    feature_ts[run_idx, onset_TR, :] += (
                        per_stim_features[stim_idx[ev['stimulus_id']]])

        hrf_kernel = double_gamma_hrf(
            duration_sec=32.0, sampling_rate_hz=1.0/TR_SEC)
        feature_ts_convolved = np.zeros_like(feature_ts)
        for run_idx in range(n_runs):
            feature_ts_convolved[run_idx] = hrf_convolve(
                feature_ts[run_idx],
                sampling_rate_hz=1.0/TR_SEC, hrf=hrf_kernel)

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
        score.attrs['mean_r'] = mean_r
        score.attrs['n_voxels_scored'] = int(len(per_voxel_r_finite))
        self._tag_score(score, candidate)
        return score

    def _fit_score_with_groups(self, feature_groups, Y_flat,
                               run_idx_per_obs, subject_per_obs):
        """Mode-specific ridge fit. Returns per-voxel Pearson r."""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold

        # Use the same CV strategy as the parent; we just change the
        # fit logic per mode.
        n_voxels = Y_flat.shape[1]
        per_voxel_r = np.full(n_voxels, np.nan, dtype=np.float64)
        unique_runs = np.unique(run_idx_per_obs)
        n_runs = len(unique_runs)
        n_held = self._cv_n_held_out_runs

        kf = KFold(n_splits=max(2, n_runs // n_held), shuffle=True,
                   random_state=0)
        fold_preds = np.zeros_like(Y_flat)
        for train_runs, test_runs in kf.split(unique_runs):
            train_run_ids = unique_runs[train_runs]
            test_run_ids = unique_runs[test_runs]
            tr_mask = np.isin(run_idx_per_obs, train_run_ids)
            te_mask = np.isin(run_idx_per_obs, test_run_ids)
            Y_tr = Y_flat[tr_mask]
            Y_te = Y_flat[te_mask]
            if self._mode == 'banded':
                # Single shared (α_v, α_a) tuned via 20% inner-train hold-out
                preds = self._banded_fit_predict(
                    feature_groups, tr_mask, te_mask, Y_tr)
                fold_preds[te_mask] = preds
            else:
                # Concat / per_modality / single-tower
                preds = np.zeros_like(Y_te)
                for group in feature_groups:
                    Xg_tr = group[tr_mask]
                    Xg_te = group[te_mask]
                    reg = Ridge(alpha=1.0).fit(Xg_tr, Y_tr)
                    preds += reg.predict(Xg_te)
                fold_preds[te_mask] = preds

        # Per-voxel Pearson on held-out predictions
        for j in range(n_voxels):
            yt = Y_flat[:, j]
            yp = fold_preds[:, j]
            if yt.std() > 0 and yp.std() > 0:
                per_voxel_r[j] = np.corrcoef(yt, yp)[0, 1]
        return per_voxel_r

    def _banded_fit_predict(self, feature_groups, tr_mask, te_mask, Y_tr):
        rng = np.random.default_rng(0)
        n_tr = int(tr_mask.sum())
        n_val = max(1, int(round(n_tr * 0.2)))
        perm = rng.permutation(n_tr)
        val_idx = perm[:n_val]
        inner_tr_idx = perm[n_val:]
        sizes = [g.shape[1] for g in feature_groups]

        def _stack(groups, mask, sub_idx=None):
            X = np.concatenate([g[mask] for g in groups], axis=1)
            if sub_idx is not None:
                X = X[sub_idx]
            return X

        X_inner_tr = _stack(feature_groups, tr_mask, inner_tr_idx)
        X_val = _stack(feature_groups, tr_mask, val_idx)
        Y_inner_tr = Y_tr[inner_tr_idx]
        Y_val = Y_tr[val_idx]

        # Hoist the expensive (n_obs × n_feat × n_feat) matmul outside the
        # α-grid loop — only the diag(λ) penalty changes per (α_v, α_a).
        # On a 100k-obs × 1024-feat problem this drops banded runtime
        # ~25× (one matmul + n_grid solves vs n_grid matmuls).
        XtX_inner = X_inner_tr.T @ X_inner_tr
        XtY_inner = X_inner_tr.T @ Y_inner_tr
        Y_val_centered = Y_val - Y_val.mean(axis=0)
        Y_val_var = (Y_val_centered ** 2).sum(axis=0)

        best_score = -np.inf
        best_alpha = (1.0, 1.0)
        for av in self.BANDED_ALPHA_GRID:
            for aa in self.BANDED_ALPHA_GRID:
                lam = np.concatenate([
                    np.full(sizes[0], av),
                    np.full(sizes[1], aa),
                ])
                W = np.linalg.solve(XtX_inner + np.diag(lam), XtY_inner)
                Y_val_pred = X_val @ W
                yp = Y_val_pred - Y_val_pred.mean(axis=0)
                num = (Y_val_centered * yp).sum(axis=0)
                den = np.sqrt(Y_val_var * (yp ** 2).sum(axis=0))
                with np.errstate(divide='ignore', invalid='ignore'):
                    r = np.where(den > 0, num / den, 0.0)
                s = float(np.mean(r))
                if s > best_score:
                    best_score = s
                    best_alpha = (av, aa)

        # Refit on full training fold with chosen α (one matmul, one solve).
        av, aa = best_alpha
        lam = np.concatenate([
            np.full(sizes[0], av),
            np.full(sizes[1], aa),
        ])
        X_full_tr = _stack(feature_groups, tr_mask)
        X_te = _stack(feature_groups, te_mask)
        XtX = X_full_tr.T @ X_full_tr
        XtY = X_full_tr.T @ Y_tr
        W = np.linalg.solve(XtX + np.diag(lam), XtY)
        return X_te @ W


# ── Factory functions for the registered variants ─────────────────


def Lahner2024BOLDMoments_timeresolved_multimodal_visualROI(
        audio_dir: Optional[str] = None, mode: str = 'concat'):
    return Lahner2024BOLDMoments_timeresolved_multimodal(
        audio_dir=audio_dir,
        reliability_threshold=0.3,
        identifier_suffix='-timeresolved-multimodal-visualROI',
        mode=mode,
    )


def Lahner2024BOLDMoments_timeresolved_multimodal_auditoryROI(
        audio_dir: Optional[str] = None, mode: str = 'concat'):
    from .auditory_roi import build_auditory_mask
    return Lahner2024BOLDMoments_timeresolved_multimodal(
        audio_dir=audio_dir,
        identifier_suffix='-timeresolved-multimodal-auditoryROI',
        mode=mode,
        voxel_mask_fn=build_auditory_mask,
    )
