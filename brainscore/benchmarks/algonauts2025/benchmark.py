"""Algonauts 2025 challenge benchmark classes.

Three subclasses share most logic:
- Algonauts2025Friends: train/test on Friends S1-S6 + Movie10 (in-dist).
  __call__ scores the candidate via per-TR frame extraction + ridge.
- Algonauts2025FriendsS7: held-out S7 — __call__ returns predicted
  per-parcel time series in the format Codabench expects.
- Algonauts2025OOD: held-out 2 h of OOD movies — same as S7 but
  different stim_set.

Phase 3 (current): video-only frame-aggregation. Extract one frame per
TR midpoint, run candidate's vision tower, stack stimulus_window TRs
of context per fMRI sample, HRF-shift, ridge-regress per subject,
per-parcel Pearson median. Multimodal (audio + transcript) and
banded-ridge α tuning come in later phases.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score


BIBTEX = """@article{gifford2025algonauts,
  title={The Algonauts Project 2025 Challenge: How the Human Brain Makes Sense of Multimodal Movies},
  author={Gifford, Alessandro T and Bersch, Domenic and St-Laurent, Marie and Pinsard, Basile
          and Boyle, Julie and Bellec, Pierre and Oliva, Aude and Roig, Gemma and Cichy, Radoslaw M},
  journal={arXiv preprint arXiv:2501.00504},
  year={2025},
}"""

# Schaefer 1000 parcellation, TR for Courtois NeuroMod
SCHAEFER_N_PARCELS = 1000
TR_SEC = 1.49

# Default fMRI assembly cache root (set by prepare_assembly.py).
# Override via the ALGONAUTS_DATA_ROOT environment variable.
DEFAULT_ASSEMBLY_ROOT = Path('~/.brainio/algonauts2025').expanduser()


class _Algonauts2025Base(BenchmarkBase):
    """Shared logic across the three splits."""

    VALID_MODES = ('concat', 'per_modality', 'banded',
                   'video_only', 'audio_only', 'language_only')
    BANDED_ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)

    def __init__(
        self,
        subject: int,
        split: str,
        identifier_suffix: str,
        stimulus_window: int = 5,
        hrf_delay: int = 3,
        excluded_samples_start: int = 5,
        excluded_samples_end: int = 5,
        mode: str = 'banded',
        assembly_root: Optional[Path] = None,
    ):
        if subject not in (1, 2, 3, 5):
            raise ValueError(
                f"Algonauts subjects are {{1, 2, 3, 5}}; got {subject}.")
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}; got {mode!r}.")

        self._subject = subject
        self._split = split
        self._stimulus_window = stimulus_window
        self._hrf_delay = hrf_delay
        self._excluded_samples_start = excluded_samples_start
        self._excluded_samples_end = excluded_samples_end
        self._mode = mode
        self._assembly_root = (Path(assembly_root)
                               if assembly_root else DEFAULT_ASSEMBLY_ROOT)
        self._assembly = None
        self._stimulus_set = None

        super().__init__(
            identifier=(f'Algonauts2025-{split}-sub{subject:02d}'
                        f'{identifier_suffix}'),
            version=1,
            parent='naturalistic',
            ceiling=Score(1.0),
            bibtex=BIBTEX,
        )

    @property
    def assembly(self):
        if self._assembly is None:
            self._assembly = self._load_assembly()
        return self._assembly

    @property
    def stimulus_set(self):
        if self._stimulus_set is None:
            self._stimulus_set = self._load_stimulus_set()
        return self._stimulus_set

    def _load_assembly(self):
        """Load this subject's per-TR Schaefer parcels for this split.

        Expects prepare_assembly.py to have produced a netCDF at
        ``{assembly_root}/algonauts2025_{split}_sub{subject:02d}.nc``.
        Raises FileNotFoundError if the data hasn't been prepared yet.
        """
        path = (self._assembly_root /
                f'algonauts2025_{self._split}_sub{self._subject:02d}.nc')
        if not path.exists():
            raise FileNotFoundError(
                f"Algonauts assembly missing at {path}. Run "
                f"`python -m brainscore.benchmarks.algonauts2025.prepare_assembly` "
                f"on EC2 first; see README.md."
            )
        import xarray as xr
        from brainscore_core.supported_data_standards.brainio.assemblies import (
            NeuronRecordingAssembly)
        data = xr.open_dataarray(str(path))
        return NeuronRecordingAssembly(data)

    def _load_stimulus_set(self):
        """Load the StimulusSet for this split. CSV with movie/episode/split
        rows, plus paths to .mkv stimuli + per-TR transcripts."""
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import (
            StimulusSet)
        path = (self._assembly_root /
                f'algonauts2025_stim_{self._split}.csv')
        if not path.exists():
            raise FileNotFoundError(
                f"Algonauts stim_set missing at {path}. Run "
                f"prepare_assembly.py on EC2 first.")
        df = pd.read_csv(path)
        out = StimulusSet(df)
        out.identifier = f'algonauts2025-{self._split}'
        out.stimulus_paths = dict(
            zip(df['stimulus_id'], df['video_path']))
        return out

    # ── Per-TR frame extraction ───────────────────────────────────

    def _frames_dir(self) -> Path:
        """Cache dir for extracted frames. Idempotent across runs."""
        return self._assembly_root / 'frames'

    def _expand_to_per_TR_frames(self, fps_offset: float = 0.5):
        """Build a frame-level StimulusSet — one row per (stim_id, TR).

        For each unique stimulus_id in this benchmark's stim_set:
        - open the .mkv with cv2
        - sample one frame at TR midpoint (TR_SEC * (t + fps_offset))
        - cache as PNG under {frames_dir}/{stim_id}/{t:04d}.png
        Returns a StimulusSet with columns (stimulus_id, t_within_run,
        frame_id, image_file_name) — frame_id is unique per row.
        """
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import (
            StimulusSet)
        import cv2
        from PIL import Image

        frames_dir = self._frames_dir()
        frames_dir.mkdir(parents=True, exist_ok=True)
        stim_df = self.stimulus_set
        # Sample counts come from the assembly: count TRs per stim_id.
        stim_to_n_TRs = self._stim_id_to_n_TRs()

        rows = []
        for _, srow in stim_df.iterrows():
            stim_id = srow['stimulus_id']
            video_path = Path(srow['video_path'])
            if not video_path.exists():
                continue
            n_TRs = stim_to_n_TRs.get(stim_id)
            if n_TRs is None:
                # No fMRI for this stim — could still happen for held-out
                # stubs (NaN-filled assemblies). Use whatever the assembly
                # says (which may still produce zero rows).
                continue
            stim_frame_dir = frames_dir / stim_id
            stim_frame_dir.mkdir(parents=True, exist_ok=True)
            cap = None
            for t in range(n_TRs):
                out = stim_frame_dir / f'{t:04d}.png'
                if not out.exists():
                    if cap is None:
                        cap = cv2.VideoCapture(str(video_path))
                        if not cap.isOpened():
                            raise IOError(f'cv2 cannot open {video_path}')
                        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                    target_sec = TR_SEC * (t + fps_offset)
                    cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
                    ok, frame = cap.read()
                    if not ok:
                        # Past end of video — fall back to last good frame
                        # and accept duplication for the trailing TRs.
                        cap.set(cv2.CAP_PROP_POS_FRAMES,
                                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
                        ok, frame = cap.read()
                        if not ok:
                            raise IOError(
                                f'cv2 cannot read trailing frame from '
                                f'{video_path} at TR {t}')
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    Image.fromarray(frame_rgb).save(out)
                rows.append({
                    'stimulus_id': stim_id,
                    't_within_run': t,
                    'frame_id': f'{stim_id}_t{t:04d}',
                    'image_file_name': str(out),
                })
            if cap is not None:
                cap.release()
        df = pd.DataFrame(rows)
        out_set = StimulusSet(df)
        out_set.identifier = (
            f'algonauts2025-{self._split}-sub{self._subject:02d}-frames')
        out_set.stimulus_paths = dict(zip(df['frame_id'], df['image_file_name']))
        return out_set

    def _stim_id_to_n_TRs(self) -> Dict[str, int]:
        """Count TRs per stimulus_id from the assembly's presentation
        coord. Both training and held-out stubs carry stimulus_id at
        the right resolution, so this works for all splits."""
        ids = list(self.assembly['stimulus_id'].values)
        from collections import Counter
        return dict(Counter(ids))

    # ── Feature extraction + alignment ────────────────────────────

    def _extract_per_TR_features(self, candidate, frame_stim_set
                                 ) -> Tuple[np.ndarray, List[str]]:
        """Run candidate's vision tower on the frame stim_set.

        Returns:
            features: (n_TRs, n_features) float32
            frame_ids: list of frame_id strings (one per row),
                preserving the order from the assembly walk so
                downstream alignment is straightforward.
        """
        # The candidate is expected to support 'vision' modality. For
        # Phase 3 single-modality scoring we route through the model's
        # own dispatch — same path Lahner-multimodal uses.
        candidate.start_recording('IT', time_bins=[(0, int(TR_SEC * 1000))])
        assembly = candidate.process(frame_stim_set)
        # Flatten any time_bin axis (vision-tower-on-still-image returns
        # (presentation, neuroid) typically, or (presentation, time_bin=1,
        # neuroid)).
        if 'time_bin' in assembly.dims:
            assembly = assembly.mean(dim='time_bin')
        feats = assembly.values.astype(np.float32)
        # Read frame_id ordering from the output assembly (must match
        # frame_stim_set order, which the wrapper preserves).
        if 'frame_id' in assembly.coords:
            frame_ids = list(assembly['frame_id'].values)
        elif 'stimulus_id' in assembly.coords:
            frame_ids = list(assembly['stimulus_id'].values)
        else:
            frame_ids = [f'row_{i}' for i in range(len(assembly))]
        return feats, frame_ids

    def _align_features_to_assembly(
        self, features: np.ndarray, frame_ids: List[str],
        frame_stim_set,
    ) -> np.ndarray:
        """Reorder per-frame features to match the assembly's
        (presentation) order. Each frame_id encodes (stimulus_id,
        t_within_run); the assembly's presentation rows are in the
        order they appear in the .h5 datasets.

        Returns X with shape (n_TRs_in_assembly, n_features).
        """
        # Build frame_id → row index in features
        feat_idx = {fid: i for i, fid in enumerate(frame_ids)}
        # If the wrapper returned 'stimulus_id' instead of 'frame_id',
        # the lookup table is by (stim_id, t) tuple — need to fall back.
        use_tuple_lookup = (frame_ids and frame_ids[0] not in feat_idx
                            and not frame_ids[0].endswith('_t0000'))

        # The frame_stim_set's order may differ from the assembly's.
        # Build (stim_id, t) → frame_id mapping from the stim_set.
        sf = frame_stim_set
        stim_t_to_frame_id = {
            (str(sf.iloc[i]['stimulus_id']),
             int(sf.iloc[i]['t_within_run'])):
            sf.iloc[i]['frame_id']
            for i in range(len(sf))
        }

        n_TRs = self.assembly.sizes['presentation']
        X = np.full((n_TRs, features.shape[1]), np.nan, dtype=np.float32)
        a_stim = list(self.assembly['stimulus_id'].values)
        a_t = list(self.assembly['t_within_run'].values)
        missing = 0
        for i in range(n_TRs):
            fid = stim_t_to_frame_id.get((str(a_stim[i]), int(a_t[i])))
            if fid is None:
                missing += 1
                continue
            row = feat_idx.get(fid)
            if row is None:
                missing += 1
                continue
            X[i] = features[row]
        if missing > 0:
            # Some TRs may lack frames (e.g., assembly TRs past the
            # video's actual duration). Fill with the last valid feature
            # vector to avoid NaNs in the regression.
            print(f'  WARNING: {missing} of {n_TRs} TRs had no frame; '
                  f'filling with last-valid-frame fallback.')
            last_valid = None
            for i in range(n_TRs):
                if not np.isnan(X[i, 0]):
                    last_valid = X[i]
                elif last_valid is not None:
                    X[i] = last_valid
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
        # Output: (n_TRs, W * n_feat) — concat of W stacked feature
        # vectors, with the leftmost being W-1 TRs in the past.
        X_stacked = np.zeros((n_TRs, W * n_feat), dtype=np.float32)
        # For each sample t, we want features at t-D-(W-1), …, t-D.
        for offset in range(W):
            shift = D + (W - 1 - offset)
            for i in range(n_TRs):
                src = i - shift
                if (src >= 0 and run_idx_per_obs[src]
                        == run_idx_per_obs[i]):
                    X_stacked[i, offset*n_feat:(offset+1)*n_feat] = (
                        X[src])
                # else: leave as zero (start of run / before HRF)
        return X_stacked

    # ── Scoring ───────────────────────────────────────────────────

    def __call__(self, candidate) -> Score:
        if self._split != 'friends':
            raise NotImplementedError(
                f"Held-out scoring (split={self._split!r}) not yet "
                f"implemented. Use Algonauts2025Friends for in-distribution "
                f"CV. Phase 3 first ships training-split scoring; held-out "
                f"prediction generation comes next."
            )
        return self._score_friends_train(candidate)

    def _score_friends_train(self, candidate) -> Score:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold

        print(f'  expanding stim_set to per-TR frames...')
        frame_stim_set = self._expand_to_per_TR_frames()
        print(f'  {len(frame_stim_set)} frames to extract')

        print(f'  extracting per-frame features (vision tower)...')
        features, frame_ids = self._extract_per_TR_features(
            candidate, frame_stim_set)
        print(f'  features: {features.shape}')

        print(f'  aligning to assembly...')
        X = self._align_features_to_assembly(
            features, frame_ids, frame_stim_set)

        # Build run_idx_per_obs by enumerating unique (stimulus_id, run)
        # in the order they appear on the presentation axis.
        a_stim = list(self.assembly['stimulus_id'].values)
        a_run = list(self.assembly['run'].values)
        run_keys = []
        run_idx_per_obs = np.empty(len(a_stim), dtype=np.int64)
        seen: Dict[Tuple[str, str], int] = {}
        for i, (s, r) in enumerate(zip(a_stim, a_run)):
            key = (str(s), str(r))
            if key not in seen:
                seen[key] = len(seen)
                run_keys.append(key)
            run_idx_per_obs[i] = seen[key]
        n_runs = len(run_keys)
        print(f'  {n_runs} unique (stim, run) blocks')

        print(f'  applying stimulus_window={self._stimulus_window} + '
              f'hrf_delay={self._hrf_delay}...')
        X_stacked = self._apply_stimulus_window_and_hrf(
            X, run_idx_per_obs)

        # Drop the first/last excluded samples per run
        keep = np.ones(len(X_stacked), dtype=bool)
        for ri in range(n_runs):
            run_mask = run_idx_per_obs == ri
            run_idxs = np.where(run_mask)[0]
            if len(run_idxs) == 0:
                continue
            for ki in range(self._excluded_samples_start):
                if ki < len(run_idxs):
                    keep[run_idxs[ki]] = False
            for ki in range(self._excluded_samples_end):
                if ki < len(run_idxs):
                    keep[run_idxs[-1 - ki]] = False
        X_stacked = X_stacked[keep]
        Y = self.assembly.values[keep]
        run_idx_kept = run_idx_per_obs[keep]
        print(f'  after exclusions: X={X_stacked.shape} Y={Y.shape}')

        # Ridge with 5-fold CV over runs (not over individual TRs —
        # avoid temporal leakage). Each run goes entirely to either
        # train or test in any given fold.
        print(f'  ridge fit (5-fold over runs, per-parcel Pearson)...')
        unique_runs = np.unique(run_idx_kept)
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        held_out_preds = np.full_like(Y, np.nan, dtype=np.float32)
        for fold_i, (tr_run_pos, te_run_pos) in enumerate(
                kf.split(unique_runs)):
            tr_runs = unique_runs[tr_run_pos]
            te_runs = unique_runs[te_run_pos]
            tr = np.isin(run_idx_kept, tr_runs)
            te = np.isin(run_idx_kept, te_runs)
            reg = Ridge(alpha=1.0).fit(X_stacked[tr], Y[tr])
            held_out_preds[te] = reg.predict(X_stacked[te]).astype(
                np.float32)
            print(f'    fold {fold_i+1}/5: train={tr.sum()} '
                  f'test={te.sum()}')

        # Per-parcel Pearson on held-out predictions
        valid = ~np.isnan(held_out_preds[:, 0])
        Yt = Y[valid]
        Yp = held_out_preds[valid]
        Yt_c = Yt - Yt.mean(axis=0, keepdims=True)
        Yp_c = Yp - Yp.mean(axis=0, keepdims=True)
        num = (Yt_c * Yp_c).sum(axis=0)
        den = np.sqrt((Yt_c ** 2).sum(axis=0)
                      * (Yp_c ** 2).sum(axis=0))
        with np.errstate(divide='ignore', invalid='ignore'):
            per_voxel_r = np.where(den > 0, num / den, np.nan)
        per_voxel_r_finite = per_voxel_r[~np.isnan(per_voxel_r)]
        median_r = float(np.median(per_voxel_r_finite))
        mean_r = float(np.mean(per_voxel_r_finite))

        score = Score(median_r / float(self.ceiling))
        score.attrs['raw'] = Score(median_r)
        score.attrs['mean_r'] = mean_r
        score.attrs['n_parcels_scored'] = int(len(per_voxel_r_finite))
        score.attrs['n_TRs'] = int(valid.sum())
        score.attrs['stimulus_window'] = self._stimulus_window
        score.attrs['hrf_delay'] = self._hrf_delay
        score.attrs['mode'] = self._mode
        score.attrs['pipeline'] = 'algonauts_video_only_frame_agg'
        return score


class Algonauts2025Friends(_Algonauts2025Base):
    """Train + score per-parcel encoding model on Friends S1-S6 +
    Movie10 (in-distribution). Cross-validated within each subject.

    Score is the median per-parcel Pearson r on held-out folds.
    """

    def __init__(self, subject: int, **kwargs):
        super().__init__(
            subject=subject, split='friends',
            identifier_suffix='', **kwargs)


class Algonauts2025FriendsS7(_Algonauts2025Base):
    """Held-out Friends Season 7 — Codabench leaderboard target.

    fMRI ground truth is withheld. __call__ returns predicted
    per-parcel time series in the format Codabench expects rather than
    a Pearson-r score.
    """

    def __init__(self, subject: int, **kwargs):
        super().__init__(
            subject=subject, split='friends_s7',
            identifier_suffix='-test', **kwargs)


class Algonauts2025OOD(_Algonauts2025Base):
    """Held-out OOD movies — Codabench OOD leaderboard target.

    Same shape as FriendsS7: returns predicted per-parcel time series
    rather than a Pearson-r score.
    """

    def __init__(self, subject: int, **kwargs):
        super().__init__(
            subject=subject, split='ood',
            identifier_suffix='-ood', **kwargs)
