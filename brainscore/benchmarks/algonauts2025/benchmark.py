"""Algonauts 2025 challenge benchmark classes.

Three subclasses share most logic:
- Algonauts2025Friends: train/test on Friends S1-S6 + Movie10 (in-dist).
  __call__ scores the candidate via per-TR frame extraction + ridge.
- Algonauts2025FriendsS7: held-out S7 — __call__ returns predicted
  per-parcel time series in the format Codabench expects.
- Algonauts2025OOD: held-out 2 h of OOD movies — same as S7 but
  different stim_set.

Default mode is video-only frame-aggregation: extract one frame per TR
midpoint, run the candidate's vision tower, stack stimulus_window TRs of
context per fMRI sample, HRF-shift, ridge-regress per subject, per-parcel
Pearson median. ``mode`` (see scoring.py) selects the combination; concat /
per_modality / banded need audio and/or language blocks from the
candidate's other towers, and raise a clear error until that multi-tower
extraction (EC2-verified) supplies them. The published fixed-encoder
(Wav2Vec2 + MiniLM) baseline reproduction lives in the reproduction script,
not in the maintained benchmark.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score

from .scoring import score_encoding_modes


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


def _ffmpeg_extract_one(args):
    """Worker for the frame-extraction Pool. Tuple-args because Pool.imap
    doesn't take starargs."""
    import subprocess
    video_path, out_dir, fps_offset, resize, n_TRs = args
    out_pat = f'{out_dir}/%04d.jpg'
    duration = TR_SEC * (n_TRs + 1)  # safety bound
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-ss', f'{TR_SEC * fps_offset:.4f}',
        '-i', video_path,
        '-t', f'{duration:.4f}',
        '-vf', f'fps=1/{TR_SEC},scale={resize}:{resize}',
        '-q:v', '3',
        out_pat,
    ]
    subprocess.run(cmd, check=False, capture_output=True)
    return video_path


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
        mode: str = 'video_only',
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

    def _expand_to_per_TR_frames(self, fps_offset: float = 0.5,
                                 resize: int = 224, n_workers: int = 8):
        """Build a frame-level StimulusSet — one row per (stim_id, TR).

        Uses ffmpeg via parallel subprocesses (~10x faster than cv2's
        Python loop). Each worker extracts all TR-midpoint frames from
        one video in a single ffmpeg call:
            ffmpeg -ss {offset} -i {video} \
                -vf "fps=1/{TR_SEC},scale={R}:{R}" -q:v 3 \
                {dir}/%04d.jpg
        Saves resized JPEGs (~10 KB each at 224×224) to
        {frames_dir}/{stim_id}/{t:04d}.jpg.
        """
        import multiprocessing as mp
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import (
            StimulusSet)

        frames_dir = self._frames_dir()
        frames_dir.mkdir(parents=True, exist_ok=True)
        stim_df = self.stimulus_set
        stim_to_n_TRs = self._stim_id_to_n_TRs()

        # Plan extraction jobs. Skip stims whose JPEGs already exist.
        rows = []
        jobs = []
        for _, srow in stim_df.iterrows():
            stim_id = srow['stimulus_id']
            video_path = Path(srow['video_path'])
            if not video_path.exists():
                continue
            n_TRs = stim_to_n_TRs.get(stim_id)
            if n_TRs is None:
                continue
            stim_frame_dir = frames_dir / stim_id
            stim_frame_dir.mkdir(parents=True, exist_ok=True)
            target_paths = [stim_frame_dir / f'{t+1:04d}.jpg'  # ffmpeg %04d starts at 1
                            for t in range(n_TRs)]
            for t, out in enumerate(target_paths):
                # PytorchWrapper indexes ``stimulus_set['stimulus_id']``
                # for path lookup, so each row's stimulus_id must be
                # globally unique. Use frame_id as the canonical ID;
                # keep the per-clip stim_id under ``clip_id`` for
                # downstream TR-alignment.
                frame_id = f'{stim_id}_t{t:04d}'
                rows.append({
                    'stimulus_id': frame_id,
                    'frame_id': frame_id,
                    'clip_id': stim_id,
                    't_within_run': t,
                    'image_file_name': str(out),
                })
            if not all(p.exists() for p in target_paths):
                jobs.append((str(video_path), str(stim_frame_dir),
                             fps_offset, resize, n_TRs))

        if jobs:
            print(f'  extracting frames for {len(jobs)} videos via '
                  f'ffmpeg (n_workers={n_workers})...')
            # 'fork' avoids the re-import that 'spawn' triggers (which
            # makes the worker re-execute the driver script's top-level
            # code). ffmpeg is just a subprocess so fork is safe here.
            ctx = mp.get_context('fork')
            with ctx.Pool(processes=n_workers) as pool:
                for done, _ in enumerate(
                        pool.imap_unordered(_ffmpeg_extract_one, jobs), 1):
                    if done % 25 == 0 or done == len(jobs):
                        print(f'    {done}/{len(jobs)} videos done',
                              flush=True)

        # ffmpeg sometimes outputs fewer frames than the assembly's
        # n_TRs expects (video shorter than the fMRI run by 1-2 TRs).
        # Fill any missing trailing frames by symlinking the last
        # existing one so downstream extraction sees a complete set.
        import shutil
        n_filled = 0
        for _, srow in stim_df.iterrows():
            stim_id = srow['stimulus_id']
            n_TRs = stim_to_n_TRs.get(stim_id)
            if n_TRs is None:
                continue
            stim_frame_dir = frames_dir / stim_id
            target_paths = [stim_frame_dir / f'{t+1:04d}.jpg'
                            for t in range(n_TRs)]
            existing = [p for p in target_paths if p.exists()]
            if not existing:
                continue
            last_existing = existing[-1]
            for p in target_paths:
                if not p.exists():
                    shutil.copy(last_existing, p)
                    n_filled += 1
        if n_filled > 0:
            print(f'  filled {n_filled} trailing frames by '
                  f'duplicating last-extracted')

        df = pd.DataFrame(rows)
        out_set = StimulusSet(df)
        out_set.identifier = (
            f'algonauts2025-{self._split}-sub{self._subject:02d}-frames')
        out_set.stimulus_paths = dict(
            zip(df['stimulus_id'], df['image_file_name']))
        return out_set

    def _stim_id_to_n_TRs(self) -> Dict[str, int]:
        """Count TRs per stimulus_id from the assembly's presentation
        coord. Both training and held-out stubs carry stimulus_id at
        the right resolution, so this works for all splits."""
        ids = list(self.assembly['stimulus_id'].values)
        from collections import Counter
        return dict(Counter(ids))

    # ── Feature extraction + alignment ────────────────────────────

    # Per-frame feature cap. CLIP's encoder.layers.10.layer_norm2 emits
    # (50_tokens × 768_hidden) = 38400 features per frame. With 162k TRs
    # × 5 stimulus_window stacking, the design matrix would be 125 GB.
    # TruncatedSVD compression to 1000 components keeps almost all
    # variance and shrinks the design matrix ~38× (matches what BLIP-2
    # / Lahner-multimodal do).
    FEATURE_DIM_CAP = 1000

    def _extract_per_TR_features(self, candidate, frame_stim_set,
                                 recording_target='IT'
                                 ) -> Tuple[np.ndarray, List[str]]:
        """Run candidate's vision tower on the frame stim_set.

        Compresses features via TruncatedSVD to FEATURE_DIM_CAP — the
        full token×hidden flatten from CLIP/BLIP-2/V-JEPA blows up the
        downstream design matrix beyond practical memory.

        Returns:
            features: (n_TRs, FEATURE_DIM_CAP) float32
            frame_ids: list of frame_id strings (one per row).
        """
        # recording_target selects the layer-mapping type: a single region name
        # (standard, e.g. 'IT'), the string 'all' (whole-brain over every mapped
        # layer), or a region backed by a CompositeSelector (units across layers).
        candidate.start_recording(recording_target, time_bins=[(0, int(TR_SEC * 1000))])
        assembly = candidate.process(frame_stim_set)
        if 'time_bin' in assembly.dims:
            assembly = assembly.mean(dim='time_bin')
        feats = assembly.values.astype(np.float32)
        if feats.shape[1] > self.FEATURE_DIM_CAP:
            from sklearn.decomposition import TruncatedSVD
            print(f'  SVD compress: {feats.shape[1]} → '
                  f'{self.FEATURE_DIM_CAP} features...')
            svd = TruncatedSVD(
                n_components=self.FEATURE_DIM_CAP, random_state=0)
            feats = svd.fit_transform(feats).astype(np.float32)
        # Source of truth for frame ordering is the INPUT frame_stim_set,
        # NOT the output assembly's coords. The brainscore_vision cache
        # round-trip can strip custom stim_set columns, but it preserves
        # row order 1:1 (paths feed in, activations come out in the same
        # order). The output assembly's row count must match the input
        # for this to hold.
        n_in = len(frame_stim_set)
        n_out = feats.shape[0]
        if n_in != n_out:
            raise ValueError(
                f"frame_stim_set has {n_in} rows but candidate.process "
                f"returned {n_out}; cannot trust input order.")
        frame_ids = list(frame_stim_set['stimulus_id'])
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
        # Build (clip_id, t) → frame_id mapping from the stim_set.
        # NOTE: frame_stim_set's 'stimulus_id' column is the unique
        # frame_id (PytorchWrapper requires this to be globally unique
        # for path lookup); the per-clip identifier lives in 'clip_id'.
        sf = frame_stim_set
        stim_t_to_frame_id = {
            (str(sf.iloc[i]['clip_id']),
             int(sf.iloc[i]['t_within_run'])):
            sf.iloc[i]['frame_id']
            for i in range(len(sf))
        }

        n_TRs = self.assembly.sizes['presentation']
        X = np.full((n_TRs, features.shape[1]), np.nan, dtype=np.float32)
        a_stim = list(self.assembly['stimulus_id'].values)
        a_t = list(self.assembly['t_within_run'].values)
        # Diagnostic: confirm dict + feat_idx coverage on the first TR
        first_key = (str(a_stim[0]), int(a_t[0]))
        first_fid = stim_t_to_frame_id.get(first_key)
        first_row = feat_idx.get(first_fid) if first_fid else None
        print(f'  diag: first TR key={first_key!r}, '
              f'fid={first_fid!r}, row={first_row!r}', flush=True)
        print(f'  diag: stim_t_to_frame_id size={len(stim_t_to_frame_id)}, '
              f'feat_idx size={len(feat_idx)}', flush=True)
        print(f'  diag: sample feat_idx keys: '
              f'{list(feat_idx.keys())[:3]}', flush=True)
        missing_fid = 0
        missing_row = 0
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
            print(f'  diag: missing breakdown — '
                  f'no frame_id in stim_t map: {missing_fid}, '
                  f'no row in feat_idx: {missing_row}', flush=True)
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
            raise ValueError(
                f"Held-out split {self._split!r} has no ground truth to score "
                f"against — it is a Codabench prediction target. Call "
                f"`generate_predictions(candidate, out_dir)` to produce the "
                f"per-parcel prediction .npy, then bundle with "
                f"`submit_codabench`. (Use Algonauts2025Friends for "
                f"in-distribution CV scoring.)"
            )
        return self._score_friends_train(candidate)

    def _score_friends_train(self, candidate) -> Score:
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

        # Honor self._mode. The candidate's vision tower supplies the
        # 'video' block; audio / language blocks (needed by concat,
        # per_modality, banded) come from the candidate's other towers —
        # supplied by multi-tower extraction (EC2-verified). Modes that
        # need an absent modality raise a clear error in the scorer.
        per_modality_stacked = {'video': X_stacked}
        print(f'  scoring mode={self._mode!r} over bands '
              f'{list(per_modality_stacked)} (run-held-out 5-fold)...')
        per_voxel_r, info = score_encoding_modes(
            per_modality_stacked, Y, run_idx_kept, self._mode,
            banded_alpha_grid=self.BANDED_ALPHA_GRID)

        per_voxel_r_finite = per_voxel_r[~np.isnan(per_voxel_r)]
        median_r = float(np.median(per_voxel_r_finite))
        mean_r = float(np.mean(per_voxel_r_finite))

        score = Score(median_r / float(self.ceiling))
        score.attrs['raw'] = Score(median_r)
        score.attrs['ceiling'] = self.ceiling   # uniform score-attr contract
        score.attrs['mean_r'] = mean_r
        score.attrs['n_parcels_scored'] = int(len(per_voxel_r_finite))
        score.attrs['n_TRs'] = int(len(Y))
        score.attrs['stimulus_window'] = self._stimulus_window
        score.attrs['hrf_delay'] = self._hrf_delay
        score.attrs['mode'] = self._mode
        score.attrs['bands'] = info.get('bands')
        score.attrs['pipeline'] = f"algonauts_{self._mode}_frame_agg"
        return score

    # ── Held-out prediction (Codabench submission) ────────────────

    def _design_matrix(self, candidate, drop_excluded=True):
        """Build the (X_stacked, Y, keep, run_idx) design matrix for this
        split's stimuli. Mirrors the front half of ``_score_friends_train``
        (extract → align → run-block index → stimulus-window/HRF stack), kept
        separate so the validated training-score path is untouched.

        ``drop_excluded`` trims the per-run start/end samples (as in scoring);
        held-out prediction sets it False so every TR gets a prediction. ``Y``
        is the recorded BOLD (training) or all-NaN (held-out stub).
        """
        frame_stim_set = self._expand_to_per_TR_frames()
        features, frame_ids = self._extract_per_TR_features(
            candidate, frame_stim_set)
        X = self._align_features_to_assembly(features, frame_ids, frame_stim_set)

        a_stim = list(self.assembly['stimulus_id'].values)
        # Held-out stubs (S7 / OOD) carry no 'run' coord — each clip is its own
        # block, so group on stimulus_id alone there.
        a_run = (list(self.assembly['run'].values)
                 if 'run' in self.assembly.coords else a_stim)
        run_idx_per_obs = np.empty(len(a_stim), dtype=np.int64)
        seen: Dict[Tuple[str, str], int] = {}
        for i, (s, r) in enumerate(zip(a_stim, a_run)):
            key = (str(s), str(r))
            if key not in seen:
                seen[key] = len(seen)
            run_idx_per_obs[i] = seen[key]
        n_runs = len(seen)

        X_stacked = self._apply_stimulus_window_and_hrf(X, run_idx_per_obs)
        keep = np.ones(len(X_stacked), dtype=bool)
        if drop_excluded:
            for ri in range(n_runs):
                run_idxs = np.where(run_idx_per_obs == ri)[0]
                for ki in range(self._excluded_samples_start):
                    if ki < len(run_idxs):
                        keep[run_idxs[ki]] = False
                for ki in range(self._excluded_samples_end):
                    if ki < len(run_idxs):
                        keep[run_idxs[-1 - ki]] = False
        Y = self.assembly.values[keep]
        return X_stacked[keep], Y, keep, run_idx_per_obs[keep]

    def generate_predictions(self, candidate, out_dir, train_benchmark=None,
                             alpha=1.0):
        """Produce held-out per-parcel BOLD predictions for the Codabench target.

        Trains a ridge encoder on this subject's Friends-train (recorded) data
        and predicts every TR of this held-out split. Writes
        ``sub-<NN>_<split>.npy`` of shape ``(n_TRs, 1000)`` into ``out_dir``.
        Real run is on EC2 (needs the downloaded stimuli + assembly).
        """
        import os
        if self._split == 'friends':
            raise ValueError(
                "generate_predictions is for held-out splits; the friends "
                "training split is scored via __call__.")
        train = train_benchmark or Algonauts2025Friends(subject=self._subject)
        X_train, Y_train, _, _ = train._design_matrix(candidate, drop_excluded=True)
        X_pred, _, _, _ = self._design_matrix(candidate, drop_excluded=False)
        from brainscore.tools.banded_ridge import ridge_fit_predict
        preds = ridge_fit_predict(X_train, Y_train, X_pred, alpha=alpha)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'sub-{self._subject:02d}_{self._split}.npy')
        np.save(path, preds)
        return {'path': path, 'predictions': preds, 'subject': self._subject,
                'split': self._split, 'n_TRs': int(preds.shape[0])}


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
