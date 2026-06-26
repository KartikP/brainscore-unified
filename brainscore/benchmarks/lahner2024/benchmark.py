"""
Lahner 2024 BOLDMoments fMRI naturalistic benchmark (unified interface).

Subjects watched 1026 short (3s) video clips while being scanned with fMRI;
responses were averaged per TR and fit with GLM to give one BOLD estimate
per (stimulus, repetition, voxel). The assembly has shape:

    (time_bin=1, neuroid=20484, presentation=10260)

where ``time_bin`` spans the full 3-second clip (single pre-computed
estimate; no time-series regression required), ``neuroid`` is cortical
vertices across L/R hemispheres, and ``presentation`` is 1026 videos × 10
repetitions. We average repetitions to get one clean response per video.

Taken from brain-score/vision PR #1249 (YingtianDt):
    - S3 bucket, version_ids, SHA1s (verbatim)
    - BOLDMoments as the stimulus-set identifier
    - Citation

Reframed for the unified interface:
    - No dependency on the vision-specific Video/Stimulus class hierarchy
    - No dependency on TemporalInferencer (works for frame-based models)
    - Video → frame extraction via cv2 (``cv2.VideoCapture``), wrapped in
      ``brainscore_core.temporal.expand_clip_to_frames``
    - Post-extraction aggregation via ``brainscore_core.temporal.temporal_bin``

Scoring pipeline:
    1. For each of 1026 videos, extract N frames at fixed timestamps
    2. Run model's ``process()`` on the expanded StimulusSet
       (1026 × N rows) → per-frame activations
    3. ``temporal_bin`` with one bin spanning (0, 3000 ms) → one vector
       per video (mean of frame features)
    4. Average neural assembly across 10 repetitions → one BOLD vector
       per video per voxel
    5. Cross-validated PLS regression → Pearson per voxel → median

## IMPORTANT: what this benchmark does NOT measure

This is a **frame-aggregated** naturalistic benchmark, not a video-native one.
For each 3-second video, the model sees N static frames (default 3: at
0.5s / 1.5s / 2.5s), processes each as an independent image, and the
N feature vectors are **mean-pooled** before regression. The model does
NOT see motion, temporal ordering, optical flow, or event dynamics —
it sees a bag of snapshots.

Consequences:
    - A score here measures **how well static spatial features of a video
      predict BOLD responses**, not how well the model understands video.
    - Image-only models (CLIP, ResNet, ViT) can produce meaningful scores
      despite having zero video understanding.
    - Two videos that differ only in temporal order (e.g., a clip played
      forward vs. reversed) produce IDENTICAL model features under this
      pipeline. If the BOLD responses differ, this benchmark cannot
      detect that difference.

To get a *true* video-native score, register a model with VideoWrapper
(see ``unified/brainscore/model_helpers/video_wrapper.py``) which feeds
the model a proper ``(T, C, H, W)`` video tensor. Video-native models
(V-JEPA, VideoMAE, TimeSformer) internally integrate over time and can
be meaningfully compared against this frame-aggregated baseline.

The score attrs include ``pipeline='frame_aggregation'`` and
``n_frames_per_video`` to make this sampling explicit on every result.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuronRecordingAssembly,
)
from brainscore_core.supported_data_standards.brainio.s3 import (
    load_assembly_from_s3, load_stimulus_set_from_s3,
)
from brainscore_core.temporal import expand_clip_to_frames, temporal_bin


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

# S3 versioning — same as brain-score/vision PR #1249
STIMULUS_ID = 'BOLDMoments'
STIMULUS_BUCKET = 'brainscore-storage/brainscore-vision/benchmarks/Lahner2024-fMRI'
STIMULUS_CSV_SHA1 = '0b27388f5898c908f58cd1f21f8f5cb3eda8536e'
STIMULUS_ZIP_SHA1 = 'dc9c3bf631632cd433d02f2f1847fd33c01ae0b3'
STIMULUS_CSV_VERSION_ID = 'WaGkWh59b1drhy1MmAVVSxh7_VT_eTay'
STIMULUS_ZIP_VERSION_ID = 'OxpOYy_3bveay9NFFFxNCVyghyAbqyIt'

ASSEMBLY_ID = 'Lahner2024-fMRI'
ASSEMBLY_VERSION_ID = 'zr_i3T9Saww44rPNJwLaxo0hgp8rYjPO'
ASSEMBLY_SHA1 = '2c7f1d2e5724b8cc3c5cf47986e956c4f13001e4'

# Videos are 3 seconds long
VIDEO_DURATION_MS = 3000
# Default: three frames per video (start/middle/end)
DEFAULT_SAMPLE_TIMES_MS = (500, 1500, 2500)


def load_stimulus_set():
    return load_stimulus_set_from_s3(
        identifier=STIMULUS_ID,
        bucket=STIMULUS_BUCKET,
        csv_sha1=STIMULUS_CSV_SHA1,
        zip_sha1=STIMULUS_ZIP_SHA1,
        csv_version_id=STIMULUS_CSV_VERSION_ID,
        zip_version_id=STIMULUS_ZIP_VERSION_ID,
    )


def load_assembly(merge_stimulus_set_meta: bool = True):
    return load_assembly_from_s3(
        identifier=ASSEMBLY_ID,
        version_id=ASSEMBLY_VERSION_ID,
        sha1=ASSEMBLY_SHA1,
        bucket=STIMULUS_BUCKET,
        cls=NeuronRecordingAssembly,
        stimulus_set_loader=load_stimulus_set,
        merge_stimulus_set_meta=merge_stimulus_set_meta,
    )


def _extract_frame_with_cv2(video_path, time_ms: float, frames_dir: Path) -> str:
    """Extract a single frame from a video at the given timestamp.

    Writes the frame as a PNG next to the video under ``frames_dir``.
    Returns the path to the extracted frame. Idempotent: if the frame
    already exists, returns its path without re-extracting.

    Kept as a module-level helper (not a class method) so it's easy to
    swap out for ffmpeg or a different backend.
    """
    import cv2  # imported here to avoid making it a brainscore_core dep

    video_path = Path(video_path)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_path = frames_dir / f'{video_path.stem}_t{int(time_ms)}.png'
    if out_path.exists():
        return str(out_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cv2 could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame_idx = int(round(time_ms / 1000.0 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(
            f"cv2 failed to read frame at {time_ms}ms "
            f"(target idx {target_frame_idx}, fps {fps}) from {video_path}")
    # cv2 uses BGR; convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Write with PIL so path ends in .png
    from PIL import Image
    Image.fromarray(frame).save(out_path)
    return str(out_path)


class Lahner2024BOLDMoments(BenchmarkBase):
    """Naturalistic fMRI benchmark: predict per-video BOLD responses.

    The fMRI data is GLM-beta-estimated per trial (one value per voxel
    per video-repetition), so there's no time-series regression needed
    at scoring time. We average 10 repetitions per video and correlate
    model features against the resulting (1026, 20484) neural matrix.

    The benchmark uses ``process()`` via the unified interface. Any
    model that declares a vision preprocessor will route through its
    activations_model after we expand videos into frame images.

    Regression: PLS with object-level cross-validation (the default used
    by MajajHong benchmarks). Ceiling: split-half across subjects
    (not computed at __init__ — defaults to 1.0 unless ``ceiling`` is
    passed in).
    """

    def __init__(
        self,
        sample_times_ms: List[float] = list(DEFAULT_SAMPLE_TIMES_MS),
        frames_dir: Optional[Path] = None,
        ceiling: Optional[float] = None,
        reliability_threshold: Optional[float] = None,
        identifier_suffix: str = '',
    ):
        self._sample_times_ms = list(sample_times_ms)
        self._frames_dir = Path(frames_dir) if frames_dir else Path(
            '/home/ubuntu/brain-score-unified/data/lahner2024/frames')
        self._assembly: Optional[NeuronRecordingAssembly] = None
        self._stimulus_set = None
        # If set, only voxels with split-half reliability ≥ threshold will
        # be included in the score. Used by the -visual-roi variant to
        # focus on stimulus-driven (typically visual) cortex.
        self._reliability_threshold = reliability_threshold
        self._voxel_mask: Optional[np.ndarray] = None  # cached after first compute

        super().__init__(
            identifier=f'Lahner2024-fMRI-naturalistic{identifier_suffix}',
            version=1,
            parent='naturalistic',
            ceiling=Score(1.0 if ceiling is None else float(ceiling)),
            bibtex=BIBTEX,
        )
        # Accepts native-video AND still-image models (the frame-aggregation
        # fallback in __call__) — any-of gate, see compatibility Check 1b.
        self.accepted_modalities = {'video', 'vision'}

    def _get_voxel_mask(self) -> Optional[np.ndarray]:
        """Return a boolean mask over voxels if reliability filtering is on.

        Computed once, cached on the instance.
        """
        if self._reliability_threshold is None:
            return None
        if self._voxel_mask is None:
            reliability = self._split_half_reliability()
            self._voxel_mask = reliability >= self._reliability_threshold
        return self._voxel_mask

    @property
    def assembly(self) -> NeuronRecordingAssembly:
        if self._assembly is None:
            self._assembly = load_assembly()
        return self._assembly

    @property
    def stimulus_set(self):
        if self._stimulus_set is None:
            self._stimulus_set = load_stimulus_set()
        return self._stimulus_set

    def _expand_videos(self):
        """Produce a StimulusSet with one row per (video, sample_time).

        Rows carry ``clip_id`` (=stimulus_id) and ``frame_time_ms``, which
        ``temporal_bin`` consumes later. The ``filename`` column in the
        BOLDMoments set gives the mp4 file; we locate it inside the
        unpacked stimuli directory.
        """
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

        stim = self.stimulus_set
        # Actual video paths live in stim.get_stimulus(...) or in stimulus_paths.
        # ``load_stimulus_set_from_s3`` returns a StimulusSet where
        # ``get_stimulus(stimulus_id)`` gives the local unpacked video path.

        def frame_extractor(video_path, t_ms):
            return _extract_frame_with_cv2(
                video_path, t_ms, frames_dir=self._frames_dir)

        # Copy the stim set with the actual local video paths in a column
        # our helper can read. We use 'video_path' as the column name.
        rows = []
        for _, row in stim.iterrows():
            video_path = stim.get_stimulus(row['stimulus_id'])
            rows.append({
                'stimulus_id': row['stimulus_id'],
                'video_path': str(video_path),
            })
        df_videos = pd.DataFrame(rows)
        videos_set = StimulusSet(df_videos)
        videos_set.identifier = stim.identifier
        videos_set.stimulus_paths = dict(zip(df_videos['stimulus_id'],
                                              df_videos['video_path']))

        return expand_clip_to_frames(
            videos_set,
            sample_times_ms=self._sample_times_ms,
            frame_extractor=frame_extractor,
            clip_id_col='stimulus_id',
            video_col='video_path',
        )

    def _videos_stimulus_set(self):
        """Produce a StimulusSet with one row per video and a ``video_path``
        column — the input shape that ``VideoWrapper`` expects.

        Unlike ``_expand_videos()``, this does NOT pre-extract frames;
        the VideoWrapper samples frames from the full video internally,
        preserving temporal order.
        """
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

        stim = self.stimulus_set
        rows = []
        for _, row in stim.iterrows():
            video_path = stim.get_stimulus(row['stimulus_id'])
            rows.append({
                'stimulus_id': row['stimulus_id'],
                'video_path': str(video_path),
            })
        df = pd.DataFrame(rows)
        videos_set = StimulusSet(df)
        videos_set.identifier = f'{stim.identifier}-videos'
        videos_set.stimulus_paths = dict(zip(df['stimulus_id'], df['video_path']))
        return videos_set

    def _average_repetitions(self) -> NeuronRecordingAssembly:
        """Average the 10 repetitions per video → one BOLD vector per video."""
        a = self.assembly.squeeze('time_bin', drop=True)  # drop the unit dim
        # Group by stimulus_id, mean over repetitions.
        # The assembly has dims (neuroid, presentation); we want
        # (neuroid, stimulus_id=1026). groupby-mean is the clean way.
        averaged = a.groupby('stimulus_id').mean('presentation')
        return averaged

    def _split_half_reliability(self, n_splits: int = 20, random_state: int = 0) -> np.ndarray:
        """Per-voxel split-half reliability across the 10 repetitions.

        For each random split of reps into two halves of 5, average within
        each half to get (n_videos, n_voxels), then Pearson-correlate the
        two halves across videos for each voxel. Average across ``n_splits``
        random splits and apply Spearman-Brown correction.

        Returns a (n_voxels,) array of reliability values. High-reliability
        voxels are stimulus-driven (signal); low-reliability voxels are
        noise-dominated.
        """
        import pandas as pd
        a = self.assembly.squeeze('time_bin', drop=True)
        # Build a per-video table: rows per (stimulus_id, repetition, neuroid)
        # then we can split reps into halves per video.
        stim_ids = np.asarray(a['stimulus_id'].values)
        reps = np.asarray(a['repetition'].values)
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
            # Build aligned (n_stimuli, n_voxels) for each half
            def _mean_of_cols(cols_per_stim):
                # Each element is array of column indices (5 per stim)
                out = np.zeros((len(unique_stims), n_voxels))
                for si, cols in enumerate(cols_per_stim):
                    out[si] = data[:, cols].mean(axis=1)
                return out
            A = _mean_of_cols(half_a_cols)
            B = _mean_of_cols(half_b_cols)
            # Vectorized per-voxel Pearson across stimuli axis
            Am = A - A.mean(axis=0, keepdims=True)
            Bm = B - B.mean(axis=0, keepdims=True)
            num = (Am * Bm).sum(axis=0)
            den = np.sqrt((Am ** 2).sum(axis=0) * (Bm ** 2).sum(axis=0))
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(den > 0, num / den, 0.0)
            reliability_sum += r
        r_half = reliability_sum / n_splits
        # Spearman-Brown: full-set reliability from half-set reliability
        r_full = 2 * r_half / (1 + r_half)
        return r_full

    def __call__(self, candidate) -> Score:
        from scipy.stats import pearsonr

        # Configure the model's recording once (both paths use it).
        candidate.start_recording('IT', time_bins=[(0, VIDEO_DURATION_MS)])

        # Dispatch on the model's declared modality support. Video-native
        # models (VideoMAE, V-JEPA) process a whole video at once and
        # preserve temporal structure; image models see static frames.
        if 'video' in getattr(candidate, 'supported_modalities', set()):
            pipeline_mode = 'video_native'
            video_stim = self._videos_stimulus_set()
            result = candidate.process(video_stim)
            # result dims expected: (presentation, time_bin, neuroid)
            # Collapse time via mean — Lahner2024 has one BOLD estimate
            # per video, so we aggregate temporal features to one vector.
            data = result.values
            if data.ndim == 3:
                model_mat = data.mean(axis=1)  # (n_videos, n_features)
            else:
                model_mat = data
            clip_ids = list(result.indexes['presentation'].get_level_values('stimulus_id'))
        else:
            pipeline_mode = 'frame_aggregation'
            frame_stim = self._expand_videos()
            per_frame = candidate.process(frame_stim)
            per_video = temporal_bin(
                per_frame,
                time_bins=[(0, VIDEO_DURATION_MS)],
            )
            model_mat = per_video.values[:, 0, :]  # (n_videos, n_features)
            clip_ids = list(
                per_video.indexes['presentation'].get_level_values('clip_id'))

        # 5. Average neural across repetitions; align to model order
        neural = self._average_repetitions()
        # neural dims: ('neuroid', 'stimulus_id') — transpose to (stim, neuroid)
        neural_aligned = neural.sel(
            stimulus_id=list(clip_ids)
        ).transpose('stimulus_id', 'neuroid')
        neural_mat = neural_aligned.values  # (n_videos, n_voxels)

        # Optional voxel-level mask (e.g., reliability-thresholded visual ROI)
        mask = self._get_voxel_mask()
        if mask is not None:
            neural_mat = neural_mat[:, mask]

        # 6. Cross-validated Ridge regression → per-voxel Pearson
        # (using sklearn directly keeps brainscore_core dependency-clean)
        # Ridge auto-centers via its intercept; explicit per-feature
        # StandardScaler over-rescales heterogeneous-variance features and
        # costs ~0.18 raw r on Lahner2024 (see 2026-04-24 replication note).
        # Pass raw features directly.
        from sklearn.model_selection import KFold
        from sklearn.linear_model import Ridge

        n = model_mat.shape[0]
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        fold_preds = np.zeros_like(neural_mat)
        for train_idx, test_idx in kf.split(np.arange(n)):
            X_train = model_mat[train_idx]
            X_test = model_mat[test_idx]
            y_train = neural_mat[train_idx]
            reg = Ridge(alpha=1.0).fit(X_train, y_train)
            fold_preds[test_idx] = reg.predict(X_test)

        # Per-voxel Pearson on the held-out predictions
        # Vectorized: correlate each column independently
        y_true = neural_mat
        y_pred = fold_preds
        n_voxels = y_true.shape[1]
        per_voxel_r = np.zeros(n_voxels)
        for j in range(n_voxels):
            yt = y_true[:, j]
            yp = y_pred[:, j]
            if yt.std() > 0 and yp.std() > 0:
                per_voxel_r[j] = np.corrcoef(yt, yp)[0, 1]
            else:
                per_voxel_r[j] = np.nan

        # Summary: median of voxels with finite scores
        per_voxel_r = per_voxel_r[~np.isnan(per_voxel_r)]
        median_r = float(np.median(per_voxel_r))
        mean_r = float(np.mean(per_voxel_r))

        score = Score(median_r / float(self.ceiling))
        score.attrs['raw'] = Score(median_r)
        score.attrs['ceiling'] = self.ceiling   # uniform score-attr contract
        score.attrs['mean_r'] = mean_r
        score.attrs['n_voxels_scored'] = int(len(per_voxel_r))
        score.attrs['per_voxel_r'] = per_voxel_r   # exposed for bootstrap CIs
        score.attrs['n_videos'] = int(n)
        # Make the pipeline explicit so downstream consumers know what
        # assumption the score was computed under.
        score.attrs['pipeline'] = pipeline_mode
        if pipeline_mode == 'frame_aggregation':
            score.attrs['sample_times_ms'] = self._sample_times_ms
            score.attrs['n_frames_per_video'] = len(self._sample_times_ms)
            score.attrs['note'] = (
                f'{len(self._sample_times_ms)} static frames per video, '
                f'mean-pooled before regression. Temporal dynamics discarded. '
                f'See benchmark docstring.'
            )
        else:  # video_native
            score.attrs['note'] = (
                'Video-native model: full temporal dynamics preserved '
                'through the encoder; features mean-pooled across output '
                'time steps for regression against the single Lahner2024 '
                'BOLD estimate per video.'
            )
        if mask is not None:
            score.attrs['voxel_mask_n_total'] = int(mask.size)
            score.attrs['voxel_mask_n_kept'] = int(mask.sum())
            score.attrs['reliability_threshold'] = float(self._reliability_threshold)
        return score


def Lahner2024BOLDMoments_visualROI():
    """ROI variant: only voxels with split-half reliability ≥ 0.3.

    The threshold 0.3 is moderate — in Lahner2024, it typically retains
    ~2000–5000 voxels concentrated in visual cortex (the only region
    where short-video stimuli reliably drive BOLD across repetitions).
    Produces a tighter, more interpretable score than whole-cortex
    median, which is dragged down by thousands of noise-dominated voxels.
    """
    return Lahner2024BOLDMoments(
        reliability_threshold=0.3,
        identifier_suffix='-visualROI',
    )
