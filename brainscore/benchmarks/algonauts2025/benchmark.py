"""Algonauts 2025 challenge benchmark classes.

- ``Algonauts2025Friends`` scores a model on Friends S1-S6 and Movie10 with
  5-fold cross-validation over stimuli.
- ``Algonauts2025FriendsS7`` and ``Algonauts2025OOD`` have no recorded fMRI.
  ``generate_predictions`` writes the predicted parcel time series for
  Codabench.

``form`` selects what the benchmark sends for each scan (TR, 1.49 s):

- ``'image'`` (default): the frame at the middle of the scan, on ``vision:image``.
- ``'video'``: the 1.49 s clip of the scan, on ``vision:video``.
- ``'video+audio'``: the clip on ``vision:video`` and the sound of the same
  1.49 s on ``audio``, in one table row.

The forms differ only in what the benchmark sends. Every form uses one method.
The benchmark predicts all 1000 parcels from the model's layers for the regions
the benchmark names: ``V1``, ``V4`` and ``IT`` for every form, and ``A1`` too for
the video+audio form (``FORM_REGIONS``). It requests exactly these
``neural:<region>`` channels, asks for each region in its own session, and
averages each answer over its time bins. Each region is one band.

The folds keep every run of one stimulus on one side, because Movie10 repeats
some clips in a second run. In each outer fold the benchmark prepares each band
on the training stimuli only: it centres the band, reduces it to at most
``LAYER_DIM_CAP`` components (a PCA) and divides the band by one scale. It then
stacks ``stimulus_window`` scans per sample and shifts them by ``hrf_delay``
scans, inside each run. A banded ridge with an intercept has one penalty per
band; the penalty tuple is the one that predicts the parcels best, on average,
on held-out stimuli inside the training stimuli. The score is the raw median per-parcel
Pearson r on the held-out stimuli. No ceiling is measured.

With ``streaming=True`` the benchmark presents the table one scan at a time:
the subject answers one scan before it gets the next. With ``streaming=False``
the subject gets the whole table at once. A subject without trial memory gives
the same features both ways.

``movies`` limits the benchmark to named movies (their ``stimulus_id`` in the
assembly). Only those movies are cut and scored. This is for short runs.
"""
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.execution_plan import ExecutionPlan
from brainscore_core.metrics import Score
from brainscore import load_dataset, load_stimulus_set
from brainscore.data.algonauts2025 import (
    assembly_identifier,
    default_data_root,
    stimulus_set_identifier,
)

from .scoring import PENALTY_PROTOCOL, PROJECTION_PROTOCOL, fit_predict_banded, score_banded


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


# One clip holds the 1.49 s of one scan. 30 frames in 1.49 s is 3000/149 fps
# (about 20.13), so every clip has the same frame count and the same duration
# whatever the frame rate of the source movie.
CLIP_FRAMES = 30
CLIP_FPS = f'{CLIP_FRAMES * 100}/{round(TR_SEC * 100)}'


def _clip_ffmpeg_command(video_path, out_pattern, resize, n_TRs=None):
    """The one ffmpeg call that cuts a movie into one clip per TR.

    Frames are resampled to ``CLIP_FPS`` and a key frame is forced every
    ``CLIP_FRAMES`` frames (``-g`` with scene-cut detection off), so the segment
    muxer cuts exactly at clip borders. ``-segment_time_delta`` keeps the cut
    from missing a key frame that lands a rounding error before ``t x TR_SEC``.
    Clip ``t`` is the movie time [t x TR_SEC, (t+1) x TR_SEC). No sound.
    """
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video_path)]
    if n_TRs is not None:
        cmd += ['-t', f'{TR_SEC * n_TRs:.4f}']
    return cmd + [
        '-an', '-vf', f'fps={CLIP_FPS},scale={resize}:{resize}',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
        '-g', str(CLIP_FRAMES), '-keyint_min', str(CLIP_FRAMES), '-sc_threshold', '0',
        '-f', 'segment', '-segment_time', f'{TR_SEC}', '-segment_time_delta', '0.01',
        '-segment_start_number', '0', '-reset_timestamps', '1', '-segment_format', 'mp4',
        str(out_pattern),
    ]


def _clip_frame_count(path) -> int:
    """Frames in a clip, from the mp4 header (0 when the file cannot be read)."""
    import subprocess
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', str(path)],
        check=False, capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def _ffmpeg_cut_clips_one(args):
    """Worker for the clip-cutting Pool: one ffmpeg call per movie.

    The call writes into a work folder. A clip moves to ``{out_dir}/{t:04d}.mp4``
    only when it has exactly ``CLIP_FRAMES`` frames and ``t < n_TRs``. A short
    clip at the end of the movie is dropped, so a clip file is always complete;
    the caller fills what is missing.
    """
    import shutil
    import subprocess
    video_path, out_dir, resize, n_TRs = args
    out_dir = Path(out_dir)
    work = out_dir / '_cut'
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run(_clip_ffmpeg_command(video_path, work / '%04d.mp4', resize, n_TRs),
                   check=False, capture_output=True)
    for cut in sorted(work.glob('*.mp4')):
        t = int(cut.stem)
        if t < n_TRs and _clip_frame_count(cut) == CLIP_FRAMES:
            shutil.move(str(cut), str(out_dir / f'{t:04d}.mp4'))
    shutil.rmtree(work, ignore_errors=True)
    return video_path


# One sound chunk holds the 1.49 s of one scan: 23840 samples, 16 kHz, mono.
_SOUND_RATE_HZ = 16000
_SOUND_SAMPLES = round(TR_SEC * _SOUND_RATE_HZ)


def _sound_ffmpeg_command(video_path, out_pattern, n_TRs=None):
    """The one ffmpeg call that cuts the sound of a movie into one WAV per TR.

    The filter resamples to 16 kHz mono, groups the samples into packets of
    exactly ``_SOUND_SAMPLES`` and numbers them from 0 (``asetpts``), so the
    segment muxer cuts at every packet. Chunk ``t`` is the sound of the movie
    time [t x TR_SEC, (t+1) x TR_SEC), counted from the start of the sound.
    No picture.
    """
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video_path)]
    if n_TRs is not None:
        cmd += ['-t', f'{TR_SEC * n_TRs:.4f}']
    return cmd + [
        '-vn', '-af',
        (f'aresample={_SOUND_RATE_HZ},aformat=sample_fmts=s16:channel_layouts=mono,'
         f'asetnsamples=n={_SOUND_SAMPLES}:p=0,asetpts=N/SR/TB'),
        '-c:a', 'pcm_s16le',
        '-f', 'segment', '-segment_time', f'{TR_SEC}', '-segment_time_delta', '0.01',
        '-segment_start_number', '0', '-reset_timestamps', '1', '-segment_format', 'wav',
        str(out_pattern),
    ]


def _wav_sample_count(path) -> int:
    """Samples in a 16 kHz mono WAV (0 when the file cannot be read or has another form)."""
    import wave
    try:
        with wave.open(str(path)) as wav:
            if wav.getframerate() != _SOUND_RATE_HZ or wav.getnchannels() != 1:
                return 0
            return wav.getnframes()
    except (OSError, EOFError, wave.Error):
        return 0


def _ffmpeg_cut_sound_one(args):
    """Worker for the sound-cutting Pool: one ffmpeg call per movie.

    Like ``_ffmpeg_cut_clips_one``: a chunk moves to ``{out_dir}/{t:04d}.wav``
    only when it has exactly ``_SOUND_SAMPLES`` samples and ``t < n_TRs``. A
    short chunk at the end of the movie is dropped; the caller fills what is
    missing.
    """
    import shutil
    import subprocess
    video_path, out_dir, n_TRs = args
    out_dir = Path(out_dir)
    work = out_dir / '_cut'
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run(_sound_ffmpeg_command(video_path, work / '%04d.wav', n_TRs),
                   check=False, capture_output=True)
    for cut in sorted(work.glob('*.wav')):
        t = int(cut.stem)
        if t < n_TRs and _wav_sample_count(cut) == _SOUND_SAMPLES:
            shutil.move(str(cut), str(out_dir / f'{t:04d}.wav'))
    shutil.rmtree(work, ignore_errors=True)
    return video_path


def _as_features(assembly):
    """``(values (n_presentations, n_units) float32, neuroid ids or None)``."""
    if 'time_bin' in assembly.dims:
        assembly = assembly.mean(dim='time_bin')
    feats = assembly.values.astype(np.float32)
    try:
        feature_ids = assembly['neuroid_id'].values
    except KeyError:
        feature_ids = None
    return feats, feature_ids


def _features_in_order(response, stimuli):
    """``_as_features`` of ``response`` with its rows in the order of ``stimuli``.

    The response must answer each row of ``stimuli`` exactly once; its rows
    are matched by ``stimulus_id``.
    """
    from brainscore.presentation import response_stimulus_ids
    wanted = [str(s) for s in stimuli['stimulus_id'].values]
    answered = [str(s) for s in response_stimulus_ids(response, wanted)]
    if answered != wanted:
        position = {sid: index for index, sid in enumerate(answered)}
        response = response.isel(presentation=[position[sid] for sid in wanted])
    return _as_features(response)


def _not_estimated() -> Score:
    """No ceiling is measured for this benchmark; the score is the raw correlation."""
    ceiling = Score(float('nan'))
    ceiling.attrs['status'] = 'not_estimated'
    return ceiling


class _Algonauts2025Base(BenchmarkBase):
    """Shared logic across the three splits."""

    # One penalty per decade from 1e-2 to 1e8; the bands are standardized.
    BANDED_ALPHA_GRID = tuple(float(10.0 ** e) for e in range(-2, 9))
    # Components kept for each band, and penalty tuples tried by the random search
    # when the full grid (5 ** number of bands) is larger.
    LAYER_DIM_CAP = 100
    BANDED_CANDIDATES = 60

    # Sends one frame per TR on `vision:image` (form='image'), one 1.49 s clip per TR
    # on `vision:video` (form='video'), or the clip and its sound in one row on
    # `vision:video` and `audio` (form='video+audio'). The benchmark stacks the TRs
    # itself, so the subject keeps no history across TRs.
    VALID_FORMS = ('image', 'video', 'video+audio')
    FORM_CHANNELS = {
        'image': frozenset({'vision:image'}),
        'video': frozenset({'vision:video'}),
        'video+audio': frozenset({'vision:video', 'audio'}),
    }
    # The regions the benchmark names for each form. Each is one band.
    FORM_REGIONS = {
        'image': ('IT', 'V1', 'V4'),
        'video': ('IT', 'V1', 'V4'),
        'video+audio': ('A1', 'IT', 'V1', 'V4'),
    }
    required_input_channels = frozenset({'vision:image'})
    requested_output_channels = frozenset(f'neural:{r}' for r in FORM_REGIONS['image'])
    trial_memory = 'none'
    _FORM_SUFFIX = {'image': '', 'video': '-video', 'video+audio': '-video-audio'}
    _N_FOLDS = 5   # 5-fold scoring over stimuli

    def __init__(
        self,
        subject: int,
        split: str,
        identifier_suffix: str,
        stimulus_window: int = 5,
        hrf_delay: int = 3,
        excluded_samples_start: Optional[int] = None,
        excluded_samples_end: int = 5,
        assembly_root: Optional[Path] = None,
        form: str = 'image',
        streaming: bool = False,
        movies: Optional[Sequence[str]] = None,
    ):
        if subject not in (1, 2, 3, 5):
            raise ValueError(
                f"Algonauts subjects are {{1, 2, 3, 5}}; got {subject}.")
        if form not in self.VALID_FORMS:
            raise ValueError(
                f"form must be one of {self.VALID_FORMS}; got {form!r}.")
        # A kept sample needs every lag of its window inside its run.
        first_full = hrf_delay + stimulus_window - 1
        if excluded_samples_start is None:
            excluded_samples_start = first_full
        if excluded_samples_start < first_full:
            raise ValueError(
                f"excluded_samples_start={excluded_samples_start} keeps samples whose "
                f"window starts before the run; use at least hrf_delay + stimulus_window "
                f"- 1 = {first_full}.")
        if movies is not None and (isinstance(movies, str) or not len(movies)):
            raise ValueError(
                "movies must be a non-empty list of stimulus ids, for example "
                "['friends_s01e01a', 'friends_s01e01b'].")

        self._form = form
        self.required_input_channels = self.FORM_CHANNELS[form]
        self.requested_output_channels = frozenset(
            f'neural:{region}' for region in self.FORM_REGIONS[form])
        self._streaming = bool(streaming)
        self._movies = None if movies is None else [str(movie) for movie in movies]
        self._subject = subject
        self._split = split
        self._stimulus_window = stimulus_window
        self._hrf_delay = hrf_delay
        self._excluded_samples_start = excluded_samples_start
        self._excluded_samples_end = excluded_samples_end
        self._assembly_root = (Path(assembly_root)
                               if assembly_root else default_data_root())
        self._assembly = None
        self._stimulus_set = None

        super().__init__(
            identifier=(f'Algonauts2025-{split}-sub{subject:02d}'
                        f'{identifier_suffix}{self._FORM_SUFFIX[form]}'),
            version=2,
            parent='naturalistic',
            ceiling=_not_estimated(),
            bibtex=BIBTEX,
        )

    @property
    def assembly(self):
        if self._assembly is None:
            self._assembly = load_dataset(
                assembly_identifier(self._split, self._subject),
                root=self._assembly_root,
            )
            if self._movies is not None:
                self._assembly = self._only_movies(self._assembly)
        return self._assembly

    def _only_movies(self, assembly):
        """The TRs of ``assembly`` that belong to ``self._movies``.

        Raises ``ValueError`` when a named movie is not in the assembly, or when
        the movies are too few for 5-fold scoring over stimuli with a 5-fold
        choice of penalties inside every training fold.
        """
        stim = np.asarray(assembly['stimulus_id'].values).astype(str)
        unknown = sorted(set(self._movies) - set(stim))
        if unknown:
            raise ValueError(
                f"movies {unknown} are not in the assembly. Use stimulus ids such as "
                f"{sorted(set(stim))[:4]}.")
        keep = np.isin(stim, self._movies)
        n_stimuli = len(set(stim[keep]))
        # A fold trains on n - ceil(n / 5) stimuli and splits them 5 ways again.
        needed = self._N_FOLDS
        while needed - math.ceil(needed / self._N_FOLDS) < self._N_FOLDS:
            needed += 1
        # Only the training split is cross-validated; a held-out split needs no split.
        if self._split == 'friends' and n_stimuli < needed:
            raise ValueError(
                f"movies give {n_stimuli} stimuli; the benchmark needs at least "
                f"{needed} for 5-fold scoring over stimuli. Name more movies.")
        return assembly.isel(presentation=np.flatnonzero(keep))

    @property
    def stimulus_set(self):
        if self._stimulus_set is None:
            self._stimulus_set = load_stimulus_set(
                stimulus_set_identifier(self._split),
                root=self._assembly_root,
            )
        return self._stimulus_set

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Declared memory-execution shape so the pre-flight is grounded in real
        cardinality, not guessed off ``len(stimulus_set)`` (which counts videos).

        - Extraction runs one table row per TR, so the held matrix has one row per
          TR (``len(assembly.stimulus_id)``). The raw width is model-dependent, so
          the pre-flight probes it: it records one requested region for one blank
          frame or clip of this form. The bands of other regions can be wider.
        - The metric fits over the TR rows MINUS the per-run excluded samples.
        - Each band is reduced to ``LAYER_DIM_CAP`` components and stacked over
          ``stimulus_window`` TRs, one band per region of the form, so the metric
          width is ``len(FORM_REGIONS[form]) * stimulus_window * LAYER_DIM_CAP``.
        - Scoring is banded ridge, and the ceiling is a precomputed constant
          (``Score(1.0)``), not a re-run metric, so ``runs_ceiling_metric=False``.
        """
        n_trs = len(self.assembly['stimulus_id'])
        try:
            from collections import Counter
            stim = self.assembly['stimulus_id'].values
            run = self.assembly['run'].values
            run_lengths = Counter((str(s), str(r)) for s, r in zip(stim, run))
            excluded = self._excluded_samples_start + self._excluded_samples_end
            # clamp per run: a run shorter than the excluded span retains 0, not a
            # negative count (subtracting excluded*n_runs globally would under-count).
            metric_obs = max(sum(max(L - excluded, 0)
                                 for L in run_lengths.values()), 1)
        except Exception:
            metric_obs = int(n_trs)  # no run metadata (e.g. a stub): safe over-count
        return ExecutionPlan(
            n_extraction_presentations=int(n_trs),
            metric_observations=metric_obs,
            metric_feature_width=(len(self.FORM_REGIONS[self._form])
                                  * self._stimulus_window * self.LAYER_DIM_CAP),
            metric_category='ridge',
            runs_ceiling_metric=False,
            probe_stimuli=(self._probe_frame_stimuli() if self._form == 'image'
                           else self._probe_clip_stimuli()),
        )

    def _probe_frame_stimuli(self):
        """A 1-row image StimulusSet (a blank 224x224 frame) for the memory
        pre-flight to probe the vision tower's raw recording width.

        The width is content-invariant, so a blank frame gives the same width as a
        real one — without running ffmpeg over a video. Mirrors the frame-level set
        built in ``_expand_to_per_TR_frames`` (stimulus_id + image_file_name +
        stimulus_paths) so the candidate processes it identically.
        """
        import pandas as pd
        from PIL import Image
        from brainscore_core.supported_data_standards.brainio.stimuli import (
            StimulusSet)
        frames_dir = self._frames_dir()
        frames_dir.mkdir(parents=True, exist_ok=True)
        probe_path = frames_dir / '_memcheck_probe.jpg'
        if not probe_path.exists():
            Image.new('RGB', (224, 224), color=(0, 0, 0)).save(probe_path)
        pid = '_memcheck_probe'
        ss = StimulusSet(pd.DataFrame(
            [{'stimulus_id': pid, 'image_file_name': str(probe_path)}]))
        ss.identifier = f'{self.identifier}-memcheck-probe'
        ss.stimulus_paths = {pid: str(probe_path)}
        return ss

    def _probe_clip_stimuli(self):
        """A 1-row video StimulusSet (a blank 224x224 clip of one TR) for the memory
        pre-flight, the clip counterpart of ``_probe_frame_stimuli``. The clip has the
        same size, frame count and frame rate as a real one."""
        import subprocess
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import (
            StimulusSet)
        clips_dir = self._clips_dir()
        clips_dir.mkdir(parents=True, exist_ok=True)
        probe_path = clips_dir / '_memcheck_probe.mp4'
        if not probe_path.exists():
            subprocess.run(
                ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i',
                 f'color=c=black:s=224x224:r={CLIP_FPS}', '-frames:v', str(CLIP_FRAMES),
                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(probe_path)],
                check=True, capture_output=True)
        pid = '_memcheck_probe'
        ss = StimulusSet(pd.DataFrame(
            [{'stimulus_id': pid, 'video_path': str(probe_path)}]))
        ss.identifier = f'{self.identifier}-memcheck-probe'
        ss.stimulus_paths = {pid: str(probe_path)}
        return ss

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
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import (
            StimulusSet)

        frames_dir = self._frames_dir()
        frames_dir.mkdir(parents=True, exist_ok=True)
        stim_df = self.stimulus_set
        stim_to_n_TRs = self._stim_id_to_n_TRs()

        # Plan extraction jobs. Skip stims whose JPEGs already exist.
        # ffmpeg's %04d starts at 1, so frame t is file t+1.
        rows, planned = self._plan_per_TR_files(
            frames_dir, '.jpg', 1, 'image_file_name', stim_to_n_TRs)
        jobs = [(video, str(out_dir), fps_offset, resize, n_TRs)
                for video, out_dir, n_TRs in planned]

        if jobs:
            print(f'  extracting frames for {len(jobs)} videos via '
                  f'ffmpeg (n_workers={n_workers})...')
            self._run_pool(_ffmpeg_extract_one, jobs, n_workers)

        # ffmpeg sometimes outputs fewer frames than the assembly's
        # n_TRs expects (video shorter than the fMRI run by 1-2 TRs).
        # Fill any missing trailing frames by symlinking the last
        # existing one so downstream extraction sees a complete set.
        n_filled = self._fill_trailing_files(
            frames_dir, '.jpg', 1, stim_to_n_TRs)
        if n_filled > 0:
            print(f'  filled {n_filled} trailing frames by '
                  f'duplicating last-extracted')

        df = pd.DataFrame(rows)
        out_set = StimulusSet(df)
        # The activations cache (@store_xarray) keys on (model, stimuli_identifier,
        # layers) -- NOT on the stimulus set's columns. A fixed identifier therefore
        # lets a cached assembly built from one column set be merged with a fresh
        # extraction built from another, which surfaces as an opaque
        # `AssertionError: Length of new_levels (N) must be <= self.nlevels (M)`
        # from deep inside pandas' MultiIndex recode. Folding the column signature
        # into the identifier makes the cache miss instead, which is the correct
        # behaviour: different columns are genuinely different stimuli metadata.
        column_signature = hashlib.md5(
            '|'.join(sorted(map(str, df.columns))).encode()).hexdigest()[:8]
        out_set.identifier = (
            f'algonauts2025-{self._split}-sub{self._subject:02d}-frames'
            f'-{column_signature}')
        out_set.stimulus_paths = dict(
            zip(df['stimulus_id'], df['image_file_name']))
        return out_set

    def _plan_per_TR_files(self, root: Path, ext: str, first_index: int,
                           path_column: str, stim_to_n_TRs: Dict[str, int]):
        """One table row per (movie, TR), and the movies that still need cutting.

        File ``t`` of movie ``stim_id`` is ``{root}/{stim_id}/{t + first_index:04d}{ext}``.
        Returns ``(rows, jobs)``; a job is ``(video_path, stim_dir, n_TRs)`` and is
        listed only when some file of the movie is missing.
        """
        rows, jobs = [], []
        for _, srow in self.stimulus_set.iterrows():
            stim_id = srow['stimulus_id']
            video_path = Path(srow['video_path'])
            if not video_path.exists():
                continue
            n_TRs = stim_to_n_TRs.get(stim_id)
            if n_TRs is None:
                continue
            stim_dir = root / stim_id
            stim_dir.mkdir(parents=True, exist_ok=True)
            target_paths = [stim_dir / f'{t + first_index:04d}{ext}'
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
                    path_column: str(out),
                })
            if not all(p.exists() for p in target_paths):
                jobs.append((str(video_path), str(stim_dir), n_TRs))
        return rows, jobs

    @staticmethod
    def _run_pool(worker, jobs, n_workers: int) -> None:
        import multiprocessing as mp
        # 'fork' avoids the re-import that 'spawn' triggers (which
        # makes the worker re-execute the driver script's top-level
        # code). ffmpeg is just a subprocess so fork is safe here.
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=n_workers) as pool:
            for done, _ in enumerate(pool.imap_unordered(worker, jobs), 1):
                if done % 25 == 0 or done == len(jobs):
                    print(f'    {done}/{len(jobs)} videos done', flush=True)

    def _fill_trailing_files(self, root: Path, ext: str, first_index: int,
                             stim_to_n_TRs: Dict[str, int]) -> int:
        """Copy the last existing file over every missing one; returns the count."""
        import shutil
        n_filled = 0
        for _, srow in self.stimulus_set.iterrows():
            stim_id = srow['stimulus_id']
            n_TRs = stim_to_n_TRs.get(stim_id)
            if n_TRs is None:
                continue
            target_paths = [root / stim_id / f'{t + first_index:04d}{ext}'
                            for t in range(n_TRs)]
            existing = [p for p in target_paths if p.exists()]
            if not existing:
                continue
            last_existing = existing[-1]
            for p in target_paths:
                if not p.exists():
                    shutil.copy(last_existing, p)
                    n_filled += 1
        return n_filled

    # ── Per-TR clip cutting (form='video') ────────────────────────

    def _clips_dir(self) -> Path:
        """Cache dir for cut clips. Idempotent across runs."""
        return self._assembly_root / 'clips'

    def _expand_to_per_TR_clips(self, resize: int = 224, n_workers: int = 8):
        """Build a clip-level StimulusSet: one row per (stim_id, TR).

        Clip ``t`` is the movie time [t x TR_SEC, (t+1) x TR_SEC), cut at
        ``resize`` x ``resize`` with no sound by one ffmpeg call per movie (see
        ``_clip_ffmpeg_command``) into ``{clips_dir}/{stim_id}/{t:04d}.mp4``.
        Every clip has ``CLIP_FRAMES`` frames. Clips the movie does not reach are
        copies of the last clip. The columns are those of the frame table, with
        ``video_path`` in place of ``image_file_name``.
        """
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import (
            StimulusSet)

        clips_dir = self._clips_dir()
        clips_dir.mkdir(parents=True, exist_ok=True)
        stim_to_n_TRs = self._stim_id_to_n_TRs()
        rows, planned = self._plan_per_TR_files(
            clips_dir, '.mp4', 0, 'video_path', stim_to_n_TRs)
        jobs = [(video, out_dir, resize, n_TRs) for video, out_dir, n_TRs in planned]
        if jobs:
            print(f'  cutting clips for {len(jobs)} videos via '
                  f'ffmpeg (n_workers={n_workers})...')
            self._run_pool(_ffmpeg_cut_clips_one, jobs, n_workers)
        n_filled = self._fill_trailing_files(clips_dir, '.mp4', 0, stim_to_n_TRs)
        if n_filled > 0:
            print(f'  filled {n_filled} trailing clips by '
                  f'duplicating last-cut')

        df = pd.DataFrame(rows)
        # A movie that ffmpeg could not cut leaves no clips. Stop here, before a model reads a missing file.
        missing = [path for path in df['video_path'] if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(df)} clips were not cut; the first is {missing[0]}. "
                "Check that ffmpeg can read the movie files.")
        out_set = StimulusSet(df)
        # Same reasoning as for the frame table: the column signature is part of
        # the identifier, and the video table has its own identifier.
        column_signature = hashlib.md5(
            '|'.join(sorted(map(str, df.columns))).encode()).hexdigest()[:8]
        out_set.identifier = (
            f'algonauts2025-{self._split}-sub{self._subject:02d}-clips'
            f'-{column_signature}')
        out_set.stimulus_paths = dict(zip(df['stimulus_id'], df['video_path']))
        return out_set

    def _sound_dir(self) -> Path:
        """Cache dir for cut sound chunks. Idempotent across runs."""
        return self._assembly_root / 'sound'

    def _expand_to_per_TR_clips_and_sound(self, resize: int = 224, n_workers: int = 8):
        """The clip table with the sound of each clip: one row per (stim_id, TR).

        Sound chunk ``t`` is the sound of the movie time [t x TR_SEC, (t+1) x TR_SEC),
        16 kHz mono, cut by one ffmpeg call per movie (see ``_sound_ffmpeg_command``)
        into ``{sound_dir}/{stim_id}/{t:04d}.wav``. Every chunk has the same number
        of samples. Chunks the movie does not reach are copies of the last chunk.
        The columns are those of the clip table plus ``audio_path``.
        """
        from brainscore_core.supported_data_standards.brainio.stimuli import (
            StimulusSet)

        clips = self._expand_to_per_TR_clips(resize=resize, n_workers=n_workers)
        sound_dir = self._sound_dir()
        sound_dir.mkdir(parents=True, exist_ok=True)
        stim_to_n_TRs = self._stim_id_to_n_TRs()
        rows, planned = self._plan_per_TR_files(
            sound_dir, '.wav', 0, 'audio_path', stim_to_n_TRs)
        if planned:
            print(f'  cutting sound for {len(planned)} videos via '
                  f'ffmpeg (n_workers={n_workers})...')
            self._run_pool(_ffmpeg_cut_sound_one, planned, n_workers)
        n_filled = self._fill_trailing_files(sound_dir, '.wav', 0, stim_to_n_TRs)
        if n_filled > 0:
            print(f'  filled {n_filled} trailing sound chunks by '
                  f'duplicating last-cut')

        sound = {row['stimulus_id']: row['audio_path'] for row in rows}
        missing = [path for path in sound.values() if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(sound)} sound chunks were not cut; the first is "
                f"{missing[0]}. Check that ffmpeg can read the sound of the movie files.")
        df = clips.copy()
        df['audio_path'] = [sound[sid] for sid in df['stimulus_id']]
        out_set = StimulusSet(df)
        column_signature = hashlib.md5(
            '|'.join(sorted(map(str, df.columns))).encode()).hexdigest()[:8]
        out_set.identifier = (
            f'algonauts2025-{self._split}-sub{self._subject:02d}-clips-sound'
            f'-{column_signature}')
        out_set.stimulus_paths = dict(zip(df['stimulus_id'], df['video_path']))
        return out_set

    def _expand_to_per_TR_stimuli(self):
        """The table of what one TR presents, in the form of this benchmark."""
        if self._form == 'video+audio':
            return self._expand_to_per_TR_clips_and_sound()
        if self._form == 'video':
            return self._expand_to_per_TR_clips()
        return self._expand_to_per_TR_frames()

    def _stim_id_to_n_TRs(self) -> Dict[str, int]:
        """Count TRs per stimulus_id from the assembly's presentation
        coord. Both training and held-out stubs carry stimulus_id at
        the right resolution, so this works for all splits."""
        ids = list(self.assembly['stimulus_id'].values)
        from collections import Counter
        return dict(Counter(ids))

    # ── Recording the bands ───────────────────────────────────────

    def _record_bands(self, candidate, table, regions) -> Dict[str, Tuple[np.ndarray, object]]:
        """``{region: (features, neuroid ids)}`` with one row per row of ``table``.

        Each region is asked for in its own session on its own channel,
        ``neural:<region>``, so the sessions together request exactly what
        pre-flight checked. One session per region is used because a legacy
        model answers one channel per session and two towers can answer on
        different time axes; the model runs once per region either way. Each
        answer is averaged over its time bins and put in the row order of
        ``table``. With ``streaming=True`` the subject gets one row at a time
        and answers it before the next; otherwise it gets the table at once.
        Nothing is reduced here: the reduction is fitted per fold in
        ``score_banded``.
        """
        from brainscore_core.streaming_helpers import stimulus_session
        bands = {}
        for region in regions:
            channel = f'neural:{region}'
            session = stimulus_session(table, [channel], None,
                                       time_bins=[(0, int(TR_SEC * 1000))],
                                       streaming=self._streaming)
            candidate.participate(session)
            features, feature_ids = _features_in_order(session.collect(channel), table)
            if features.shape[0] != len(table):
                raise ValueError(
                    f"the table has {len(table)} rows but {channel} answered "
                    f"{features.shape[0]}; cannot trust input order.")
            print(f'  band {region}: {features.shape}')
            bands[region] = (features, feature_ids)
        return bands

    def _assembly_bands(self, candidate, regions):
        """The bands of ``_record_bands``, aligned to the TR rows of the assembly."""
        table = self._expand_to_per_TR_stimuli()
        print(f"  recording {['neural:' + region for region in regions]} for {len(table)} rows "
              f"({'stream' if self._streaming else 'table'})...")
        frame_ids = list(table['stimulus_id'])
        return {region: (self._align_features_to_assembly(features, frame_ids, table), ids)
                for region, (features, ids) in self._record_bands(candidate, table, regions).items()}

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
            # video's actual duration). Fill each with the nearest earlier
            # valid TR of the same run (the first valid one at a run start),
            # never across a run boundary.
            print(f'  WARNING: {missing} of {n_TRs} TRs had no frame; '
                  f'filling from the same run.')
            run_idx, _ = self._runs_and_keep(drop_excluded=False)
            for run in np.unique(run_idx):
                rows = np.flatnonzero(run_idx == run)
                valid = rows[~np.isnan(X[rows, 0])]
                if not len(valid):
                    raise ValueError(f'Run {run} of the assembly has no TR with features.')
                last = X[valid[0]].copy()
                for i in rows:
                    if np.isnan(X[i, 0]):
                        X[i] = last
                    else:
                        last = X[i]
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
        # Output: (n_TRs, W * n_feat), the concat of W stacked feature vectors,
        # the leftmost W-1 TRs in the past. Sample t gets features at
        # t-D-(W-1), ..., t-D from its own run; a lag before the run start is zero.
        X_stacked = np.zeros((n_TRs, W * n_feat), dtype=X.dtype)
        rows = np.arange(n_TRs)
        for offset in range(W):
            src = rows - (D + (W - 1 - offset))
            inside = src >= 0
            inside[inside] = run_idx_per_obs[src[inside]] == run_idx_per_obs[rows[inside]]
            X_stacked[rows[inside], offset * n_feat:(offset + 1) * n_feat] = X[src[inside]]
        return X_stacked

    # ── Scoring ───────────────────────────────────────────────────

    def _runs_and_keep(self, drop_excluded: bool = True):
        """``(run index per TR, keep mask)`` over the assembly's TR rows.

        Raises when the rows of a run are not in ``t_within_run`` order.

        A run is one unique ``(stimulus_id, run)`` block, numbered in order of
        first appearance. Held-out stubs (S7, OOD) carry no ``run`` coordinate, so
        each clip is its own run there. ``keep`` is False for the first
        ``excluded_samples_start`` and the last ``excluded_samples_end`` TRs of
        each run when ``drop_excluded`` is True.
        """
        a_stim = list(self.assembly['stimulus_id'].values)
        a_run = (list(self.assembly['run'].values)
                 if 'run' in self.assembly.coords else a_stim)
        run_idx = np.empty(len(a_stim), dtype=np.int64)
        seen: Dict[Tuple[str, str], int] = {}
        for i, (s, r) in enumerate(zip(a_stim, a_run)):
            run_idx[i] = seen.setdefault((str(s), str(r)), len(seen))
        # The stack reads earlier rows as earlier scans: each run's rows must be
        # in t_within_run order.
        steps = np.asarray(self.assembly['t_within_run'].values)
        for ri in range(len(seen)):
            run_steps = steps[run_idx == ri]
            if np.any(np.diff(run_steps.astype(np.int64)) <= 0):
                raise ValueError(
                    f'The TRs of run {list(seen)[ri]} are not in t_within_run order.')
        keep = np.ones(len(run_idx), dtype=bool)
        if drop_excluded:
            for ri in range(len(seen)):
                run_idxs = np.where(run_idx == ri)[0]
                for ki in range(self._excluded_samples_start):
                    if ki < len(run_idxs):
                        keep[run_idxs[ki]] = False
                for ki in range(self._excluded_samples_end):
                    if ki < len(run_idxs):
                        keep[run_idxs[-1 - ki]] = False
        return run_idx, keep

    def _stacker(self, run_idx):
        """Stack the stimulus window and shift by the HRF delay within ``run_idx``'s runs."""
        return lambda X: self._apply_stimulus_window_and_hrf(X, run_idx)

    def __call__(self, candidate) -> Score:
        if self._split != 'friends':
            raise ValueError(
                f"Held-out split {self._split!r} has no ground truth to score "
                f"against; it is a Codabench prediction target. Call "
                f"`generate_predictions(candidate, out_dir)` to produce the "
                f"per-parcel prediction .npy, then bundle those .npy files into a "
                f"zip in the Codabench layout (one entry per subject/episode) and "
                f"upload it. (Use Algonauts2025Friends for in-distribution CV scoring.)"
            )
        return self._score_friends_train(candidate)

    def _stimulus_groups(self) -> np.ndarray:
        """The stimulus of each TR row: the unit that the folds keep on one side.

        Movie10 repeats some clips in a second run with the same ``stimulus_id``;
        both runs of a clip land in the same fold.
        """
        return np.asarray(self.assembly['stimulus_id'].values).astype(str)

    def _score_friends_train(self, candidate) -> Score:
        # The same pre-flight as score(): N/A before any cutting or recording.
        from brainscore_core.compatibility import check_compatibility
        check_compatibility(candidate, self)
        regions = list(self.FORM_REGIONS[self._form])
        bands = self._assembly_bands(candidate, regions)
        run_idx, keep = self._runs_and_keep()
        groups = self._stimulus_groups()
        Y = self.assembly.values
        print(f'  {len(np.unique(run_idx))} runs of {len(np.unique(groups))} stimuli; '
              f'{int(keep.sum())} of {len(keep)} TRs after exclusions')
        print(f'  banded ridge over bands {list(bands)} (5 folds over stimuli)...')
        per_parcel_r, info = score_banded(
            bands, Y, groups, keep, self._stacker(run_idx),
            dim_cap=self.LAYER_DIM_CAP, alpha_grid=self.BANDED_ALPHA_GRID,
            n_candidates=self.BANDED_CANDIDATES)

        finite = per_parcel_r[~np.isnan(per_parcel_r)]
        median_r = float(np.median(finite))
        # No ceiling is measured: the score is the raw median correlation.
        score = Score(median_r)
        score.attrs.update(
            raw=Score(median_r),
            normalized=False,
            ceiling_status='not_estimated',
            metric='median_per_parcel_pearson_r',
            mean_r=float(np.mean(finite)),
            n_parcels_scored=int(len(finite)),
            # The per-parcel scores in Schaefer order, for a cortical surface. A
            # parcel that could not be scored stays NaN, unpainted.
            per_parcel_r=per_parcel_r,
            n_TRs=int(keep.sum()),
            stimulus_window=self._stimulus_window,
            hrf_delay=self._hrf_delay,
            excluded_samples_start=self._excluded_samples_start,
            excluded_samples_end=self._excluded_samples_end,
            cv_groups='stimulus_id',
            form=self._form,
            presentation='stream' if self._streaming else 'table',
            bands=info['bands'],
            band_channels={band: f'neural:{band}' for band in info['bands']},
            components_per_band=info['components_per_band'],
            alpha_grid=info['alpha_grid'],
            alpha_candidates=info['alpha_candidates'],
            penalty_protocol=PENALTY_PROTOCOL,
            penalties_per_fold=info['penalties_per_fold'],
            feature_projection_protocol=PROJECTION_PROTOCOL,
        )
        if self._form != 'image':
            score.attrs['clip_seconds'] = TR_SEC
            score.attrs['frames_per_clip'] = CLIP_FRAMES
        if self._form == 'video+audio':
            score.attrs['sound_rate_hz'] = _SOUND_RATE_HZ
        if self._movies is not None:
            score.attrs['movies'] = list(self._movies)
        return score

    # ── Held-out prediction (Codabench submission) ────────────────

    def generate_predictions(self, candidate, out_dir, train_benchmark=None):
        """Produce held-out per-parcel BOLD predictions for the Codabench target.

        The method is the scoring method without the outer folds. The benchmark
        records the same bands on this subject's Friends training films and on
        this held-out split. It fits the preparation of each band on the
        training films, chooses the penalties on held-out training
        stimuli, fits one banded ridge on all training films and predicts every
        TR of this split. It writes ``sub-<NN>_<split>.npy`` of shape
        ``(n_TRs, 1000)`` into ``out_dir``.

        ``train_benchmark`` must be the Friends training split of the same
        subject, with the same form, stimulus window, HRF delay and streaming.
        Its ``movies`` limit the training films; this benchmark's ``movies``
        limit the predicted films.
        """
        import os
        if self._split == 'friends':
            raise ValueError(
                "generate_predictions is for held-out splits; the friends "
                "training split is scored via __call__.")
        train = train_benchmark or Algonauts2025Friends(
            subject=self._subject, form=self._form, streaming=self._streaming)
        if train._subject != self._subject or train._split != 'friends':
            raise ValueError('Prediction requires Friends training data for the same subject.')
        if train._form != self._form:
            raise ValueError('Training and prediction must present the same form.')
        if train._streaming != self._streaming:
            raise ValueError('Training and prediction must present the table the same way '
                             '(streaming).')
        if (train._stimulus_window, train._hrf_delay) != (
                self._stimulus_window, self._hrf_delay):
            raise ValueError('Training and prediction must use the same stimulus window and HRF delay.')
        from brainscore_core.compatibility import check_compatibility
        check_compatibility(candidate, self)
        regions = list(self.FORM_REGIONS[self._form])
        train_bands = train._assembly_bands(candidate, regions)
        train_runs, train_keep = train._runs_and_keep()
        predict_bands = self._assembly_bands(candidate, regions)
        predict_runs, _ = self._runs_and_keep(drop_excluded=False)
        preds, summary = fit_predict_banded(
            train_bands, train.assembly.values, train._stimulus_groups(), train_keep,
            train._stacker(train_runs), predict_bands, self._stacker(predict_runs),
            dim_cap=self.LAYER_DIM_CAP, alpha_grid=self.BANDED_ALPHA_GRID,
            n_candidates=self.BANDED_CANDIDATES)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'sub-{self._subject:02d}_{self._split}.npy')
        np.save(path, preds)
        return {'path': path, 'predictions': preds, 'subject': self._subject,
                'split': self._split, 'n_TRs': int(preds.shape[0]), 'bands': regions,
                'penalties': summary, 'penalty_protocol': PENALTY_PROTOCOL,
                'feature_projection_protocol': PROJECTION_PROTOCOL}


class Algonauts2025Friends(_Algonauts2025Base):
    """Train + score per-parcel encoding model on Friends S1-S6 +
    Movie10 (in-distribution). Cross-validated within each subject.

    Score is the raw median per-parcel Pearson r on held-out stimuli.
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
