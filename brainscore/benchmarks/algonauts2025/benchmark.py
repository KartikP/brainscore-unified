"""Algonauts 2025 challenge benchmark classes.

Three subclasses share most logic:
- Algonauts2025Friends: train/test on Friends S1-S6 + Movie10 (in-dist).
  __call__ scores the candidate via per-TR frame extraction + ridge.
- Algonauts2025FriendsS7: held-out S7 — __call__ returns predicted
  per-parcel time series in the format Codabench expects.
- Algonauts2025OOD: held-out 2 h of OOD movies — same as S7 but
  different stim_set.

``form`` selects what the benchmark sends for each scan (TR, 1.49 s):

- ``'image'`` (default): the frame at the middle of the scan, on ``vision:image``.
  The benchmark records ``neural:IT``.
- ``'video'``: the 1.49 s clip of the scan, on ``vision:video``. The benchmark
  records ``neural:IT``.
- ``'video+audio'``: the clip on ``vision:video`` and the sound of the same
  1.49 s on ``audio``, in one table row. The benchmark records ``neural:IT``
  for the video band and ``neural:A1`` for the audio band. A model that cannot
  give both outputs gets N/A at pre-flight. The default mode is ``'banded'``.

For every form the benchmark stacks ``stimulus_window`` scans of features per
fMRI sample, shifts them by ``hrf_delay`` scans, fits a ridge regression per
subject and reports the median per-parcel Pearson r. ``mode`` (see scoring.py)
selects how the bands combine. The image and video forms give one band
(``video``), so only ``video_only`` runs for them.

With ``streaming=True`` (video+audio form only) the benchmark presents the
table one scan at a time: the subject answers one scan before it gets the
next. With ``streaming=False`` the subject gets the whole table at once. A
subject without trial memory gives the same features both ways.

``movies`` limits the benchmark to named movies (their ``stimulus_id`` in the
assembly). Only those movies are cut and scored. This is for short runs.

The published fixed-encoder (Wav2Vec2 + MiniLM) baseline reproduction lives
in the reproduction script, not in the maintained benchmark.
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

from .scoring import score_encoding_modes
from .projection import FeatureProjection


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


def _record_features(subject, stimuli, region: str, *, time_bins):
    """Record ``neural:<region>`` through ``participate`` only, in the order of ``stimuli``.

    ``region`` is a region the subject records, or ``'all'``. ``neural:all``
    is the core's address for every mapped region, recorded in one extraction
    exactly as ``start_recording('all')`` did, so the feature order and count
    do not change. The response must answer each frame exactly once; its rows
    are put in the order of ``stimuli`` by ``stimulus_id``.
    """
    from brainscore_core.streaming_helpers import neural_response
    response = neural_response(subject, stimuli, record=region, time_bins=time_bins)
    return _features_in_order(response, stimuli)


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


class _Algonauts2025Base(BenchmarkBase):
    """Shared logic across the three splits."""

    VALID_MODES = ('concat', 'per_modality', 'banded',
                   'video_only', 'audio_only', 'language_only')
    BANDED_ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)

    # Sends one frame per TR on `vision:image` (form='image'), one 1.49 s clip per TR
    # on `vision:video` (form='video'), or the clip and its sound in one row on
    # `vision:video` and `audio` (form='video+audio'), and asks for the regions it
    # records. The benchmark stacks the TRs itself, so the subject keeps no history
    # across TRs.
    VALID_FORMS = ('image', 'video', 'video+audio')
    FORM_CHANNELS = {
        'image': frozenset({'vision:image'}),
        'video': frozenset({'vision:video'}),
        'video+audio': frozenset({'vision:video', 'audio'}),
    }
    # The band each recorded region gives (form='video+audio').
    _BAND_CHANNELS = {'video': 'neural:IT', 'audio': 'neural:A1'}
    required_input_channels = frozenset({'vision:image'})
    requested_output_channels = frozenset({'neural:IT'})
    trial_memory = 'none'
    _FORM_SUFFIX = {'image': '', 'video': '-video', 'video+audio': '-video-audio'}
    _MIN_RUNS = 5   # run-held-out 5-fold scoring

    def __init__(
        self,
        subject: int,
        split: str,
        identifier_suffix: str,
        stimulus_window: int = 5,
        hrf_delay: int = 3,
        excluded_samples_start: int = 5,
        excluded_samples_end: int = 5,
        mode: Optional[str] = None,
        assembly_root: Optional[Path] = None,
        form: str = 'image',
        streaming: bool = False,
        movies: Optional[Sequence[str]] = None,
        record: str = 'regions',
    ):
        if subject not in (1, 2, 3, 5):
            raise ValueError(
                f"Algonauts subjects are {{1, 2, 3, 5}}; got {subject}.")
        if form not in self.VALID_FORMS:
            raise ValueError(
                f"form must be one of {self.VALID_FORMS}; got {form!r}.")
        if record not in ('regions', 'all'):
            raise ValueError(f"record must be 'regions' or 'all'; got {record!r}.")
        if record == 'all' and form != 'video+audio':
            raise ValueError("record='all' is for form='video+audio'.")
        if mode is None:
            # With record='all' every mapped layer of the model is one band.
            mode = ('concat' if record == 'all'
                    else 'banded' if form == 'video+audio' else 'video_only')
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}; got {mode!r}.")
        if streaming and form != 'video+audio':
            raise ValueError(
                "streaming=True is for form='video+audio'; the image and video "
                "forms present the table in one piece.")
        if movies is not None and (isinstance(movies, str) or not len(movies)):
            raise ValueError(
                "movies must be a non-empty list of stimulus ids, for example "
                "['friends_s01e01a', 'friends_s01e01b'].")

        self._form = form
        self._record = record
        self.required_input_channels = self.FORM_CHANNELS[form]
        if form == 'video+audio':
            self.requested_output_channels = frozenset(self._bands().values())
        self._streaming = bool(streaming)
        self._movies = None if movies is None else [str(movie) for movie in movies]
        self._subject = subject
        self._split = split
        self._stimulus_window = stimulus_window
        self._hrf_delay = hrf_delay
        self._excluded_samples_start = excluded_samples_start
        self._excluded_samples_end = excluded_samples_end
        self._mode = mode
        self._assembly_root = (Path(assembly_root)
                               if assembly_root else default_data_root())
        self._assembly = None
        self._stimulus_set = None

        super().__init__(
            identifier=(f'Algonauts2025-{split}-sub{subject:02d}'
                        f'{identifier_suffix}'
                        f"{'-wholebrain' if record == 'all' else self._FORM_SUFFIX[form]}"),
            version=1,
            parent='naturalistic',
            ceiling=Score(1.0),
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
        the movies give too few runs for run-held-out 5-fold scoring (banded
        mode also needs 5 runs inside every training fold).
        """
        stim = np.asarray(assembly['stimulus_id'].values).astype(str)
        unknown = sorted(set(self._movies) - set(stim))
        if unknown:
            raise ValueError(
                f"movies {unknown} are not in the assembly. Use stimulus ids such as "
                f"{sorted(set(stim))[:4]}.")
        keep = np.isin(stim, self._movies)
        runs = {(s, str(r)) for s, r in zip(stim[keep], np.asarray(
            assembly['run'].values)[keep])}
        # A banded fold trains on n - ceil(n / 5) runs and splits them 5 ways again.
        needed = self._MIN_RUNS
        if self._mode == 'banded':
            while needed - math.ceil(needed / self._MIN_RUNS) < self._MIN_RUNS:
                needed += 1
        if len(runs) < needed:
            raise ValueError(
                f"movies give {len(runs)} runs; mode={self._mode!r} needs at least "
                f"{needed} for run-held-out 5-fold scoring. Name more movies.")
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

        - Extraction runs one frame per TR, so the held matrix has one row per TR
          (``len(assembly.stimulus_id)``) — an order of magnitude more than videos.
          The raw per-frame width is model-dependent (materialised before the SVD
          cap), so it is PROBED — with the IT recording target and a synthetic 1-row
          image frame, so an image-only model isn't handed a raw video row.
        - The metric fits over the TR rows MINUS the per-run excluded samples.
        - Per-TR features are SVD-capped to ``FEATURE_DIM_CAP`` then stacked over
          ``stimulus_window`` TRs, so the metric width is at most
          ``stimulus_window * FEATURE_DIM_CAP`` (an upper bound for raw < cap).
        - Scoring is (banded) ridge, and the ceiling is a precomputed constant
          (``Score(1.0)``), not a re-run metric — so ``runs_ceiling_metric=False``.
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
            metric_feature_width=(self._stimulus_window * self.FEATURE_DIM_CAP
                                  * len(self._bands())),
            metric_category='ridge',
            runs_ceiling_metric=False,
            recording_target='IT',
            probe_stimuli=(self._probe_frame_stimuli() if self._form == 'image'
                           else self._probe_clip_stimuli()),
        )

    def _bands(self) -> Dict[str, str]:
        """``{band: recorded channel}`` for the form of this benchmark."""
        if self._record == 'all':
            # Every layer the model mapped, in one extraction, predicts the whole brain.
            return {'all': 'neural:all'}
        if self._form == 'video+audio':
            return dict(self._BAND_CHANNELS)
        return {'video': 'neural:IT'}

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

    # ── Feature extraction + alignment ────────────────────────────

    # Per-frame feature cap. CLIP's encoder.layers.10.layer_norm2 emits
    # (50_tokens × 768_hidden) = 38400 features per frame. With 162k TRs
    # × 5 stimulus_window stacking, the design matrix would be 125 GB.
    # TruncatedSVD compression to 1000 components keeps almost all
    # variance and shrinks the design matrix ~38× (matches what BLIP-2
    # / Lahner-multimodal do).
    FEATURE_DIM_CAP = 1000

    def _extract_per_TR_features(self, candidate, frame_stim_set,
                                 recording_target='IT', *, projection=None,
                                 fit_projection=False,
                                 ) -> Tuple[np.ndarray, List[str]]:
        """Run candidate's vision tower on the frame stim_set.

        Compresses features via TruncatedSVD to FEATURE_DIM_CAP — the
        full token×hidden flatten from CLIP/BLIP-2/V-JEPA blows up the
        downstream design matrix beyond practical memory. Held-out prediction
        passes a projection fitted on training frames. With no projection this
        fits on all supplied frames, the historical transductive CV protocol.

        Returns:
            features: (n_TRs, FEATURE_DIM_CAP) float32
            frame_ids: list of frame_id strings (one per row).
        """
        # recording_target is a region the subject declares as neural:<region>
        # (standard, e.g. 'IT', or a region backed by a CompositeSelector), or
        # the string 'all' (every mapped region in one extraction, as before).
        feats, feature_ids = _record_features(
            candidate, frame_stim_set, recording_target,
            time_bins=[(0, int(TR_SEC * 1000))])
        if projection is None:
            projection = FeatureProjection(self.FEATURE_DIM_CAP)
            fit_projection = True
        feats = (projection.fit_transform(feats, feature_ids) if fit_projection
                 else projection.transform(feats, feature_ids))
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
                f"frame_stim_set has {n_in} rows but the subject "
                f"returned {n_out}; cannot trust input order.")
        frame_ids = list(frame_stim_set['stimulus_id'])
        return feats, frame_ids

    def _record_bands(self, candidate, table) -> Dict[str, np.ndarray]:
        """``{band: X}`` for the video+audio form, each X aligned to the assembly.

        One session presents ``table`` (a clip and its sound per row) and asks
        for every band channel. With ``streaming=True`` the subject gets one row
        at a time and answers it before the next; otherwise it gets the table
        at once. Each band is projected to ``FEATURE_DIM_CAP`` like the video
        band of the other forms (fitted on all input rows).
        """
        from brainscore_core.streaming_helpers import stimulus_session
        bands = self._bands()
        if self._record == 'all':
            # neural:all is every region the subject declares. Each region is asked for
            # on its own channel, because two towers can have different time axes; the
            # benchmark takes the mean over time and joins the units into one band.
            bands = {'all': tuple(sorted(
                channel for channel in candidate.out_channels
                if channel.startswith('neural:') and channel != 'neural:all'))}
        else:
            bands = {band: (channel,) for band, channel in bands.items()}
        requested = sorted({channel for channels in bands.values() for channel in channels})
        session = stimulus_session(table, requested, None,
                                   time_bins=[(0, int(TR_SEC * 1000))],
                                   streaming=self._streaming)
        candidate.participate(session)
        frame_ids = list(table['stimulus_id'])
        X_bands = {}
        for band, channels in bands.items():
            channel = ' + '.join(channels)
            parts = [_features_in_order(session.collect(one), table) for one in channels]
            feats = np.concatenate([part for part, _ in parts], axis=1)
            feature_ids = (np.concatenate([ids for _, ids in parts])
                           if all(ids is not None for _, ids in parts) else None)
            if feats.shape[0] != len(table):
                raise ValueError(
                    f"the table has {len(table)} rows but {channel} answered "
                    f"{feats.shape[0]}; cannot trust input order.")
            feats = FeatureProjection(self.FEATURE_DIM_CAP).fit_transform(feats, feature_ids)
            print(f'  {band} band ({channel}): {feats.shape}')
            X_bands[band] = self._align_features_to_assembly(feats, frame_ids, table)
        return X_bands

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
                f"then bundle those .npy files into a zip in the Codabench layout "
                f"(one entry per subject/episode) and upload it. "
                f"(Use Algonauts2025Friends for "
                f"in-distribution CV scoring.)"
            )
        return self._score_friends_train(candidate)

    def _score_friends_train(self, candidate) -> Score:
        print(f'  expanding stim_set to per-TR frames...')
        frame_stim_set = self._expand_to_per_TR_stimuli()
        # Diagnostic escape hatch: cap the design so a memory investigation costs
        # seconds rather than a full extraction. Truncating the OUTER stimulus set
        # does not work -- the per-TR set is rebuilt here from the movie files --
        # so the cap has to be applied after expansion. The resulting score is
        # meaningless by construction and is only for profiling.
        cap = getattr(self, '_debug_max_trs', None)
        if cap is not None:
            frame_stim_set = frame_stim_set.iloc[:int(cap)]
            print(f'  DEBUG CAP: truncated to {len(frame_stim_set)} frames '
                  f'(score is invalid; profiling only)')
        print(f'  {len(frame_stim_set)} frames to extract')

        if self._form == 'video+audio':
            print(f'  recording {sorted(self._bands().values())} '
                  f"({'stream' if self._streaming else 'table'})...")
            X_bands = self._record_bands(candidate, frame_stim_set)
        else:
            print(f'  extracting per-frame features (vision tower)...')
            features, frame_ids = self._extract_per_TR_features(
                candidate, frame_stim_set)
            print(f'  features: {features.shape}')

            print(f'  aligning to assembly...')
            X_bands = {'video': self._align_features_to_assembly(
                features, frame_ids, frame_stim_set)}

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
        stacked_bands = {band: self._apply_stimulus_window_and_hrf(X, run_idx_per_obs)
                         for band, X in X_bands.items()}

        # Drop the first/last excluded samples per run
        keep = np.ones(len(run_idx_per_obs), dtype=bool)
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
        Y = self.assembly.values[keep]
        run_idx_kept = run_idx_per_obs[keep]

        # Honor self._mode. The image and video forms give the 'video' band
        # only; the video+audio form gives 'video' (neural:IT) and 'audio'
        # (neural:A1). Modes that need an absent band raise a clear error in
        # the scorer.
        per_modality_stacked = {band: X_stacked[keep]
                                for band, X_stacked in stacked_bands.items()}
        print(f'  after exclusions: '
              f'{ {band: X.shape for band, X in per_modality_stacked.items()} } Y={Y.shape}')
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
        # The per-parcel scores themselves, in Schaefer order, so they can be rendered
        # on a cortical surface. NaNs are kept in place: a parcel that could not be
        # scored must stay unpainted rather than collapse to a neighbour's value.
        score.attrs['per_parcel_r'] = per_voxel_r
        score.attrs['n_TRs'] = int(len(Y))
        score.attrs['stimulus_window'] = self._stimulus_window
        score.attrs['hrf_delay'] = self._hrf_delay
        score.attrs['mode'] = self._mode
        score.attrs['bands'] = info.get('bands')
        score.attrs['presentation'] = 'stream' if self._streaming else 'table'
        if self._movies is not None:
            score.attrs['movies'] = list(self._movies)
        if self._form == 'video+audio':
            # The clip and the sound of one TR are the stimulus, in one row.
            score.attrs['pipeline'] = f"algonauts_{self._mode}_clip_sound"
            score.attrs['form'] = 'video+audio'
            score.attrs['clip_seconds'] = TR_SEC
            score.attrs['frames_per_clip'] = CLIP_FRAMES
            score.attrs['sound_rate_hz'] = _SOUND_RATE_HZ
            score.attrs['band_channels'] = self._bands()
            if 'chosen_alphas_per_fold' in info:
                score.attrs['chosen_alphas_per_fold'] = info['chosen_alphas_per_fold']
        elif self._form == 'video':
            # The clip of one TR is the stimulus. Nothing is aggregated over frames
            # by the benchmark, and no sound or speech is sent.
            score.attrs['pipeline'] = f"algonauts_{self._mode}_clip"
            score.attrs['form'] = 'video'
            score.attrs['clip_seconds'] = TR_SEC
            score.attrs['frames_per_clip'] = CLIP_FRAMES
        else:
            score.attrs['pipeline'] = f"algonauts_{self._mode}_frame_agg"
        score.attrs['feature_projection_protocol'] = 'transductive_svd_all_input_frames'
        return score

    # ── Held-out prediction (Codabench submission) ────────────────

    def _design_matrix(self, candidate, drop_excluded=True, *, projection=None,
                       fit_projection=False):
        """Build the (X_stacked, Y, keep, run_idx) design matrix for this
        split's stimuli. Mirrors the front half of ``_score_friends_train``
        (extract → align → run-block index → stimulus-window/HRF stack), kept
        separate so the validated training-score path is untouched.

        ``drop_excluded`` trims the per-run start/end samples (as in scoring);
        held-out prediction sets it False so every TR gets a prediction. ``Y``
        is the recorded BOLD (training) or all-NaN (held-out stub).
        """
        if self._form == 'video+audio':
            raise NotImplementedError(
                "Held-out prediction takes one band; the video+audio form is "
                "scored on the Friends training split only.")
        frame_stim_set = self._expand_to_per_TR_stimuli()
        features, frame_ids = self._extract_per_TR_features(
            candidate, frame_stim_set, projection=projection,
            fit_projection=fit_projection)
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
        train = train_benchmark or Algonauts2025Friends(
            subject=self._subject, form=self._form)
        if train._subject != self._subject or train._split != 'friends':
            raise ValueError('Prediction requires Friends training data for the same subject.')
        if train._form != self._form:
            raise ValueError('Training and prediction must present the same form (image or video).')
        if (train._stimulus_window, train._hrf_delay) != (
                self._stimulus_window, self._hrf_delay):
            raise ValueError('Training and prediction must use the same stimulus window and HRF delay.')
        projection = FeatureProjection(train.FEATURE_DIM_CAP)
        X_train, Y_train, _, _ = train._design_matrix(
            candidate, drop_excluded=True, projection=projection, fit_projection=True)
        X_pred, _, _, _ = self._design_matrix(
            candidate, drop_excluded=False, projection=projection)
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
