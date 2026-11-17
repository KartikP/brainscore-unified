"""Data plugin for Lahner2024 BOLDMoments assets.

The stimuli and the responses are hosted. Two local assets are prepared on this
machine from the hosted clips, by the scripts in ``data_packaging/``: the frames
a stills-only subject sees (``prepare_frames``) and the sound tracks
(``prepare_audio_tracks``).
"""

from pathlib import Path
from typing import Optional

from brainscore import data_builder_registry, data_registry, stimulus_set_registry
from brainscore.data import local
from brainscore.data.packaging import DataBuilder


STIMULUS_ID = 'BOLDMoments'
STIMULUS_BUCKET = 'brainscore-storage/brainscore-vision/benchmarks/Lahner2024-fMRI'
STIMULUS_CSV_SHA1 = '0b27388f5898c908f58cd1f21f8f5cb3eda8536e'
STIMULUS_ZIP_SHA1 = 'dc9c3bf631632cd433d02f2f1847fd33c01ae0b3'
STIMULUS_CSV_VERSION_ID = 'WaGkWh59b1drhy1MmAVVSxh7_VT_eTay'
STIMULUS_ZIP_VERSION_ID = 'OxpOYy_3bveay9NFFFxNCVyghyAbqyIt'

ASSEMBLY_ID = 'Lahner2024-fMRI'
ASSEMBLY_VERSION_ID = 'zr_i3T9Saww44rPNJwLaxo0hgp8rYjPO'
ASSEMBLY_SHA1 = '2c7f1d2e5724b8cc3c5cf47986e956c4f13001e4'

TIMERESOLVED_ASSEMBLY_ID = 'Lahner2024-fMRI-timeresolved'
TIMERESOLVED_ASSEMBLY_VERSION_ID: Optional[str] = '6VFZOroMxaNg1P_xSQAYwGq7.Fc63zz_'
TIMERESOLVED_ASSEMBLY_SHA1: Optional[str] = '4d06589dde6dddf273a489a64da1337940d5fafd'
TIMERESOLVED_EVENTS_ID = 'Lahner2024-fMRI-timeresolved-events'
TIMERESOLVED_EVENTS_VERSION_ID: Optional[str] = 'MYar7u8KE_D.i4MXZ83GurH1EDECZ.jo'
TIMERESOLVED_EVENTS_SHA1: Optional[str] = '783c5b33a75f121812a49b8b0a7639b9ed07c6a3'
TIMERESOLVED_MOTION_ID = 'Lahner2024-fMRI-timeresolved-motion'
TIMERESOLVED_MOTION_VERSION_ID: Optional[str] = 'wR1CwATaRjpEgvpR4ozr_UzkTS6yitjR'
TIMERESOLVED_MOTION_SHA1: Optional[str] = '827130c78f6b706b52f6d0d28962eb2301bee00b'

# The frames a stills-only subject sees from each 3-second clip: start, middle, end.
DEFAULT_SAMPLE_TIMES_MS = (500, 1500, 2500)

_FRAMES_COMMAND = 'python -m brainscore.data.lahner2024.data_packaging.prepare_frames'
_SOUND_COMMAND = 'python -m brainscore.data.lahner2024.data_packaging.prepare_audio_tracks'
_CLIPS_SOURCE = ('https://openneuro.org/datasets/ds005165 (BOLD Moments; the clips are the '
                 'hosted stimulus set BOLDMoments, which brainscore.load_stimulus_set fetches)')

# Both assets are needed only by the route a subject takes (frames for a subject
# that takes vision:image and not vision:video, sound for one that takes audio),
# so no benchmark identifier is listed: the benchmark checks them when it reads them.
LAHNER_FRAMES = local.register(local.LocalAsset(
    name='lahner2024-frames',
    env_var='BRAINSCORE_LAHNER_FRAMES_DIR',
    default_path='brainscore-data/lahner2024/frames',
    kind='stimuli',
    why_local='The frames are cut on this machine from the hosted clips; '
              'Brain-Score hosts the clips, not the frames.',
    source=_CLIPS_SOURCE,
    obtain='Nothing to download: the extractor reads the hosted clips.',
    prepare=_FRAMES_COMMAND,
))

LAHNER_SOUND = local.register(local.LocalAsset(
    name='lahner2024-sound',
    env_var='BRAINSCORE_LAHNER_SOUND_DIR',
    default_path='.brainio/lahner2024_audio_16k',
    kind='stimuli',
    why_local='The sound tracks (16 kHz mono WAV, one per clip) are demuxed on this '
              'machine from the hosted clips; Brain-Score does not host them.',
    source=_CLIPS_SOURCE,
    obtain='Nothing to download: the converter reads the hosted clips (it needs ffmpeg).',
    prepare=_SOUND_COMMAND,
))


_OPENNEURO = ('BOLD Moments (Lahner et al. 2024), OpenNeuro ds005165 v1.0.4, '
              'https://openneuro.org/datasets/ds005165; licence as OpenNeuro states it for ds005165 '
              '(not recorded in this plugin).')
_TIMESERIES_SOURCE = (f'{_OPENNEURO} The fMRIPrep derivatives `derivatives/versionB/fmriprep` and the '
                      'raw `events.tsv` files, read from s3://openneuro.org/ds005165 with no account.')

data_builder_registry['lahner2024'] = lambda: DataBuilder(
    source=f'{_OPENNEURO} The hosted files ({STIMULUS_ID} stimuli and {ASSEMBLY_ID} GLM responses) '
           'came from vision pull request #1249 (YingtianDt), whose packaging code is not in any '
           'repository read so far.',
    access='public, no account (the OpenNeuro data); the hosted files download with no account.',
    layout='not known: the GLM betas and clips of ds005165 that the hosted files were made from.',
    needs='not known.',
    writes=f'`stimulus_{STIMULUS_ID}.csv`, `stimulus_{STIMULUS_ID}.zip`, `assy_{ASSEMBLY_ID}.nc`. '
           'Use the hosted files: `brainscore.load_dataset` fetches them.',
    sha1={f'stimulus_{STIMULUS_ID}.csv': STIMULUS_CSV_SHA1, f'stimulus_{STIMULUS_ID}.zip': STIMULUS_ZIP_SHA1,
          f'assy_{ASSEMBLY_ID}.nc': ASSEMBLY_SHA1},
    cannot='No script builds the GLM responses and the BOLDMoments stimuli from OpenNeuro. A maintainer '
           'needs: the packaging code of vision pull request #1249, or a new script from the ds005165 '
           'GLM betas and clips; the rule that keeps 1,026 of the 1,102 videos (933 training-set, 93 '
           'test-set), which is not written anywhere found; then a cloud run that compares the rebuilt '
           'files with the registered sha1 or, for the netCDF file, with the hosted values.',
)

data_builder_registry['lahner2024-frames'] = lambda: DataBuilder(
    source=_CLIPS_SOURCE,
    access='public, no account: the hosted clips download with no account.',
    layout='none: the script reads the hosted clips through brainscore.load_stimulus_set.',
    needs='opencv-python and Pillow; the hosted clips (fetched once); minutes.',
    writes=f'one PNG per clip and time ({", ".join(map(str, DEFAULT_SAMPLE_TIMES_MS))} ms): '
           f'`<clip>_t<ms>.png`. Use them: set {LAHNER_FRAMES.env_var} to the output folder.',
    build='python -m brainscore.data.lahner2024.data_packaging.prepare_frames --frames-dir {output}',
    fetches=True,
)

data_builder_registry['lahner2024-sound'] = lambda: DataBuilder(
    source=_CLIPS_SOURCE,
    access='public, no account: the hosted clips download with no account.',
    layout='none: the script reads the hosted clips through brainscore.load_stimulus_set.',
    needs='ffmpeg and ffprobe on the PATH; the hosted clips (fetched once); minutes.',
    writes=f'one 16 kHz mono WAV per clip (`<clip>.wav`; a silent file for a clip with no sound) and '
           f'`manifest.json`. Use them: set {LAHNER_SOUND.env_var} to the output folder.',
    build='python -m brainscore.data.lahner2024.data_packaging.prepare_audio_tracks --audio-dir {output}',
    fetches=True,
)

data_builder_registry['lahner2024-timeresolved'] = lambda: DataBuilder(
    source=_TIMESERIES_SOURCE,
    access='public, no account.',
    layout='none: the script lists and downloads the files itself, into the source folder as its '
           'cache, and deletes each participant\'s files after use.',
    needs='the `aws` command, nibabel and nilearn; about 20 GB of disk at a time (about 200 GB pass '
          'through); hours.',
    writes=f'`assy_{TIMERESOLVED_ASSEMBLY_ID}.nc` and `{TIMERESOLVED_EVENTS_ID}.csv`. Use them: put each '
           'at `$BRAINIO_HOME/<sha1>/<file name>`.',
    build=f'python -m brainscore.data.lahner2024.data_packaging.prepare_timeresolved_assembly '
          f'--output-path {{output}}/assy_{TIMERESOLVED_ASSEMBLY_ID}.nc '
          f'--events-path {{output}}/{TIMERESOLVED_EVENTS_ID}.csv '
          '--data-cache {source} --checkpoint-path {source}/run_records.pkl',
    sha1={f'assy_{TIMERESOLVED_ASSEMBLY_ID}.nc': TIMERESOLVED_ASSEMBLY_SHA1,
          f'{TIMERESOLVED_EVENTS_ID}.csv': TIMERESOLVED_EVENTS_SHA1},
    fetches=True,
)

data_builder_registry['lahner2024-motion'] = lambda: DataBuilder(
    source=_TIMESERIES_SOURCE,
    access='public, no account.',
    layout='none: the script lists and downloads the fMRIPrep confounds files itself, into the source '
           'folder as its cache.',
    needs='boto3; a small download; minutes.',
    writes=f'`{TIMERESOLVED_MOTION_ID}.csv`. Use it: put it at `$BRAINIO_HOME/<sha1>/<file name>`.',
    build=f'python -m brainscore.data.lahner2024.data_packaging.prepare_motion_sidecar '
          f'--output-path {{output}}/{TIMERESOLVED_MOTION_ID}.csv --cache-dir {{source}}',
    sha1={f'{TIMERESOLVED_MOTION_ID}.csv': TIMERESOLVED_MOTION_SHA1},
    fetches=True,
)

data_builder_registry['destrieux-atlas'] = lambda: DataBuilder(
    source='the Destrieux 2009 surface atlas on fsaverage5, as nilearn publishes it '
           '(`nilearn.datasets.fetch_atlas_surf_destrieux`); its own terms.',
    access='public, no account.',
    layout='none.',
    needs='nilearn.',
    writes='Nilearn fetches the atlas on first use and the auditory-ROI benchmark reads it as published.',
)


def frame_file(frames_dir: Path, video_path, time_ms: float) -> Path:
    """The prepared frame of ``video_path`` at ``time_ms``: ``<stem>_t<ms>.png``."""
    return Path(frames_dir) / f'{Path(video_path).stem}_t{int(time_ms)}.png'


MOTION_COLUMNS = [
    'trans_x', 'trans_y', 'trans_z',
    'rot_x', 'rot_y', 'rot_z',
    'framewise_displacement',
    'csf', 'white_matter',
]


def _require_hosted(name: str, version_id: Optional[str], sha1: Optional[str],
                    prep_script: str) -> None:
    if version_id is None or sha1 is None:
        raise RuntimeError(
            f"{name} is not yet hosted on S3. Run {prep_script} on EC2 and "
            "update unified/brainscore/data/lahner2024/__init__.py with the "
            "resulting version and sha1."
        )


def load_lahner2024_stimulus_set():
    from brainscore_core.supported_data_standards.brainio.s3 import (
        load_stimulus_set_from_s3,
    )

    return load_stimulus_set_from_s3(
        identifier=STIMULUS_ID,
        bucket=STIMULUS_BUCKET,
        csv_sha1=STIMULUS_CSV_SHA1,
        zip_sha1=STIMULUS_ZIP_SHA1,
        csv_version_id=STIMULUS_CSV_VERSION_ID,
        zip_version_id=STIMULUS_ZIP_VERSION_ID,
    )


def load_lahner2024_assembly(merge_stimulus_set_meta: bool = True):
    from brainscore import load_stimulus_set
    from brainscore_core.supported_data_standards.brainio.assemblies import (
        NeuronRecordingAssembly,
    )
    from brainscore_core.supported_data_standards.brainio.s3 import (
        load_assembly_from_s3,
    )

    return load_assembly_from_s3(
        identifier=ASSEMBLY_ID,
        version_id=ASSEMBLY_VERSION_ID,
        sha1=ASSEMBLY_SHA1,
        bucket=STIMULUS_BUCKET,
        cls=NeuronRecordingAssembly,
        stimulus_set_loader=lambda: load_stimulus_set(STIMULUS_ID),
        merge_stimulus_set_meta=merge_stimulus_set_meta,
    )


def load_lahner2024_timeresolved_assembly(
        merge_stimulus_set_meta: bool = False):
    from brainscore import load_stimulus_set
    from brainscore_core.supported_data_standards.brainio.assemblies import (
        NeuronRecordingAssembly,
    )
    from brainscore_core.supported_data_standards.brainio.s3 import (
        load_assembly_from_s3,
    )

    _require_hosted(
        TIMERESOLVED_ASSEMBLY_ID,
        TIMERESOLVED_ASSEMBLY_VERSION_ID,
        TIMERESOLVED_ASSEMBLY_SHA1,
        "python -m brainscore.data.lahner2024.data_packaging.prepare_timeresolved_assembly",
    )
    return load_assembly_from_s3(
        identifier=TIMERESOLVED_ASSEMBLY_ID,
        version_id=TIMERESOLVED_ASSEMBLY_VERSION_ID,
        sha1=TIMERESOLVED_ASSEMBLY_SHA1,
        bucket=STIMULUS_BUCKET,
        cls=NeuronRecordingAssembly,
        stimulus_set_loader=lambda: load_stimulus_set(STIMULUS_ID),
        merge_stimulus_set_meta=merge_stimulus_set_meta,
    )


def _load_csv_sidecar(identifier: str, version_id: Optional[str],
                      sha1: Optional[str], prep_script: str):
    """A hosted CSV sidecar, fetched once, checked against its sha1 and cached."""
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.s3 import get_path

    _require_hosted(identifier, version_id, sha1, prep_script)
    bucket, folder = STIMULUS_BUCKET.split('/', 1)
    return pd.read_csv(get_path(identifier, 'csv', bucket, version_id, sha1,
                                filename_prefix='', folder_name=folder))


def load_lahner2024_timeresolved_events():
    return _load_csv_sidecar(
        TIMERESOLVED_EVENTS_ID,
        TIMERESOLVED_EVENTS_VERSION_ID,
        TIMERESOLVED_EVENTS_SHA1,
        "python -m brainscore.data.lahner2024.data_packaging.prepare_timeresolved_assembly",
    )


def load_lahner2024_timeresolved_motion():
    return _load_csv_sidecar(
        TIMERESOLVED_MOTION_ID,
        TIMERESOLVED_MOTION_VERSION_ID,
        TIMERESOLVED_MOTION_SHA1,
        "python -m brainscore.data.lahner2024.data_packaging.prepare_motion_sidecar",
    )


stimulus_set_registry[STIMULUS_ID] = load_lahner2024_stimulus_set
data_registry[ASSEMBLY_ID] = load_lahner2024_assembly
data_registry[TIMERESOLVED_ASSEMBLY_ID] = load_lahner2024_timeresolved_assembly
data_registry[TIMERESOLVED_EVENTS_ID] = load_lahner2024_timeresolved_events
data_registry[TIMERESOLVED_MOTION_ID] = load_lahner2024_timeresolved_motion
