"""Data plugin for locally prepared Algonauts2025 / CNeuroMod assets.

Two packaging scripts in ``data_packaging/`` build the prepared files from the
native competition files: ``prepare_assembly`` (the fMRI and the movie table)
and ``prepare_stimuli`` (one file per scan: frames, clips and sound). This
module loads them: the assemblies and the movie tables through the registries,
and the per-scan tables through ``per_scan_stimuli``.
"""

import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence

from brainscore import data_registry, stimulus_set_registry
from brainscore.data import local


PARTICIPANTS = (1, 2, 3, 5)
SPLITS = ('friends', 'friends_s7', 'ood')

# One scan (TR) of Courtois NeuroMod lasts 1.49 s.
TR_SEC = 1.49
# One clip holds the 1.49 s of one scan. 30 frames in 1.49 s is 3000/149 fps
# (about 20.13), so every clip has the same frame count and the same duration
# whatever the frame rate of the source movie.
CLIP_FRAMES = 30
CLIP_FPS = f'{CLIP_FRAMES * 100}/{round(TR_SEC * 100)}'
# One sound chunk holds the 1.49 s of one scan: 23840 samples, 16 kHz, mono.
SOUND_RATE_HZ = 16000
SOUND_SAMPLES = round(TR_SEC * SOUND_RATE_HZ)

# The prepared media, one file per scan: folder under the prepared root, file
# suffix, number of the file of the first scan, and the table column.
MEDIA = {
    'frames': ('frames', '.jpg', 1, 'image_file_name'),
    'clips': ('clips', '.mp4', 0, 'video_path'),
    'sound': ('sound', '.wav', 0, 'audio_path'),
}
_PREPARE_STIMULI = 'python -m brainscore.data.algonauts2025.data_packaging.prepare_stimuli'


def default_data_root() -> Path:
    """The folder of prepared files: the local asset ``algonauts2025-prepared``."""
    return local.ALGONAUTS_PREPARED.resolved()


def _split_identifier(split: str) -> str:
    return split.replace('_', '-')


def assembly_identifier(split: str, participant: int) -> str:
    return f'Algonauts2025-{_split_identifier(split)}-sub{participant:02d}'


def stimulus_set_identifier(split: str) -> str:
    return f'Algonauts2025-{_split_identifier(split)}'


def load_algonauts2025_assembly(
        split: str, participant: int, root: Optional[Path] = None):
    """Load one participant/split per-TR Schaefer parcel assembly."""
    root = Path(root).expanduser() if root is not None else default_data_root()
    path = root / f'algonauts2025_{split}_sub{participant:02d}.nc'
    if not path.exists():
        raise FileNotFoundError(
            f"Algonauts assembly missing at {path}. Run "
            "python -m brainscore.data.algonauts2025.data_packaging.prepare_assembly "
            "on EC2 first; see README.md."
        )

    import xarray as xr
    from brainscore_core.supported_data_standards.brainio.assemblies import (
        NeuronRecordingAssembly,
    )

    data = xr.open_dataarray(str(path))
    return NeuronRecordingAssembly(data)


def load_algonauts2025_stimulus_set(split: str, root: Optional[Path] = None):
    """Load one split's movie metadata and stimulus paths."""
    root = Path(root).expanduser() if root is not None else default_data_root()
    path = root / f'algonauts2025_stim_{split}.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"Algonauts stim_set missing at {path}. Run "
            "python -m brainscore.data.algonauts2025.data_packaging.prepare_assembly "
            "on EC2 first."
        )

    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import (
        StimulusSet,
    )

    df = pd.read_csv(path)
    out = StimulusSet(df)
    out.identifier = f'algonauts2025-{split}'
    out.stimulus_paths = dict(zip(df['stimulus_id'], df['video_path']))
    return out


def scans_per_stimulus(assembly) -> Dict[str, int]:
    """The number of scans of each ``stimulus_id`` in ``assembly``.

    Both the training assemblies and the held-out placeholders carry
    ``stimulus_id`` per scan, so this works for all splits.
    """
    return dict(Counter(list(assembly['stimulus_id'].values)))


def scan_files(stimulus_set, n_scans: Dict[str, int], root: Path, kind: str):
    """``(stimulus_id, movie file, folder, files)`` for each movie that has scans.

    File ``t`` of a movie is ``<root>/<folder of kind>/<stimulus_id>/<t + first>``.
    A movie whose source file is absent, or that has no scans in ``n_scans``,
    is left out.
    """
    folder_name, suffix, first, _ = MEDIA[kind]
    movies = []
    for _, row in stimulus_set.iterrows():
        stimulus_id = row['stimulus_id']
        video_path = Path(row['video_path'])
        if not video_path.exists():
            continue
        n = n_scans.get(stimulus_id)
        if n is None:
            continue
        folder = Path(root) / folder_name / stimulus_id
        movies.append((stimulus_id, str(video_path), folder,
                       [folder / f'{t + first:04d}{suffix}' for t in range(n)]))
    return movies


def per_scan_stimuli(stimulus_set, n_scans: Dict[str, int], media: Sequence[str],
                     split: str, participant: int, root: Path):
    """The prepared stimulus table: one row per (movie, scan).

    ``media`` names the prepared files of each row, a key of ``MEDIA`` each
    (``['frames']``, ``['clips']`` or ``['clips', 'sound']``). The columns are
    ``stimulus_id`` (``<movie>_t<scan>``, unique per row), ``frame_id`` (the
    same), ``clip_id`` (the movie), ``t_within_run`` and one path column per
    medium. Raises ``FileNotFoundError`` that names the packaging command when
    a prepared file is missing.
    """
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    rows, columns, missing, n_files = None, {}, [], 0
    for kind in media:
        movies = scan_files(stimulus_set, n_scans, root, kind)
        files = [path for *_, paths in movies for path in paths]
        missing += [path for path in files if not path.exists()]
        n_files += len(files)
        columns[MEDIA[kind][3]] = [str(path) for path in files]
        if rows is None:
            rows = [{'stimulus_id': f'{stimulus_id}_t{t:04d}', 'frame_id': f'{stimulus_id}_t{t:04d}',
                     'clip_id': stimulus_id, 't_within_run': t}
                    for stimulus_id, _, _, paths in movies for t in range(len(paths))]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {n_files} prepared stimulus files are missing; the first is "
            f"{missing[0]}. Prepare them with `{_PREPARE_STIMULI} --split {split} "
            f"--participant {participant} --media {' '.join(media)} --prepared-root {root}`; "
            "see brainscore/data/algonauts2025/README.md.")
    df = pd.DataFrame(rows)
    for column, paths in columns.items():
        df[column] = paths
    table = StimulusSet(df)
    # The activations cache keys on the stimuli identifier, not on the columns,
    # so the identifier carries a signature of the sorted column names: a table
    # with other columns is other stimuli to the cache.
    column_signature = hashlib.md5('|'.join(sorted(map(str, df.columns))).encode()).hexdigest()[:8]
    table.identifier = (f"algonauts2025-{split}-sub{participant:02d}-{'-'.join(media)}"
                        f'-{column_signature}')
    table.stimulus_paths = dict(zip(df['stimulus_id'], df[MEDIA[media[0]][3]]))
    return table


def probe_stimuli(kind: str, root: Path, identifier: str):
    """A one-row table of one blank scan (a 224 x 224 frame or clip) for the memory check.

    The width of a model's answer does not depend on the content, so a blank
    file of the real size, frame count and frame rate gives the real width.
    """
    import subprocess
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    folder_name, suffix, _, column = MEDIA[kind]
    folder = Path(root) / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    probe_path = folder / f'_memcheck_probe{suffix}'
    if not probe_path.exists():
        if kind == 'frames':
            from PIL import Image
            Image.new('RGB', (224, 224), color=(0, 0, 0)).save(probe_path)
        else:
            subprocess.run(
                ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i',
                 f'color=c=black:s=224x224:r={CLIP_FPS}', '-frames:v', str(CLIP_FRAMES),
                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(probe_path)],
                check=True, capture_output=True)
    pid = '_memcheck_probe'
    table = StimulusSet(pd.DataFrame([{'stimulus_id': pid, column: str(probe_path)}]))
    table.identifier = f'{identifier}-memcheck-probe'
    table.stimulus_paths = {pid: str(probe_path)}
    return table


def _assembly_loader(split: str, participant: int):
    return lambda root=None: load_algonauts2025_assembly(
        split, participant, root=root)


def _stimulus_set_loader(split: str):
    return lambda root=None: load_algonauts2025_stimulus_set(
        split, root=root)


for _split in SPLITS:
    stimulus_set_registry[stimulus_set_identifier(_split)] = (
        _stimulus_set_loader(_split))
    for _participant in PARTICIPANTS:
        data_registry[assembly_identifier(_split, _participant)] = (
            _assembly_loader(_split, _participant))
