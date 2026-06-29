"""Data plugin for locally prepared Algonauts2025 / CNeuroMod assets."""

import os
from pathlib import Path
from typing import Optional

from brainscore import data_registry, stimulus_set_registry


SUBJECTS = (1, 2, 3, 5)
SPLITS = ('friends', 'friends_s7', 'ood')


def default_data_root() -> Path:
    return Path(
        os.environ.get('ALGONAUTS_DATA_ROOT', '~/.brainio/algonauts2025')
    ).expanduser()


def _split_identifier(split: str) -> str:
    return split.replace('_', '-')


def assembly_identifier(split: str, subject: int) -> str:
    return f'Algonauts2025-{_split_identifier(split)}-sub{subject:02d}'


def stimulus_set_identifier(split: str) -> str:
    return f'Algonauts2025-{_split_identifier(split)}'


def load_algonauts2025_assembly(
        split: str, subject: int, root: Optional[Path] = None):
    """Load one subject/split per-TR Schaefer parcel assembly."""
    root = Path(root).expanduser() if root is not None else default_data_root()
    path = root / f'algonauts2025_{split}_sub{subject:02d}.nc'
    if not path.exists():
        raise FileNotFoundError(
            f"Algonauts assembly missing at {path}. Run "
            "python -m brainscore.data.algonauts2025.prepare_assembly "
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
            "python -m brainscore.data.algonauts2025.prepare_assembly "
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


def _assembly_loader(split: str, subject: int):
    return lambda root=None: load_algonauts2025_assembly(
        split, subject, root=root)


def _stimulus_set_loader(split: str):
    return lambda root=None: load_algonauts2025_stimulus_set(
        split, root=root)


for _split in SPLITS:
    stimulus_set_registry[stimulus_set_identifier(_split)] = (
        _stimulus_set_loader(_split))
    for _subject in SUBJECTS:
        data_registry[assembly_identifier(_split, _subject)] = (
            _assembly_loader(_split, _subject))
