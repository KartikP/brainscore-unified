"""Load a prepared Sun 2025 build; the stimulus table carries no reference answer."""
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def load_prepared(root: str | Path | None = None, *, verify_clips: bool = True) -> tuple[Path, dict]:
    """The build folder and its manifest, after the checksum of every listed file."""
    from brainscore.data import local
    from . import IDENTIFIER, PREPARED
    from .data_packaging.prepare import checksum
    root = Path(root).expanduser() if root else local.path(PREPARED.name)
    if not (root / 'manifest.json').is_file():
        raise local.LocalDataMissing(PREPARED.instructions())
    raw = (root / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    if manifest.get('dataset') != IDENTIFIER or manifest.get('schema_version') != 1:
        raise ValueError(f'{root / "manifest.json"} is not a Sun2025 build of schema 1; '
                         f'prepare it again: {PREPARED.prepare}')
    for name, expected in manifest['files'].items():
        if verify_clips or not name.startswith('clips/'):
            if checksum(root / name) != expected:
                raise ValueError(f'Checksum of {name} changed; prepare the build again: {PREPARED.prepare}')
    manifest['build_id'] = hashlib.sha256(raw).hexdigest()[:16]
    return root, manifest


def load_stimulus_set(root: str | Path | None = None) -> "StimulusSet":
    """One row per clip: stimulus_id, base_id, scene, modulation, flip, video_path."""
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    from . import IDENTIFIER
    root, manifest = load_prepared(root)
    stimuli = StimulusSet(json.loads((root / 'stimuli.json').read_text()))
    if len(stimuli) != manifest['trials'] or stimuli.stimulus_id.duplicated().any():
        raise ValueError('The stimulus count or identities do not match the manifest')
    stimuli['video_path'] = [str(root / value) for value in stimuli.video_path]
    stimuli.identifier = f'{IDENTIFIER}-clips-{manifest["build_id"]}'
    stimuli.stimulus_paths = dict(zip(stimuli.stimulus_id, stimuli.video_path))
    stimuli.attrs['manifest'] = manifest
    return stimuli


def load_dataset(root: str | Path | None = None) -> "BehavioralAssembly":
    """The three participants' mean x/y report (unflipped) for each scene and modulation."""
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    from . import IDENTIFIER
    root, manifest = load_prepared(root, verify_clips=False)
    rows = pd.DataFrame(json.loads((root / 'human-responses.json').read_text()))
    ids = sorted(rows.base_id.unique())
    ordered = rows.set_index(['base_id', 'participant']).loc[[(i, p) for i in ids for p in (1, 2, 3)]]
    meta = rows.drop_duplicates('base_id').set_index('base_id').loc[ids]
    reference = BehavioralAssembly(
        ordered[['x', 'y']].to_numpy().reshape(len(ids), 3, 2),
        dims=['presentation', 'participant', 'axis'],
        coords={'stimulus_id': ('presentation', ids),
                'scene': ('presentation', meta.scene.to_numpy()),
                'modulation': ('presentation', meta.modulation.to_numpy()),
                'participant': [1, 2, 3], 'axis': ['x', 'y']})
    reference.attrs.update(identifier=f'{IDENTIFIER}-human-{manifest["build_id"]}',
                           complete=bool(manifest['complete']),
                           speed_units=manifest['speed_units'], coordinates=manifest['coordinates'])
    return reference
