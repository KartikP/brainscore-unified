"""Load prepared Sun 2025 data without presenting reference answers to subjects."""
import hashlib
import json
import os
from pathlib import Path

from . import IDENTIFIER


def load_prepared(root=None, *, verify_clips=True):
    root = Path(root or os.environ.get('BRAINSCORE_SUN2025_ROOT', '~/.brainio/sun2025')).expanduser()
    if not (root / 'manifest.json').is_file():
        raise FileNotFoundError('Prepare Sun2025-second-order-motion with brainscore.data; '
                                'set BRAINSCORE_SUN2025_ROOT or pass root=...')
    raw = (root / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    if manifest.get('dataset') != IDENTIFIER or manifest.get('schema_version') != 1:
        raise ValueError('Unsupported Sun2025 data manifest')
    from .data_packaging.prepare import checksum
    for name, expected in manifest['files'].items():
        if not verify_clips and name.startswith('clips/'):
            continue
        if checksum(root / name) != expected:
            raise ValueError(f'Prepared checksum mismatch: {name}')
    manifest['build_id'] = hashlib.sha256(raw).hexdigest()[:16]
    return root, manifest


def load_stimulus_set(root=None):
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    root, manifest = load_prepared(root)
    stimuli = StimulusSet(json.loads((root / 'stimuli.json').read_text()))
    if len(stimuli) != manifest['trials'] or stimuli.stimulus_id.duplicated().any():
        raise ValueError('Stimulus count or identity mismatch')
    stimuli['video_path'] = [str(root / value) for value in stimuli.video_path]
    stimuli.identifier = f'{IDENTIFIER}-clips-v1-{manifest["build_id"]}'
    stimuli.stimulus_paths = dict(zip(stimuli.stimulus_id, stimuli.video_path))
    return stimuli


def load_dataset(root=None):
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    root, manifest = load_prepared(root, verify_clips=False)
    rows = pd.DataFrame(json.loads((root / 'human-responses.json').read_text()))
    ids = sorted(rows.base_id.unique())
    ordered = rows.set_index(['base_id', 'participant']).loc[[(sid, p) for sid in ids for p in (1, 2, 3)]]
    values = ordered[['x', 'y']].to_numpy().reshape(len(ids), 3, 2)
    meta = rows.drop_duplicates('base_id').set_index('base_id').loc[ids]
    result = BehavioralAssembly(values, dims=['presentation', 'participant', 'axis'], coords={
        'stimulus_id': ('presentation', ids),
        'scene': ('presentation', meta.scene.to_numpy()),
        'modulation': ('presentation', meta.modulation.to_numpy()),
        'participant': [1, 2, 3], 'axis': ['x', 'y'],
        'archived_gt_x': ('presentation', meta.archived_gt_x.to_numpy()),
        'archived_gt_y': ('presentation', meta.archived_gt_y.to_numpy()),
    })
    result.attrs.update(identifier=f'{IDENTIFIER}-human-{manifest["build_id"]}',
                        provenance=manifest, measure='reported', reference='three human participants',
                        speed_units=manifest['speed_units'], coordinates=manifest['coordinates'])
    return result
