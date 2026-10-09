"""Load a validated local EWoK build through the ordinary data registries."""
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def load_prepared(root: str | Path | None = None) -> tuple[list[dict], dict]:
    root = Path(root or os.environ.get('BRAINSCORE_EWOK_ROOT', '~/.brainio/ewok-core-1.0')).expanduser()
    try:
        manifest = json.loads((root / 'manifest.json').read_text())
        raw = (root / 'items.json').read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(
            'EWoK is not prepared. Run python -m brainscore.data prepare '
            'EWoK-core-1.0 --source /path/to/native-data --output /path/to/build, '
            'then set BRAINSCORE_EWOK_ROOT or pass root=...'
        ) from None
    if manifest.get('schema_version') != 1 or manifest.get('dataset') != 'EWoK-core-1.0':
        raise ValueError('Unsupported EWoK build manifest')
    if hashlib.sha256(raw).hexdigest() != manifest['items_sha256']:
        raise ValueError('EWoK data checksum changed; rebuild from the native source')
    items = json.loads(raw)
    if len(items) != manifest['items'] or not items:
        raise ValueError('EWoK build item count is inconsistent')
    return items, manifest


def load_stimulus_set(root: str | Path | None = None) -> "StimulusSet":
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    rows, manifest = load_prepared(root)
    stimuli = StimulusSet(rows)
    stimuli['stimulus_id'] = [f"{manifest['items_sha256'][:16]}:{r['item_id']}" for r in rows]
    stimuli.identifier = f"EWoK-core-1.0-{manifest['items_sha256'][:16]}"
    stimuli.stimulus_paths = {}
    return stimuli


def load_dataset(root: str | Path | None = None) -> "BehavioralAssembly":
    """Package the correct context for each target alongside its stimulus metadata."""
    import numpy as np
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    stimuli = load_stimulus_set(root)
    # Every item pairs target 1 with context 1 and target 2 with context 2.
    # These are answer keys, not measured human responses.
    assembly = BehavioralAssembly(
        np.tile([1, 2], (len(stimuli), 1)),
        dims=['presentation', 'choice'],
        coords={
            'stimulus_id': ('presentation', stimuli['stimulus_id'].tolist()),
            'domain': ('presentation', stimuli['Domain'].tolist()),
            'family': ('presentation', stimuli['family'].tolist()),
            'version': ('presentation', stimuli['Version'].tolist()),
            'choice': ('choice', [1, 2]),
        },
    )
    assembly.attrs.update(stimulus_set=stimuli, target_type='correct_context_not_human_recordings')
    return assembly
