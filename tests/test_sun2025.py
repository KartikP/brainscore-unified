"""Sun2025 data plugin on a made-up prepared build."""
import hashlib
import json

import pytest

import brainscore
from brainscore.data import local

pytestmark = pytest.mark.unit

HUMAN = {1: (2.0, -1.0), 2: (0.0, 3.0)}   # made-up (x, y) per modulation, equal for all participants


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def build(tmp_path):
    """A made-up build in the prepared form: scene 1, modulations 1 and 2, four flips each."""
    root = tmp_path / 'sun2025'
    (root / 'clips').mkdir(parents=True)
    stimuli, humans = [], []
    for modulation, (x, y) in HUMAN.items():
        base_id = f'sun2025-s01-m{modulation}-v1'
        for flip in (1, 2, 3, 4):
            name = f'clips/{base_id}-f{flip}.avi'
            (root / name).write_bytes(f'made-up clip {base_id} {flip}'.encode())
            stimuli.append(dict(stimulus_id=f'{base_id}-f{flip}', base_id=base_id, scene=1,
                                modulation=modulation, flip=flip, video_path=name))
        humans += [dict(base_id=base_id, scene=1, modulation=modulation, participant=p, x=x, y=y)
                   for p in (1, 2, 3)]
    (root / 'stimuli.json').write_text(json.dumps(stimuli))
    (root / 'human-responses.json').write_text(json.dumps(humans))
    files = {path.relative_to(root).as_posix(): _sha(path) for path in root.rglob('*') if path.is_file()}
    (root / 'manifest.json').write_text(json.dumps(dict(
        dataset='Sun2025-second-order-motion', schema_version=1, scenes=[1], complete=False,
        files=files, trials=len(stimuli), speed_units='displayed pixels per frame',
        coordinates='x right, y down')))
    return root


def test_missing_data_names_the_prepare_command(tmp_path, monkeypatch):
    monkeypatch.setenv('BRAINSCORE_SUN2025_ROOT', str(tmp_path / 'absent'))
    with pytest.raises(local.LocalDataMissing, match='python -m brainscore.data prepare Sun2025'):
        brainscore.load_stimulus_set('Sun2025-second-order-motion')


def test_a_changed_file_is_refused(build):
    (build / 'clips' / 'sun2025-s01-m1-v1-f1.avi').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Checksum'):
        brainscore.load_stimulus_set('Sun2025-second-order-motion', root=build)
