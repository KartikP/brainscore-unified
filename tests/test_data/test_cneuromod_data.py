"""The CNeuroMod data plugin on a tiny made-up source (no download): the shared part and two data sets."""
import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from brainscore.data import package_data
from brainscore.data.cneuromod.data_packaging import package, shared
from tests.cneuromod_stand_in import IMAGE_BYTES, make_source

pytestmark = pytest.mark.unit


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path / 'home'))
    return make_source(tmp_path / 'clones')


def _build(name, source, tmp_path, part=None):
    return package_data(name, source=source, output=tmp_path / 'out', part=part)


def test_each_tr_takes_the_event_that_holds_its_start_and_none_at_the_run_edges():
    events = pd.DataFrame({'onset': [0.0, 4.47, 12.0], 'duration': [2.98, 2.98, 5.0]})
    rows = shared.tr_events(9, 1.49, events)       # TR starts 0, 1.49, ..., 11.92
    assert rows.tolist() == [0, 0, -1, 1, 1, -1, -1, -1, -1]   # 2.98 ends trial 0; 12.0 starts after the last TR


def test_one_participant_writes_only_that_participant(source, tmp_path):
    out = _build('cneuromod-things', source, tmp_path, part='sub-02')
    assert sorted(p.name for p in out.glob('sub-*')) == ['sub-02']
    trials = xr.open_dataarray(out / 'sub-02' / 'ses-01' / 'trials.nc')
    assert trials.sizes['presentation'] == 6                      # 2 runs x 3 trials
    assert set(trials['region'].values) == {'V1', 'FFA'}          # masks of the visual atlases only
    assert trials['stimulus_id'].values[0] == 'img110'
    assert trials['repetition'].values.tolist()[:3] == [1, 2, 1]
    # The response of a voxel is the beta of its own grid column (V1 voxel (1,1,1) is grid column 0).
    v1 = trials.where(trials['region'] == 'V1', drop=True)
    assert int(v1['voxel'].values[0]) == 0 and float(v1.values[0, 0]) == 1.0
    assert (out / 'sub-02' / 'masks').is_dir() and (out / 'atlas' / 'parcels.csv').exists()


def test_one_session_and_an_absent_session(source, tmp_path):
    out = _build('cneuromod-things', source, tmp_path, part='sub-01/ses-01')
    assert (out / 'sub-01' / 'ses-01' / 'trials.nc').exists()
    with pytest.raises(ValueError, match='no ses-09'):
        package_data('cneuromod-things', source=source, output=tmp_path / 'other', part='sub-01/ses-09')


def test_sub04_is_refused_and_never_packaged(source, tmp_path):
    with pytest.raises(ValueError, match='restricted'):
        _build('cneuromod-movie10-timeseries', source, tmp_path, part='sub-04')
    out = package_data('cneuromod-movie10-timeseries', source=source, output=tmp_path / 'all')
    assert 'sub-04' not in {p.name for p in out.iterdir()}


def test_the_stimulus_table_holds_identifiers_and_no_file_content(source, tmp_path):
    out = _build('cneuromod-things', source, tmp_path)
    stimuli = pd.read_csv(out / 'sub-01' / 'ses-01' / 'stimuli.csv')
    assert stimuli['stimulus_id'].is_unique and 'img110' in set(stimuli['stimulus_id'])
    assert 'osf.io' in stimuli['source'].iloc[0]
    for path in out.rglob('*'):
        if path.is_file():
            assert path.suffix not in ('.jpg', '.png', '.mkv', '.mp4'), path
            assert IMAGE_BYTES not in path.read_bytes(), path


def test_a_second_data_set_uses_the_shared_part_unchanged(source, tmp_path):
    out = _build('cneuromod-movie10-timeseries', source, tmp_path, part='sub-01')
    series = xr.open_dataarray(out / 'sub-01' / 'ses-01' / 'timeseries_cneuromod2026.nc')
    bourne = series.where(series['task'] == 'bourne01', drop=True)
    assert bourne['stimulus_id'].values.tolist() == ['bourne01'] * 9      # 12 s of film hold all 9 TR starts
    life = series.where(series['task'] == 'life01', drop=True)
    assert set(life['run'].values) == {2}
    assert series['parcel_name'].values.tolist() == ['LH_Vis_1', 'HIP-head-m-rh']
    assert pd.read_csv(out / 'sub-01' / 'atlas' / 'parcels.csv')['parcel'].tolist() == [1, 1001]
    events = pd.read_csv(out / 'sub-01' / 'ses-01' / 'events.csv')
    assert events['repetition'].tolist() == [1, 2]


def test_a_new_data_set_needs_only_its_module(source, tmp_path, monkeypatch):
    """A module with the stated lines, and nothing else changed, packages through the shared part."""
    module = types.ModuleType('brainscore.data.cneuromod.data_packaging.made_up')
    module.NAME, module.REPOSITORY, module.TR_SECONDS = 'movie10', 'movie10', 1.49
    module.PARTICIPANTS = ('sub-01',)
    module.STIMULUS_SOURCE, module.STIMULUS_COLUMNS = 'made up', ()
    module.event_patterns = lambda participant, session: ['task-*_events.tsv']
    module.run_events = lambda clone, participant, run: pd.DataFrame(
        {'onset': [1.49], 'duration': [1.49], 'stimulus_id': ['made-up'], 'repetition': [1]})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    package.main(['--dataset', 'made_up', '--form', 'cneuromod2026', '--source', str(source),
                  '--out', str(tmp_path / 'out')])
    series = xr.open_dataarray(tmp_path / 'out' / 'sub-01' / 'ses-01' / 'timeseries_cneuromod2026.nc')
    assert series['stimulus_id'].values[:3].tolist() == ['', 'made-up', '']


def test_a_mask_on_another_grid_is_resampled_to_the_grid():
    import nibabel
    grid = nibabel.Nifti1Image(np.ones((2, 2, 2), np.uint8), np.diag([2.0, 2.0, 2.0, 1.0]))
    fine = np.zeros((4, 4, 4), np.uint8)
    fine[2:, :, :] = 1                                   # the half with x >= 2 mm
    mask = nibabel.Nifti1Image(fine, np.diag([1.0, 1.0, 1.0, 1.0]))
    assert shared.region_columns(mask, grid).tolist() == [4, 5, 6, 7]


def test_a_file_whose_content_is_not_downloaded_yet_is_listed(tmp_path):
    """An annexed file before ``datalad get`` is a link to absent content; it must be planned and fetched."""
    folder = tmp_path / 'things.glmsingle' / 'sub-01'
    folder.mkdir(parents=True)
    (folder / 'betas.h5').symlink_to('../.git/annex/objects/MD5E-s11315631976--2c46.h5')
    found = package._files(tmp_path, 'things.glmsingle', ['sub-01/betas.h5', 'sub-01/*.h5'])
    assert [path.name for path in found] == ['betas.h5', 'betas.h5']
    assert shared.annex_size(found[0]) == 11315631976


def test_two_builds_give_the_same_bytes(source, tmp_path):
    """The README states checksums, so a rebuild must give the same sha1 for every file."""
    import hashlib
    sums = []
    for name in ('first', 'second'):
        out = package_data('cneuromod-things', source=source, output=tmp_path / name, part='sub-01')
        sums.append({str(p.relative_to(out)): hashlib.sha1(p.read_bytes()).hexdigest()
                     for p in out.rglob('*') if p.is_file() and p.name != 'build.json'})
    assert sums[0] == sums[1] and any(name.endswith('.nc') for name in sums[0])
