"""The CNeuroMod data plugin on a tiny made-up source (no download): the shared part and two data sets."""
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from brainscore.data import package_data
from brainscore.data.cneuromod.data_packaging import package, shared, things
from tests.cneuromod_stand_in import IMAGE_BYTES, make_source

pytestmark = pytest.mark.unit


@pytest.fixture
def source(tmp_path, monkeypatch):
    from brainscore.data import packaging
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path / 'home'))
    monkeypatch.setattr(packaging, '_expected', lambda name, builder: {})   # made-up data: no recorded sums
    return make_source(tmp_path / 'clones')


def _build(name, source, tmp_path, part=None):
    return package_data(name, source=source, output=tmp_path / 'out', part=part)


def test_each_tr_takes_the_event_that_holds_its_start_and_none_at_the_run_edges():
    events = pd.DataFrame({'onset': [0.0, 4.47, 12.0], 'duration': [2.98, 2.98, 5.0]})
    rows = shared.tr_events(np.arange(9), 1.49, events)       # TR starts 0, 1.49, ..., 11.92
    assert rows.tolist() == [0, 0, -1, 1, 1, -1, -1, -1, -1]   # 2.98 ends trial 0; 12.0 starts after the last TR


def test_one_participant_writes_only_that_participant(source, tmp_path):
    out = _build('cneuromod-things', source, tmp_path, part='sub-02')
    assert sorted(p.name for p in out.glob('sub-*')) == ['sub-02']
    trials = xr.open_dataarray(out / 'sub-02' / 'ses-01' / 'trials.nc')
    assert trials.sizes['presentation'] == 6                      # 2 runs x 3 trials
    assert set(trials['region'].values) == {'V1', 'FFA', 'cortex'}   # visual masks and the cortex rows
    assert trials['stimulus_id'].values[0] == 'img110'
    assert trials['repetition'].values.tolist()[:3] == [1, 2, 1]
    # The response of a voxel is the beta of its own grid column (V1 voxel (1,1,1) is grid column 0).
    v1 = trials.where(trials['region'] == 'V1', drop=True)
    assert int(v1['voxel'].values[0]) == 0 and float(v1.values[0, 0]) == 1.0
    assert (out / 'sub-02' / 'masks').is_dir() and (out / 'atlas' / 'parcels.csv').exists()


def test_each_region_has_one_mask_set_and_the_group_parcels_only_without_an_own_mask(source, tmp_path):
    out = _build('cneuromod-things', source, tmp_path)
    for participant, ffa_atlas in (('sub-01', 'fLocVisionTask'), ('sub-06', 'fLocVisionKanwisher')):
        trials = xr.open_dataarray(out / participant / 'ses-01' / 'trials.nc')
        regions = pd.DataFrame({name: trials[name].values for name in ('region', 'atlas')}).drop_duplicates()
        assert regions['region'].is_unique
        assert regions.set_index('region').loc['FFA', 'atlas'] == ffa_atlas


def test_the_cortex_rows_are_the_cortical_labels_of_the_segmentation_on_the_trial_grid(source, tmp_path):
    out = _build('cneuromod-things', source, tmp_path, part='sub-01')
    trials = xr.open_dataarray(out / 'sub-01' / 'ses-01' / 'trials.nc')
    cortex = trials.isel(neuroid=trials['region'].values == 'cortex')
    assert set(cortex['atlas'].values) == {'aparcaseg'}
    assert sorted(zip(cortex['i'].values, cortex['j'].values, cortex['k'].values)) == [(1, 1, 1), (1, 2, 1), (2, 1, 1)]
    # The response of a cortex voxel is the beta of its own grid column.
    np.testing.assert_array_equal(cortex.values[:3], np.arange(24, dtype=np.float32).reshape(3, 8)[:, cortex['voxel'].values] + 1)


def test_a_segmentation_on_another_grid_is_refused(tmp_path):
    nibabel = pytest.importorskip('nibabel')
    grid = nibabel.Nifti1Image(np.ones((2, 2, 2), np.uint8), np.eye(4))
    nibabel.Nifti1Image(np.full((3, 3, 3), 1001, np.int16), np.eye(4)).to_filename(tmp_path / 'dseg.nii.gz')
    with pytest.raises(ValueError, match='not on the grid'):
        things.cortex_neuroids(tmp_path / 'dseg.nii.gz', grid)


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
    nibabel = pytest.importorskip('nibabel')
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


def test_an_assembly_has_the_same_bytes_under_any_hash_seed(tmp_path):
    """Two processes with other string hash seeds (as two separate builds are) write the same file."""
    import hashlib
    import os
    import subprocess
    code = ('import sys, numpy as np, pandas as pd; from pathlib import Path; '
            'from brainscore.data.cneuromod.data_packaging import shared; '
            "p = pd.DataFrame({name: ['a', 'b'] for name in 'abcdefghij'}); "
            "n = pd.DataFrame({name: [1, 2, 3] for name in 'klmnopqrst'}); "
            'shared.write_assembly(np.zeros((2, 3)), p, n, Path(sys.argv[1]))')
    digests = set()
    for seed in ('1', '2', '3'):
        path = tmp_path / f'{seed}.nc'
        subprocess.run([sys.executable, '-W', 'ignore', '-c', code, str(path)], check=True,
                       env={**os.environ, 'PYTHONHASHSEED': seed}, capture_output=True)
        digests.add(hashlib.sha1(path.read_bytes()).hexdigest())
    assert len(digests) == 1


def _repeated_trials(n_images=200, seed=0):
    """Each image 3 times, trials shuffled in time; voxel 0 all signal, voxel 1 all noise, voxel 2 equal parts."""
    rng = np.random.default_rng(seed)
    image = np.repeat([f'img{i:03d}' for i in range(n_images)], 3)
    rng.shuffle(image)
    signal = pd.Series(rng.standard_normal(n_images), index=[f'img{i:03d}' for i in range(n_images)])[image].to_numpy()
    noise = rng.standard_normal((len(image), 2))
    return np.column_stack([signal, noise[:, 0], signal + noise[:, 1]]), image


def test_the_noise_ceiling_follows_the_signal_share_and_ignores_a_shift_and_scale_per_voxel():
    from brainscore.benchmarks import noise_ceiling
    responses, image = _repeated_trials()
    ceiling, used = noise_ceiling(responses, image, things.answered_viewing(image, np.ones(len(image))))
    assert used == 200
    assert ceiling[0] == pytest.approx(100) and ceiling[1] < 5
    assert ceiling[2] == pytest.approx(75, abs=10)          # ncsnr 1: 100 * 1 / (1 + 1/3)
    viewing = things.answered_viewing(image, np.ones(len(image)))
    moved, _ = noise_ceiling(responses * [2, 3, 4] + [5, -1, 7], image, viewing)
    assert moved == pytest.approx(ceiling)


def test_the_noise_ceiling_leaves_out_an_image_with_a_trial_without_response():
    from brainscore.benchmarks import noise_ceiling
    responses, image = _repeated_trials()
    answered = np.ones(len(image), bool)
    answered[0] = False
    _, used = noise_ceiling(responses, image, things.answered_viewing(image, np.where(answered, 1.0, np.nan)))
    assert used == 199


def test_the_noise_ceiling_takes_the_answered_trials_of_an_image_shown_more_than_3_times():
    from brainscore.benchmarks import noise_ceiling
    responses, image = _repeated_trials()
    shown_again = image == image[0]
    extra = np.where(shown_again)[0][0]
    responses = np.vstack([responses[:extra], [[99.0, 99.0, 99.0]], responses[extra:]])   # a 4th trial, no response
    image = np.insert(image, extra, image[0])
    answered = np.ones(len(image), bool)
    answered[extra] = False
    ceiling, used = noise_ceiling(responses, image, things.answered_viewing(image, np.where(answered, 1.0, np.nan)))
    assert used == 200
    assert ceiling[0] == pytest.approx(100)


def test_an_event_value_is_written_with_the_digits_of_the_source_file(tmp_path):
    """The default pandas float parser of some versions changes the last digit (2.3210708713531574 read as
    2.321070871353158): the written event table must not depend on the pandas version."""
    from brainscore.data.cneuromod.data_packaging import things
    func = tmp_path / 'sub-01' / 'ses-001' / 'func'
    func.mkdir(parents=True)
    (func / 'sub-01_ses-001_task-things_run-1_events.tsv').write_text(
        'onset\tduration\timage_path\trepetition\tresponse_time\n'
        '3.0\t2.98\timages_fmri/girl/girl_02s.jpg\t1\t2.3210708713531574\n')
    events = shared.run_table(things, tmp_path, 'sub-01', {'session': 1, 'task': 'things', 'run': 1})
    shared.write_csv(events, tmp_path / 'events.csv')
    assert ',2.3210708713531574,' in (tmp_path / 'events.csv').read_text()


_REFUSAL = ('enable-sibling(impossible): . (sibling) [git-annex does not use special remote conp-ria-storage-http '
            'until its configuration is verified, and there is no interactive session to confirm it:\n'
            'This is the configuration of the remote:\n\ttype=httpalso cost=50 url={url} autoenable=true\n'
            'If so, run:  git-annex verifyremote conp-ria-storage-http --safe 5\n'
            'If not, run: git-annex verifyremote conp-ria-storage-http --unsafe]')


def _enable_runs(monkeypatch, refusals):
    """Stand-in for the programs of ``shared.clone``: the first ``len(refusals)`` enables print a refusal."""
    import subprocess
    calls, left = [], list(refusals)

    def run(argv, *args, **kwargs):
        calls.append(list(argv))
        refused = argv[:2] == ['datalad', 'siblings'] and left
        return types.SimpleNamespace(returncode=1 if refused else 0, stdout=left.pop(0) if refused else '', stderr='')
    monkeypatch.setattr(subprocess, 'run', run)
    return calls


@pytest.mark.parametrize('refusals, verified', [([], False), ([_REFUSAL.format(
    url='https://sftp.conp.ca/users/cneuromod/ria-conp/4de/annex/objects/')], True)])
def test_the_public_store_is_confirmed_only_when_git_annex_asks(tmp_path, monkeypatch, refusals, verified):
    (tmp_path / 'things' / '.git').mkdir(parents=True)
    calls = _enable_runs(monkeypatch, refusals)
    shared.clone(tmp_path, 'things')
    verifies = [call for call in calls if call[:2] == ['git-annex', 'verifyremote']]
    assert verifies == ([['git-annex', 'verifyremote', 'conp-ria-storage-http', '--safe', '5']] if verified else [])
    assert calls[-1][:2] == ['datalad', 'siblings']       # enabled after the confirmation


def test_a_store_at_another_address_is_not_confirmed(tmp_path, monkeypatch):
    (tmp_path / 'things' / '.git').mkdir(parents=True)
    calls = _enable_runs(monkeypatch, [_REFUSAL.format(url='https://example.org/annex/objects/')])
    with pytest.raises(RuntimeError, match='example.org'):
        shared.clone(tmp_path, 'things')
    assert not [call for call in calls if call[:2] == ['git-annex', 'verifyremote']]


@pytest.mark.parametrize('version, accepted', [('10.20230126', True), ('10.20261026-g1a2b', True),
                                               ('8.20210223', False), ('', False)])
def test_git_annex_older_than_10_20230126_is_refused_with_the_remedy(monkeypatch, version, accepted):
    monkeypatch.setattr(shared.shutil, 'which', lambda program: f'/bin/{program}')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=version, stderr=''))
    if accepted:
        shared.check_prerequisites()
    else:
        with pytest.raises(RuntimeError, match='pipx inject'):
            shared.check_prerequisites()


def test_an_unlocked_annexed_file_without_its_content_is_fetched_and_sized(tmp_path, monkeypatch):
    """Some CNeuroMod text files are unlocked: before the download the file holds its annex key."""
    (tmp_path / '.datalad').mkdir()
    pointer, here = tmp_path / 'run-1_desc-confounds_timeseries.tsv', tmp_path / 'events.tsv'
    pointer.write_text('/annex/objects/MD5E-s718296--65d8a978149c2772ac92fb2a3569070b.tsv\n')
    here.write_text('onset\tduration\n')
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda argv, **kwargs: calls.append(list(argv)))
    shared.get([pointer, here])
    assert calls == [['datalad', 'get', pointer.name]]
    assert shared.annex_size(pointer) == 718296
    with pytest.raises(FileNotFoundError, match='confounds'):
        shared.existing([pointer])


def test_the_viewing_order_counts_the_answered_trials_of_an_image_in_time_and_not_the_released_label():
    """The released ``repetition`` can be 4 or 5 (out-of-order sessions); the viewing order is 1, 2, 3 in time,
    over answered trials only."""
    from brainscore.data.cneuromod.data_packaging import things
    image = np.array(['a', 'b', 'a', 'a', 'b', 'a'])
    response = np.array([1.0, -1.0, np.nan, 2.0, 1.0, -2.0])     # the 2nd trial of 'a' has no response
    assert things.answered_viewing(image, response).tolist() == [1, 1, 0, 2, 2, 3]


def test_a_mario_tr_has_its_repetition_its_video_frames_and_the_seconds_each_button_was_held(source, tmp_path):
    out = _build('cneuromod-mario-timeseries', source, tmp_path, part='sub-01')
    series = xr.open_dataarray(out / 'sub-01' / 'ses-01' / 'timeseries_cneuromod2026.nc')
    # TR starts 0, 1.49, ..., 11.92; w1l1 holds [0.5, 6.5), w1l2 [8, 12)
    assert series['stimulus_id'].values[[0, 1, 5, 6]].tolist() == [
        '', 'sub-01_ses-001_task-mario_level-w1l1_rep-000', '', 'sub-01_ses-001_task-mario_level-w1l2_rep-001']
    assert series['level'].values[[0, 1, 6]].tolist() == ['', 'w1l1', 'w1l2']
    assert (series['frame_start'].values[1], series['frame_stop'].values[1]) == (59, 148)   # 60 frames per second
    assert series['frame_stop'].values[4] == 360          # the TR ends after its repetition: its last frame
    assert series['frame_start'].values[0] == -1
    assert series['button_A'].values[:3] == pytest.approx([0.49, 0.51, 0.0])
    assert series['button_RIGHT'].values[5] == pytest.approx(0.2)
    assert series['presses_A'].values[:3].tolist() == [1, 0, 0]               # a press counts in the TR where it starts
    stimuli = pd.read_csv(out / 'sub-01' / 'ses-01' / 'stimuli.csv')
    assert stimuli['replay_video'].tolist() == [f'sub-01/ses-001/beh/videos/{name}.mp4'
                                               for name in stimuli['stimulus_id']]
    assert set(stimuli['level']) == {'w1l1', 'w1l2'}


@pytest.mark.skipif(__import__('shutil').which('ffmpeg') is None, reason='ffmpeg is needed')
def test_the_mario_media_of_a_tr_are_its_frames_its_middle_frame_and_the_sound_of_the_same_frames(source, tmp_path):
    import cv2
    from scipy.io import wavfile
    from brainscore.data.cneuromod.data_packaging import mario
    out = _build('cneuromod-mario-timeseries', source, tmp_path, part='sub-01')
    replays = tmp_path / 'replays'
    for name, seconds in (('w1l1_rep-000', 6), ('w1l2_rep-001', 4)):     # 60 frames per second, with sound
        video = replays / 'sub-01' / 'ses-001' / 'beh' / 'videos' / f'sub-01_ses-001_task-mario_level-{name}.mp4'
        video.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'testsrc=size=240x224:rate=60:duration={seconds}',
                        '-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate=32000:duration={seconds}',
                        '-shortest', str(video)], check=True)
    table = pd.read_csv(mario.write_media(out, 'sub-01', replays))
    assert table['tr'].tolist() == [1, 2, 3, 4, 6, 7, 8]                  # the TRs that show a repetition
    from brainscore.data.cneuromod.data import load_mario_media
    media = load_mario_media('sub-01', out)
    assert all(path.endswith('.png') for path in media.stimulus_paths.values())    # a picture extractor reads these
    first = table.iloc[0]
    capture = cv2.VideoCapture(str(out / first['video_path']))
    assert first['frames'] == 148 - 59
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == mario.CLIP_FRAMES == 89   # every clip lasts one TR
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(60)
    assert cv2.imread(str(out / first['image_path'])).shape == (224, 240, 3)
    rate, sound = wavfile.read(out / first['audio_path'])
    assert rate == 16000 and len(sound) == 89 * 16000 // 60 and np.abs(sound).max() > 0
    last = table.loc[table['tr'] == 4].iloc[0]
    assert last['frames'] == 360 - 327                                    # the play ends inside the TR
    _, sound = wavfile.read(out / last['audio_path'])
    assert np.abs(sound[last['frames'] * 16000 // 60 + 1:]).max() == 0     # silent after the play
    (out / first['video_path']).unlink()
    mario.main(['--packaged', str(out), '--part', 'sub-01', '--replays', str(replays)])
    assert not (out / first['video_path']).exists()                       # a finished participant is skipped


def test_the_mario_hand_contrast_is_positive_where_the_left_thumb_drives_the_parcel(tmp_path):
    from brainscore.data.cneuromod.validation import mario_checks
    from brainscore.temporal import hrf_convolve
    rng = np.random.default_rng(0)
    n = 200
    left, right = rng.integers(0, 4, n) * (rng.random(n) < 0.5), rng.integers(0, 4, n) * (rng.random(n) < 0.5)
    presentation = pd.DataFrame({'session': 1, 'run': np.repeat([1, 2], n // 2), 'tr': np.tile(np.arange(n // 2), 2),
                                 'frame_start': 0, 'frame_stop': 89, **{f'presses_{b}': 0 for b in 'A B UP DOWN LEFT RIGHT'.split()}})
    presentation['presses_LEFT'], presentation['presses_A'] = left, right
    drive = {'7Networks_RH_SomMot_1': left, '7Networks_LH_SomMot_1': right, '7Networks_LH_Vis_1': np.zeros(n)}
    values = np.stack([np.concatenate([hrf_convolve(d[:n // 2, None], 1 / 1.49)[:, 0], hrf_convolve(d[n // 2:, None], 1 / 1.49)[:, 0]])
                       for d in drive.values()], axis=1) + 0.1 * rng.standard_normal((n, 3))
    neuroid = pd.DataFrame({'parcel': [1, 2, 3], 'parcel_name': list(drive)})
    shared.write_assembly(values, presentation, neuroid, tmp_path / 'sub-01' / 'ses-01' / 'timeseries_cneuromod2026.nc')
    table = mario_checks.glm(tmp_path, tmp_path / 'out').set_index('hemisphere')
    assert table.loc['RH', 'mean_t'] > 5 and table.loc['LH', 'mean_t'] < -5
    assert mario_checks.network('7Networks_LH_DorsAttn_Post_3') == 'DorsAttn' and mario_checks.network('HIP-head-m-rh') != 'HIP'
