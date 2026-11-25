"""The CNeuroMod time series made from the fMRIPrep outputs, on tiny made-up files (no download).

The programs (git, DataLad, the separate environment and CNeuroMod's extraction code) are replaced
by a stand-in at the subprocess boundary; the stand-in extraction writes the files the code writes.
"""
import glob
import json
import subprocess
import sys
import types
from pathlib import Path

import h5py
import nibabel
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from brainscore.data.cneuromod.data_packaging import extraction, movie10, package, shared, things
from tests.cneuromod_stand_in import N_TRS, make_source

pytestmark = pytest.mark.unit

# The two parcellation configurations of the code at ae310a53, as they are.
CONFIGS = {
    'schaefer1000Tian50Nette128': """# @package _global_

space: MNI152NLin2009cAsym
use_template_gm: False
resample_gm: False
gm_path: ./atlases/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_sub-*_res-func_label-boldGMintersect_desc-fromFS_dseg.nii.gz
gm_name: indivBoldGMfromFS

build_func_mask: False

parcel_type: dseg
use_template_parcel: True
mask_parcel: True
parcellation: ./atlases/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_atlas-Schaefer2018TianS3NettekovenAsym_desc-1000Parcels7Networks50Subcort128Cereb_dseg.nii.gz

parcel_name: Schaefer2018TianS3NettekovenAsym
desc: 1000Parcels7Networks50Subcort128Cereb""",
    'gm_epi': """# @package _global_

space: T1w
use_template_gm: False
resample_gm: False
gm_path: ./atlases/tpl-sub*T1w/tpl-sub*T1w_res-func_label-boldGMintersect_desc-fromFS_dseg.nii.gz
gm_name: indivBoldGMfromFS

build_func_mask: False

parcel_type: mask
use_template_parcel: False
mask_parcel: True
parcellation: ./atlases/tpl-sub*T1w/tpl-sub*T1w_res-func_label-boldGMintersect_desc-fromFS_dseg.nii.gz

parcel_name: Freesurfer
desc: BoldGM_res-func""",
}
REQUIREMENTS = '-e .\n\n# external requirements\nclick\nawscli\nflake8\npython-dotenv>=0.5.1\n\nh5py==3.10.0\nnumpy==1.25.2\n'
FMRIPREP_PIN = '5155ba0d43836a4451141b90b5a70507f41205cf'
ATLASES_PIN = '0b78173e9addc2bbd793b64c8b7184d07355c321'
# A THINGS run: part-mag and a run number; a movie10 run: neither.
BOLD = {'things': '{p}_{s}_task-things_run-{r}_part-mag_space-{space}_desc-preproc_bold.nii.gz',
        'movie10': '{p}_{s}_task-bourne0{r}_space-{space}_desc-preproc_bold.nii.gz'}


def _nifti(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nibabel.Nifti1Image(np.asarray(data, dtype=np.int16), np.eye(4)).to_filename(path)


def _annexed(path: Path, write) -> None:
    """An annexed file whose content is not here: a dangling link; the stand-in ``datalad get`` fills it."""
    content = path.parent / f'.content-{path.name}'
    write(content)
    path.symlink_to(f'{content}.absent')


def make_tree(root: Path, dataset: str, sessions=('ses-01',), non_steady=()) -> Path:
    """The clones the builder reads: code, atlases, fMRIPrep outputs (annexed, not downloaded yet). The runs
    in ``non_steady`` have their first volume marked non-steady in the confounds table, as fMRIPrep marks it."""
    base = root / 'extraction'
    code = base / extraction.CODE
    (code / '.git').mkdir(parents=True, exist_ok=True)
    (code / 'atlases').mkdir(exist_ok=True)
    (code / 'requirements.txt').write_text(REQUIREMENTS)
    for name, text in CONFIGS.items():
        (path := code / 'timeseries' / 'config' / 'parcellation' / f'{name}.yaml').parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for name, text in CONFIGS.items():
        for key in ('gm_path', 'parcellation'):
            atlas = extraction.read_config(code / 'timeseries' / 'config' / 'parcellation' / f'{name}.yaml')[key]
            path = base / shared.ATLASES / Path(atlas.replace('*', '01')).relative_to('atlases')
            if not path.is_symlink():
                path.parent.mkdir(parents=True, exist_ok=True)
                _annexed(path, lambda content: _nifti(np.ones((2, 2, 2)), content))
    for folder in (base / shared.ATLASES, base / 'sourcedata' / f'{dataset}.fmriprep'):
        (folder / '.git').mkdir(parents=True, exist_ok=True)
        (folder / '.datalad').mkdir(exist_ok=True)
    (root / f'{dataset}.timeseries' / '.git').mkdir(parents=True, exist_ok=True)
    for session in sessions:
        func = base / 'sourcedata' / f'{dataset}.fmriprep' / 'sub-01' / session / 'func'
        for run in (1, 2):
            for space in ('T1w', 'MNI152NLin2009cAsym'):
                bold = func / BOLD[dataset].format(p='sub-01', s=session, r=run, space=space)
                bold.parent.mkdir(parents=True, exist_ok=True)
                _annexed(bold, nibabel.Nifti1Image(np.zeros((2, 2, 2, N_TRS), np.float32), np.eye(4)).to_filename)
                _annexed(bold.with_name(bold.name.replace('_desc-preproc_bold', '_desc-brain_mask')),
                         lambda content: _nifti(np.ones((2, 2, 2)), content))
            flags = [int(volume == 0 and run in non_steady) for volume in range(N_TRS)]
            _annexed(func / (bold.name.split('_space-')[0] + '_desc-confounds_timeseries.tsv'),
                     lambda content, flags=flags: content.write_text(
                         'a\tnon_steady_state_outlier00\n' + ''.join(f'0.5\t{flag}\n' for flag in flags)))
    return base


def stand_in(monkeypatch, code_commit=extraction.CODE_COMMIT, short=(), missing=(), private=()):
    """Replace ``subprocess.run``. The extraction writes, as the code does, the participant's atlas or mask
    and one HDF5 dataset per present BOLD run not yet in the file; ``short`` runs lose a row, ``missing``
    runs are not written (the code catches the error of a run and goes on)."""
    calls = []

    def run(argv, *args, **kwargs):
        argv = [str(word) for word in argv]
        calls.append(types.SimpleNamespace(argv=argv, cwd=kwargs.get('cwd'), env=kwargs.get('env')))
        out = ''
        if argv[:3] == ['git', 'annex', 'whereis']:        # on the public store unless its session is private
            out = ''.join(json.dumps({'file': f, 'whereis': [] if any(f'/{p}/' in f for p in private)
                                      else [{'description': 'conp-ria-storage [conp-ria-storage-http]'}]}) + '\n'
                          for f in argv[4:])
        elif argv[:2] == ['git', '-C'] and argv[3] == 'ls-tree':
            path = argv[5]
            out = f'160000 commit {code_commit if path.startswith("code/") else ATLASES_PIN if path == "atlases" else FMRIPREP_PIN}\t{path}\n'
        elif argv[1:3] == ['-m', 'venv']:
            Path(argv[-1], 'bin').mkdir(parents=True, exist_ok=True)
        elif argv[:2] == ['datalad', 'get']:
            for relative in argv[2:]:
                path = Path(kwargs['cwd']) / relative
                content = Path(str(path.readlink()).removesuffix('.absent'))
                path.unlink()
                path.write_bytes(content.read_bytes())
        elif argv[:4] == ['datalad', 'drop', '--reckless', 'availability']:
            for relative in argv[4:]:           # the content goes; the annex link stays
                path = Path(kwargs['cwd']) / relative
                path.unlink()
                path.symlink_to('.git/annex/objects/gone')
        elif argv[1:3] == ['-m', 'timeseries.run']:
            _extract(dict(word.split('=', 1) for word in argv[3:]), short, missing)
        return types.SimpleNamespace(returncode=0, stdout=out, stderr='')
    monkeypatch.setattr(subprocess, 'run', run)
    return calls


def _extract(options: dict, short, missing) -> None:
    space = 'MNI152NLin2009cAsym' if options['parcellation'] == 'schaefer1000Tian50Nette128' else 'T1w'
    dataset, subject = options['dataset'], options['subject_list'].strip('[]"')
    func = Path(options['output_dir'].strip("'")) / f'sub-{subject}' / 'func'
    func.mkdir(parents=True, exist_ok=True)
    stem = f'sub-{subject}_task-{dataset}_space-{space}'
    _nifti(np.ones((2, 2, 2)), func / f'{stem}_label-GM_desc-indivBoldGMfromFS_mask.nii.gz')
    if space == 'T1w':
        atlas, width = f'{stem}_atlas-Freesurfer_label-BoldGM_res-func', 3
        _nifti(np.arange(8).reshape(2, 2, 2) > 4, func / f'{atlas}_mask.nii.gz')
    else:
        atlas, width = f'{stem}_atlas-Schaefer2018TianS3NettekovenAsym_desc-1000Parcels7Networks50Subcort128Cereb', 2
        _nifti([[[1, 1], [1, 1]], [[1001, 1001], [0, 0]]], func / f'{atlas}_dseg.nii.gz')
    fmriprep = Path(options['data_dir'].strip("'")) / f'{dataset}.fmriprep' / f'sub-{subject}'
    with h5py.File(func / f'{atlas}_timeseries.h5', 'a') as handle:
        for bold in sorted(Path(path) for path in glob.glob(f'{fmriprep}/ses-*/func/*_space-{space}_desc-preproc_bold.nii.gz')):
            if not bold.exists():
                continue                        # not downloaded
            run = extraction.run_name(bold)
            group = handle.require_group(run.split('_')[0])
            if f'{run}_timeseries' in group or run in missing:
                continue
            confounds = pd.read_csv(extraction.confounds_of(bold), sep='\t')     # non-steady volumes are left out
            rows = nibabel.load(bold).shape[3] - int(confounds['non_steady_state_outlier00'].sum()) - (run in short)
            group.create_dataset(f'{run}_timeseries', data=np.zeros((rows, width), np.float32))


def _gets(calls) -> set:
    return {word for call in calls if call.argv[:2] == ['datalad', 'get'] for word in call.argv[2:]}


@pytest.mark.parametrize('dataset, module', [('things', things), ('movie10', movie10)])
@pytest.mark.parametrize('form', ['cneuromod2026', 'voxel_native'])
def test_the_files_of_their_configuration_are_fetched_and_the_output_has_the_released_names(
        tmp_path, monkeypatch, dataset, module, form):
    make_tree(tmp_path, dataset)
    calls = stand_in(monkeypatch)
    files = extraction.build(module, form, tmp_path, 'sub-01', None)
    space = {'cneuromod2026': 'MNI152NLin2009cAsym', 'voxel_native': 'T1w'}[form]
    atlas = ({'tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_sub-01_res-func_label-boldGMintersect_desc-fromFS_dseg.nii.gz',
              'tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_atlas-Schaefer2018TianS3NettekovenAsym_'
              'desc-1000Parcels7Networks50Subcort128Cereb_dseg.nii.gz'} if form == 'cneuromod2026'
             else {'tpl-sub01T1w/tpl-sub01T1w_res-func_label-boldGMintersect_desc-fromFS_dseg.nii.gz'})
    runs = [BOLD[dataset].format(p='sub-01', s='ses-01', r=run, space=space) for run in (1, 2)]
    per_run = {f'sub-01/ses-01/func/{name}' for bold in runs for name in (
        bold, bold.replace('_desc-preproc_bold', '_desc-brain_mask'),
        bold.split('_space-')[0] + '_desc-confounds_timeseries.tsv')}
    assert _gets(calls) == atlas | per_run
    stem = f'sub-01_task-{dataset}_space-{space}'
    released = ({f'{stem}_atlas-cneuromod26_desc-1134Parcels_timeseries.h5',
                 f'{stem}_atlas-cneuromod26_desc-1134Parcels_dseg.nii.gz'} if form == 'cneuromod2026'
                else {f'{stem}_desc-voxelwise_timeseries.h5'}) | {f'{stem}_label-GMfromFS_desc-indivFunc_mask.nii.gz'}
    assert {path.name for path in files} == released
    assert all(path.parent.parent.name == form for path in files)       # the reader takes the form from here


def test_the_code_runs_in_its_own_environment_with_its_pinned_versions_only(tmp_path, monkeypatch):
    base = make_tree(tmp_path, 'things')
    monkeypatch.setenv('PYTHONPATH', '/the/caller/packages')
    calls = stand_in(monkeypatch)
    extraction.build(things, 'cneuromod2026', tmp_path, 'sub-01', 'ses-01')
    venv, = [call for call in calls if call.argv[1:3] == ['-m', 'venv']]
    pip, = [call for call in calls if call.argv[1:4] == ['-m', 'pip', 'install']]
    code, = [call for call in calls if call.argv[1:3] == ['-m', 'timeseries.run']]
    assert venv.argv[0] == sys.executable and Path(venv.argv[-1]) == base / 'env'
    assert pip.argv[0] == code.argv[0] == str(base / 'env' / 'bin' / 'python')     # never the caller's Python
    assert [word for word in pip.argv if '==' in word or word in ('awscli', 'click', 'flake8')] == \
        ['h5py==3.10.0', 'numpy==1.25.2']
    for call in (pip, code):
        assert 'PYTHONPATH' not in call.env
        assert call.env['PIP_CACHE_DIR'] == str(base / 'pip-cache') and call.env['TMPDIR'] == str(base / 'tmp')
    assert Path(code.cwd) == base / extraction.CODE          # the configuration's atlas paths are relative to it
    assert (base / extraction.CODE / 'atlases').resolve() == (base / shared.ATLASES).resolve()
    stand_in(monkeypatch)
    extraction.build(things, 'cneuromod2026', tmp_path, 'sub-01', 'ses-01')
    assert not [call for call in calls if call.argv[1:3] == ['-m', 'venv']][1:]      # made once


@pytest.mark.parametrize('fault', ['short', 'missing'])
def test_a_short_or_missing_run_raises_with_its_name(tmp_path, monkeypatch, fault):
    make_tree(tmp_path, 'things')
    stand_in(monkeypatch, **{fault: ('ses-01_task-things_run-2',)})
    with pytest.raises(RuntimeError, match=f'ses-01_task-things_run-2: {"8 rows" if fault == "short" else "missing"}'):
        extraction.build(things, 'cneuromod2026', tmp_path, 'sub-01', None)


def test_a_finished_session_is_skipped_and_its_bold_files_dropped(tmp_path, monkeypatch):
    base = make_tree(tmp_path, 'things', sessions=('ses-01', 'ses-02'))
    calls = stand_in(monkeypatch)
    extraction.build(things, 'cneuromod2026', tmp_path, 'sub-01', 'ses-01')
    bold = base / 'sourcedata' / 'things.fmriprep' / 'sub-01' / 'ses-01' / 'func' / BOLD['things'].format(
        p='sub-01', s='ses-01', r=1, space='MNI152NLin2009cAsym')
    assert bold.is_symlink() and not bold.exists()          # dropped after the check
    del calls[:]
    files = extraction.build(things, 'cneuromod2026', tmp_path, 'sub-01', None)
    assert [call for call in calls if call.argv[1:3] == ['-m', 'timeseries.run']]
    assert not {path for path in _gets(calls) if '/ses-01/' in path}       # nothing of ses-01 fetched again
    with h5py.File(next(path for path in files if path.suffix == '.h5'), 'r') as handle:
        assert sorted(handle) == ['ses-01', 'ses-02'] and len(handle['ses-02']) == 2


def test_a_session_on_no_public_store_is_left_out_and_named(tmp_path, monkeypatch, capsys):
    make_tree(tmp_path, 'things', sessions=('ses-01', 'ses-02'))
    calls = stand_in(monkeypatch, private=('ses-02',))
    files = extraction.build(things, 'cneuromod2026', tmp_path, 'sub-01', None)
    assert not {path for path in _gets(calls) if '/ses-02/' in path}        # nothing of it fetched
    with h5py.File(next(path for path in files if path.suffix == '.h5'), 'r') as handle:
        assert sorted(handle) == ['ses-01']
    assert 'no public store' in capsys.readouterr().out
    record = tmp_path / 'extraction' / 'output' / 'things' / 'cneuromod2026' / 'not_public.txt'
    assert record.read_text().split()[:2] == ['sub-01', 'ses-02']


def test_another_code_commit_is_refused(tmp_path, monkeypatch):
    make_tree(tmp_path, 'things')
    stand_in(monkeypatch, code_commit='f957fb46b29b7f1cb2f5a5c3d92691201c5961a4')
    with pytest.raises(RuntimeError, match='f957fb46'):
        extraction.build(things, 'cneuromod2026', tmp_path, 'sub-01', None)


def test_a_release_commit_that_is_not_public_is_made_with_the_pinned_commit_when_the_data_set_names_it(
        tmp_path, monkeypatch, capsys):
    from brainscore.data.cneuromod.data_packaging import mario
    assert mario.UNPUBLISHED_RELEASE_CODE == '4492c476e8d94d4c302046fc12df5ec9a67da03d'   # as mario.timeseries names it
    monkeypatch.setitem(BOLD, 'mario', '{p}_{s}_task-mario_run-{r}_space-{space}_desc-preproc_bold.nii.gz')
    make_tree(tmp_path, 'mario')
    calls = stand_in(monkeypatch, code_commit=mario.UNPUBLISHED_RELEASE_CODE)
    files = extraction.build(mario, 'cneuromod2026', tmp_path, 'sub-01', None)
    assert any(path.name.endswith('_timeseries.h5') for path in files)
    assert any(extraction.CODE_COMMIT in call.argv for call in calls if 'checkout' in call.argv)
    assert 'not public' in capsys.readouterr().out
    with pytest.raises(RuntimeError, match='4492c476'):         # only the data set that names it
        extraction.sources(things, tmp_path)


def test_package_data_reads_the_made_time_series_with_the_per_tr_reader(tmp_path, monkeypatch):
    """Run 2 has a non-steady first volume, which the code leaves out: its rows are volumes 1 to 8."""
    from brainscore.data import package_data, packaging
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path / 'home'))
    monkeypatch.setattr(packaging, '_expected', lambda name, builder: {})
    monkeypatch.setattr(shared, 'check_prerequisites', lambda: None)
    root = make_source(tmp_path / 'home' / 'sources' / 'cneuromod')
    make_tree(root, 'things', non_steady=(2,))
    calls = stand_in(monkeypatch)
    out = package_data('cneuromod-things-timeseries', output=tmp_path / 'out', part='sub-01/ses-01')
    enabled = {Path(call.argv[3]).name for call in calls if call.argv[:2] == ['datalad', 'siblings']}
    assert 'things.timeseries' not in enabled     # its files are not on the public store: read for its commits
    series = xr.open_dataarray(out / 'sub-01' / 'ses-01' / 'timeseries_cneuromod2026.nc')
    assert series.shape == (2 * N_TRS - 1, 2)                      # 2 runs x their kept volumes, 2 parcels
    second = series.sel(presentation=series['run'] == 2)
    assert second['tr'].values.tolist() == list(range(1, N_TRS))   # each row at the start of its own volume
    assert np.allclose(second['time'].values, np.arange(1, N_TRS) * 1.49)
    assert (out / 'sub-01' / 'atlas' / 'parcels.csv').read_text().splitlines()[1:] == ['1,LH_Vis_1', '1001,HIP-head-m-rh']
    assert package.extraction is extraction
