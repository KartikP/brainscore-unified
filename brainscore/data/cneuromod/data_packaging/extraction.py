"""The CNeuroMod time series made from the public fMRIPrep outputs, as CNeuroMod made its release.

``build`` runs CNeuroMod's own extraction code (``cneuromod_extract_tseries``, MIT licence) at the
commit that the released time series name, in a separate Python environment that it makes and
never in the caller's. It works one session at a time: it downloads the session's fMRIPrep
outputs, extracts, checks each run, then drops the session's BOLD files, so a participant larger
than the disk fits. A run already in the HDF5 file is not extracted again. The files get the
released names, and the per-TR reader of ``shared`` reads them as it reads a release.

Folders, all under the download root (``$BRAINSCORE_DATA_HOME/sources/cneuromod``), in
``extraction/``: ``env`` (the environment), ``pip-cache`` and ``tmp`` (pip and temporary files),
``cneuromod_extract_tseries`` (the code; its ``atlases`` links to ``anat.atlases``),
``sourcedata/<data set>.fmriprep``, ``output/<data set>/<form>`` (the files as the code writes
them) and ``timeseries/<data set>/<form>/sub-XX`` (the same files under the released names).
"""
import contextlib
import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from brainscore.data.cneuromod.data_packaging import shared

CODE = 'cneuromod_extract_tseries'
CODE_COMMIT = 'ae310a53e035b15bf9936edf6683ae778620d5c0'    # the commit that made the released time series
# form -> (the code's parcellation configuration, {part of the code's file name: part of the released name}),
# as the release's code/run_extract_cneuromod26.sh and run_extract_voxt1w.sh rename them
FORMS = {
    'cneuromod2026': ('schaefer1000Tian50Nette128', {
        'atlas-Schaefer2018TianS3NettekovenAsym_desc-1000Parcels7Networks50Subcort128Cereb_':
            'atlas-cneuromod26_desc-1134Parcels_',
        'label-GM_desc-indivBoldGMfromFS_mask': 'label-GMfromFS_desc-indivFunc_mask'}),
    'voxel_native': ('gm_epi', {
        'atlas-Freesurfer_label-BoldGM_res-func_timeseries': 'desc-voxelwise_timeseries',
        'label-GM_desc-indivBoldGMfromFS_mask': 'label-GMfromFS_desc-indivFunc_mask'}),
}


def build(module, form: str, root: Path, participant: str, session: str | None) -> list:
    """Make the time series of ``participant`` (all sessions, or ``session``) in ``form``; the released
    files of the participant. ``<data set>.timeseries`` is cloned in ``root`` (git only) for its commits."""
    if form not in FORMS:
        raise ValueError(f'form {form!r} is not made from the fMRIPrep outputs; the forms are {", ".join(FORMS)}')
    parcellation, renames = FORMS[form]
    base = root / 'extraction'
    pins = sources(module, root)
    print('CNeuroMod time series from the fMRIPrep outputs; commits: '
          + ', '.join(f'{name} {commit[:8]}' for name, commit in pins.items()))
    code = base / CODE
    config = read_config(code / 'timeseries' / 'config' / 'parcellation' / f'{parcellation}.yaml')
    short = participant.split('-', 1)[1]
    atlas = sorted({base / shared.ATLASES / Path(config[key].replace('*', short)).relative_to('atlases')
                    for key in ('gm_path', 'parcellation')})
    shared.get(atlas)
    python, environ = environment(base, code)
    fmriprep = base / 'sourcedata' / f'{module.NAME}.fmriprep'
    work = base / 'output' / module.NAME / form
    func = work / participant / 'func'
    for name in shared.sessions(fmriprep, participant, session):
        bolds = sorted(Path(path) for path in glob.glob(
            str(fmriprep / participant / name / 'func' / f'*_space-{config["space"]}_desc-preproc_bold.nii.gz')))
        runs = {run_name(bold): bold for bold in bolds}
        if not runs or set(runs) <= done_runs(func):
            print(f'{participant} {name}: ' + ('no BOLD run' if not runs else 'every run is extracted already'))
            continue
        inputs = [path for bold in bolds for path in (
            bold, bold.with_name(bold.name.replace('_desc-preproc_bold', '_desc-brain_mask')),
            confounds_of(bold))]
        shared.plan(inputs, base)
        shared.get(inputs)
        # Relative atlas paths in the configuration: run from the code folder. Quoted: a path may hold a space.
        subprocess.run([str(python), '-m', 'timeseries.run', f'dataset={module.NAME}', f'parcellation={parcellation}',
                        f'subject_list=["{short}"]', f"output_dir='{work}'", f"data_dir='{base / 'sourcedata'}'"],
                       cwd=code, env=environ, check=True)
        import nibabel
        confounds = {run: confounds_of(bold) for run, bold in runs.items()}
        for run, bold in runs.items():          # fMRIPrep writes one confounds row per BOLD volume
            if len(shared.read_tsv(confounds[run])) != nibabel.load(bold).shape[3]:
                raise RuntimeError(f'run {run}: its confounds table and its BOLD file differ in volumes')
        check_runs(func, confounds, columns(func, config))
        # The public store is read over https, where git-annex cannot confirm a copy; the files came from it.
        subprocess.run(['datalad', 'drop', '--reckless', 'availability',
                        *(str(bold.relative_to(fmriprep)) for bold in bolds)], cwd=fmriprep, check=True)
    # a run extracted before its volumes were stored (an older build): stored now, from its confounds
    unstamped = set(stamped(func, False))
    check_runs(func, {run_name(path): path for path in map(Path, glob.glob(str(
        fmriprep / participant / 'ses-*' / 'func' / '*_desc-confounds_timeseries.tsv')))
        if run_name(path) in unstamped}, columns(func, config))
    return released(func, base / 'timeseries' / module.NAME / form / participant, renames)


def confounds_of(bold: Path) -> Path:
    return bold.with_name(bold.name.split('_space-')[0] + '_desc-confounds_timeseries.tsv')


def kept_volumes(confounds: Path) -> np.ndarray:
    """The volumes the code keeps: each volume (one confounds row) but those fMRIPrep marks non-steady
    (``non_steady_state_outlier*``), which the denoise strategy (``simple``) leaves out."""
    flags = shared.read_tsv(confounds).filter(like='non_steady_state_outlier').to_numpy()
    return np.flatnonzero(~(flags > 0).any(axis=1)) if flags.size else np.arange(len(shared.read_tsv(confounds)))


def stamped(func: Path, value: bool) -> list:
    """The runs of the HDF5 file that have (``value`` True) or lack the stored ``volumes``."""
    import h5py
    h5 = _h5(func)
    if h5 is None:
        return []
    with h5py.File(h5, 'r') as handle:
        return [name.removesuffix('_timeseries') for group in handle for name in handle[group]
                if ('volumes' in handle[group][name].attrs) == value]


def sources(module, root: Path) -> dict:
    """Clone the code, ``anat.atlases`` and ``<data set>.fmriprep`` at the commits the release names
    (the submodules of ``<data set>.timeseries``, and ``atlases`` of the code). ``{repository: commit}``."""
    base = root / 'extraction'
    release = root / f'{module.NAME}.timeseries'
    if not (release / '.git').exists():     # git only: its files are not on the public store
        subprocess.run(['git', 'clone', '-q', shared.GITHUB.format(release.name), str(release)], check=True)
    fmriprep = f'{module.NAME}.fmriprep'
    named = submodule_commit(release, f'code/{CODE}')
    if named != CODE_COMMIT and named != getattr(module, 'UNPUBLISHED_RELEASE_CODE', None):
        raise RuntimeError(f'{release.name} names {CODE} at {named}, not {CODE_COMMIT}, the commit whose '
                           'environment this builder makes. Add that commit here only after a check of its pins.')
    if named != CODE_COMMIT:
        print(f'{release.name} names {CODE} at {named}, a commit that is not public: made with {CODE_COMMIT}.')
    pins = {CODE: CODE_COMMIT, fmriprep: submodule_commit(release, f'sourcedata/{fmriprep}')}
    code = base / CODE
    if not (code / '.git').exists():
        subprocess.run(['git', 'clone', '-q', shared.GITHUB.format(CODE), str(code)], check=True)
    checkout(code, CODE_COMMIT)
    pins[shared.ATLASES] = submodule_commit(code, 'atlases')
    checkout(shared.clone(base, shared.ATLASES), pins[shared.ATLASES])
    link = code / 'atlases'
    if not link.is_symlink():
        link.rmdir()                # the empty folder of the submodule
        link.symlink_to(Path('..') / shared.ATLASES)
    checkout(shared.clone(base / 'sourcedata', fmriprep), pins[fmriprep])
    return pins


def submodule_commit(repository: Path, path: str) -> str:
    words = subprocess.run(['git', '-C', str(repository), 'ls-tree', 'HEAD', path],
                           capture_output=True, text=True, check=True).stdout.split()
    if len(words) < 3 or words[1] != 'commit':
        raise RuntimeError(f'{repository} names no commit of {path}')
    return words[2]


def checkout(folder: Path, commit: str) -> None:
    if subprocess.run(['git', '-C', str(folder), 'cat-file', '-e', f'{commit}^{{commit}}']).returncode != 0:
        subprocess.run(['git', '-C', str(folder), 'fetch', '-q', 'origin', commit], check=True)
    subprocess.run(['git', '-C', str(folder), 'checkout', '-q', commit], check=True)


def read_config(path: Path) -> dict:
    """A parcellation configuration of the code: one ``key: value`` per line."""
    return {key.strip(): value.strip() for key, _, value in
            (line.partition(':') for line in path.read_text().splitlines()) if value.strip() and not key.startswith('#')}


def environment(base: Path, code: Path) -> tuple:
    """``(python, environment variables)`` of a separate environment with the pinned versions of the
    code's ``requirements.txt`` only (its unpinned tools, such as awscli, are left out). The pip cache
    and the temporary files stay under ``base``; nothing of the caller's environment is visible."""
    environ = {key: value for key, value in os.environ.items() if key not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}
    environ.update(PIP_CACHE_DIR=str(base / 'pip-cache'), TMPDIR=str(base / 'tmp'))
    (base / 'tmp').mkdir(parents=True, exist_ok=True)
    pinned = sorted(line.split('#')[0].strip() for line in (code / 'requirements.txt').read_text().splitlines()
                    if '==' in line.split('#')[0])
    env = base / 'env'
    python, made = env / 'bin' / 'python', env / 'pinned.txt'
    if not made.exists() or made.read_text() != '\n'.join(pinned):
        subprocess.run([sys.executable, '-m', 'venv', '--clear', str(env)], env=environ, check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--quiet', *pinned], env=environ, check=True)
        made.write_text('\n'.join(pinned))
    return python, environ


def run_name(bold: Path) -> str:
    """The HDF5 name of a BOLD run without ``_timeseries``, as the code makes it: ``ses-02_task-things_run-1``."""
    found = shared.entities(bold.name.split('.')[0])
    return f'ses-{found["ses"]}_task-{found["task"]}' + (f'_run-{found["run"]}' if 'run' in found else '')


def _h5(func: Path) -> Path | None:
    return next(iter(sorted(func.glob('*_timeseries.h5'))), None)


def done_runs(func: Path) -> set:
    import h5py
    h5 = _h5(func)
    if h5 is None:
        return set()
    with h5py.File(h5, 'r') as handle:
        return {name.removesuffix('_timeseries') for group in handle for name in handle[group]}


def columns(func: Path, config: dict) -> int:
    """The columns of each run: the parcels of the participant's atlas (``dseg``) or the voxels of its mask."""
    import nibabel
    kind = config['parcel_type']
    label = 'label-' if kind == 'mask' else 'desc-'
    parcel = next(func.glob(f'*_atlas-{config["parcel_name"]}_{label}{config["desc"]}_{kind}.nii.gz'))
    data = np.asarray(nibabel.load(parcel).dataobj)
    return int((data > 0).sum()) if kind == 'mask' else len(np.unique(data[data > 0]))


def check_runs(func: Path, confounds: dict, width: int) -> None:
    """Each run is in the HDF5 file with one row per kept volume (``kept_volumes`` of its confounds table) and
    ``width`` columns; then the kept volumes are stored on the run (attribute ``volumes``), where the per-TR
    reader times each row. The code catches every error of a run and goes on, so a failed run is found here."""
    import h5py
    if not confounds:
        return
    h5 = _h5(func)
    with h5py.File(h5, 'a') if h5 is not None else contextlib.nullcontext({}) as handle:
        datasets = {name.removesuffix('_timeseries'): handle[group][name] for group in handle for name in handle[group]}
        for run, table in sorted(confounds.items()):
            kept = kept_volumes(table)
            shape = datasets[run].shape if run in datasets else None
            if shape != (len(kept), width):
                found = 'missing' if shape is None else f'{shape[0]} rows x {shape[1]} columns'
                raise RuntimeError(
                    f'run {run}: {found}, expected {len(kept)} rows (its kept volumes) x {width} columns, in {func}. '
                    'The extraction printed the cause above ("could not process file"). Delete the HDF5 file in that '
                    'folder to extract every run again, then run the same command.')
            datasets[run].attrs['volumes'] = kept


def released(func: Path, folder: Path, renames: dict) -> list:
    """The code's files under the released names in ``folder``: hard links, so a later session appended to
    the HDF5 file is in both."""
    folder.mkdir(parents=True, exist_ok=True)
    for path in sorted(func.iterdir()):
        for theirs, ours in renames.items():
            if theirs in path.name:
                target = folder / path.name.replace(theirs, ours)
                target.unlink(missing_ok=True)
                os.link(path, target)
    files = sorted(folder.iterdir())
    if not any(path.name.endswith('_timeseries.h5') for path in files):
        raise FileNotFoundError(f'no time series was made in {func}')
    return files
