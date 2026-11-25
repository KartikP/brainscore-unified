"""The shared part of the CNeuroMod packaging, written once for every CNeuroMod data set.

A data set module (``things.py``, ``movie10.py``) gives only what differs: its participants,
its repetition time, the repository of its event files, how one run's event table is read and
which stimulus identifier each event has. Everything else is here: the download with DataLad,
the reading of the time series, masks and atlas, the TR-to-event join and the written files.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

GITHUB = 'https://github.com/courtois-neuromod/{}.git'
PUBLIC_STORE = 'conp-ria-storage-http'     # the CONP store over https; the others need an account or ssh
PUBLIC_STORE_URL = 'https://sftp.conp.ca/users/cneuromod/'
_CONFIRM = re.compile(r'verifyremote \S+ --safe (\S+)')
RESTRICTED = ('sub-04',)    # needs a data transfer agreement with CNeuroMod: never packaged
ATLASES = 'anat.atlases'
LABELS = ('tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_atlas-Schaefer2018TianS3NettekovenAsym_'
          'desc-1000Parcels7Networks50Subcort128Cereb.tsv')
# Measured on a cloud machine (us-east-2) on 2026-11-24 for the THINGS files; see the README.
MEASURED_MB_PER_SECOND = 24
_ANNEX_SIZE = re.compile(r'-s(\d+)--')
_PREFIXES = ('face', 'body', 'scene')     # fLoc mask labels: faceFFA -> FFA


def number(label) -> int:
    """The number in a BIDS label: ``ses-024`` -> 24, ``run-1`` -> 1, ``'3'`` -> 3."""
    found = re.search(r'\d+', str(label))
    if found is None:
        raise ValueError(f'no number in {label!r}')
    return int(found.group())


def entities(name: str) -> dict:
    """``ses-02_task-things_run-1_timeseries`` -> ``{'ses': '02', 'task': 'things', 'run': '1'}``."""
    return dict(part.split('-', 1) for part in Path(name).name.split('_') if '-' in part)


# --- participants and sessions ---------------------------------------------------------------

def scope(module, part: str) -> tuple:
    """``(participants, session or None)`` for ``part``: '' (all), ``sub-01`` or ``sub-01/ses-01``.

    A restricted participant (sub-04) is refused, also when a module lists it.
    """
    wanted, _, session = (part or '').strip('/').partition('/')
    participants = [wanted] if wanted else [p for p in module.PARTICIPANTS if p not in RESTRICTED]
    for participant in participants:
        if participant in RESTRICTED:
            raise ValueError(f'{participant} is restricted: CNeuroMod shares it only under a data transfer '
                             'agreement, and Brain-Score never packages it. Choose another participant: '
                             f'{", ".join(p for p in module.PARTICIPANTS if p not in RESTRICTED)}.')
        if participant not in module.PARTICIPANTS:
            raise ValueError(f'{participant} is not a participant of {module.NAME}; '
                             f'the participants are {", ".join(module.PARTICIPANTS)}.')
    if session and not re.fullmatch(r'ses-\d+', session):
        raise ValueError(f'a part is "sub-XX" or "sub-XX/ses-NN", not {part!r}')
    return participants, session or None


def sessions(clone: Path, participant: str, session: str | None) -> list:
    """The session folders of a participant in a BIDS clone, or the one asked for."""
    found = sorted(path.name for path in (clone / participant).glob('ses-*') if path.is_dir())
    if session is None:
        return found
    matches = [name for name in found if number(name) == number(session)]
    if not matches:
        raise ValueError(f'{participant} has no {session}; it has {", ".join(found) or "no session"}.')
    return matches


# --- download ------------------------------------------------------------------------------

def clone(cache: Path, repository: str) -> Path:
    """A DataLad clone of one CNeuroMod repository in ``cache`` (no recursion; kept for a resume),
    with the public store (``conp-ria-storage-http``) enabled: a clone does not always enable it."""
    folder = cache / repository
    if not (folder / '.git').exists():
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run(['datalad', 'clone', GITHUB.format(repository), str(folder)], check=True)
    _enable_public_store(folder)
    return folder


def _enable_public_store(folder: Path) -> None:
    """Enable the public store. git-annex 10.20261113 (not 10.20261026) first asks to confirm a special
    remote: confirm it when, and only when, it asks and the remote is the CNeuroMod store of CONP."""
    enable = ['datalad', 'siblings', '-d', str(folder), 'enable', '-s', PUBLIC_STORE]
    result = subprocess.run(enable, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout, end='')
        return
    message = result.stdout + result.stderr
    asked = _CONFIRM.search(message)
    if asked is None:
        raise RuntimeError(f'enabling {PUBLIC_STORE} in {folder} failed:\n{message}')
    if f'url={PUBLIC_STORE_URL}' not in message:
        raise RuntimeError(f'git-annex asks to confirm {PUBLIC_STORE} in {folder}, but it is not at '
                           f'{PUBLIC_STORE_URL}: not confirmed. Check the clone; its message:\n{message}')
    subprocess.run(['git-annex', 'verifyremote', PUBLIC_STORE, '--safe', asked.group(1)], cwd=folder, check=True)
    subprocess.run(enable, check=True)


def _pointer(path: Path) -> str:
    """The annex key an unlocked annexed file names while its content is not here (a small file that
    holds ``/annex/objects/<key>``); '' for any other file."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024:
        return ''
    text = path.read_bytes()
    return text.decode(errors='replace').strip() if text.startswith(b'/annex/objects/') else ''


def present(path: Path) -> bool:
    """The file's content is here: not a dangling annex link and not an unlocked annex pointer file."""
    return path.exists() and not _pointer(path)


def annex_size(path: Path) -> int:
    """The size of an annexed file from its key (the file need not be present); a file in git: its size."""
    key = os.readlink(path) if path.is_symlink() else _pointer(path)
    found = _ANNEX_SIZE.search(key)
    if found:
        return int(found.group(1))
    return path.stat().st_size if path.exists() else 0


_INSTALL = ('Install DataLad and git-annex 10.20230126 or later, for example `pipx install datalad` and '
            '`pipx inject --include-apps datalad git-annex`, then run the same command again.')


def check_prerequisites() -> None:
    missing = [program for program in ('datalad', 'git-annex') if shutil.which(program) is None]
    if missing:
        raise RuntimeError(f'{" and ".join(missing)} not found. {_INSTALL}')
    version = subprocess.run(['git-annex', 'version', '--raw'], capture_output=True, text=True).stdout.strip()
    found = re.match(r'(\d+)\.(\d+)', version)
    if found is None or (int(found.group(1)), int(found.group(2))) < (10, 20230126):
        raise RuntimeError(f'git-annex {version or "of unknown version"} found; this command needs '
                           f'10.20230126 or later. {_INSTALL}')


def plan(paths: list, cache: Path) -> int:
    """Print what the download needs before it starts; stop when the disk is too small. The bytes to get."""
    missing = [path for path in paths if not present(path)]
    total = sum(annex_size(path) for path in paths)
    to_get = sum(annex_size(path) for path in missing)
    free = shutil.disk_usage(cache).free
    print('CNeuroMod download plan')
    print('  Needs: DataLad and git-annex on the PATH; the clones use https (no account, no key).')
    print(f'  Files: {len(paths)} ({total / 1e9:.2f} GB); already here: {len(paths) - len(missing)}; '
          f'to download: {len(missing)} ({to_get / 1e9:.2f} GB) into {cache}')
    if MEASURED_MB_PER_SECOND:
        print(f'  Time: about {to_get / 1e6 / MEASURED_MB_PER_SECOND / 60:.0f} minutes at the measured '
              f'{MEASURED_MB_PER_SECOND} MB/s (your rate may differ)')
    print(f'  Disk: {free / 1e9:.1f} GB free; a stop keeps what is downloaded, and the same command resumes.')
    if to_get * 1.1 > free:
        raise RuntimeError(f'not enough disk: {to_get / 1e9:.2f} GB to download, {free / 1e9:.1f} GB free in '
                           f'{cache}. Free some space or set BRAINSCORE_DATA_HOME to a larger disk.')
    return to_get


def get(paths: list) -> None:
    """``datalad get`` of the listed paths only, grouped by clone."""
    by_clone = {}
    for path in paths:
        if not present(path):
            root = next(parent for parent in path.parents if (parent / '.datalad').exists())
            by_clone.setdefault(root, []).append(str(path.relative_to(root)))
    for root, relative in by_clone.items():
        for start in range(0, len(relative), 200):
            subprocess.run(['datalad', 'get', *relative[start:start + 200]], cwd=root, check=True)


def existing(paths: list) -> list:
    """The paths, each present; a missing one raises FileNotFoundError that names it."""
    for path in paths:
        if not present(path):
            raise FileNotFoundError(f'{path} is missing (its content was not downloaded)')
    return paths


# --- reading -------------------------------------------------------------------------------

def read_timeseries(path: Path):
    """Each run of a CNeuroMod time series HDF5, any form: ``(entities, array TRs x columns)``.

    Group = session, dataset = run (``ses-02_task-things_run-1_timeseries``); ``entities`` has
    ``session`` (int), ``task``, ``run`` (int, 1 when the name has no run) and ``volumes``: the BOLD
    volume of each row, stored by the builder (the extraction leaves out non-steady volumes); a file
    without it (a released file): every volume, 0, 1, 2, ...
    """
    import h5py
    with h5py.File(path, 'r') as handle:
        for group in sorted(handle, key=number):
            for name in sorted(handle[group]):
                found = entities(name)
                array = np.asarray(handle[group][name], dtype=np.float32)
                run = {'session': number(group), 'task': found.get('task', ''), 'run': int(found.get('run', 1)),
                       'volumes': np.asarray(handle[group][name].attrs.get('volumes', np.arange(len(array))))}
                yield run, array


def region_columns(mask, grid) -> np.ndarray:
    """The columns of a voxel grid inside a region mask: indices into the True voxels of ``grid``
    in C order (the order of ``nilearn.masking.apply_mask``). The mask is resampled to the grid
    by nearest neighbour when the two grids differ."""
    if mask.shape[:3] != grid.shape[:3] or not np.allclose(mask.affine, grid.affine):
        from nilearn.image import resample_to_img
        mask = resample_to_img(mask, grid, interpolation='nearest', force_resample=True, copy_header=True)
    inside = np.asarray(mask.dataobj).reshape(grid.shape[:3]) > 0
    on_grid = np.asarray(grid.dataobj) > 0
    return np.flatnonzero(inside[on_grid])


def region_neuroids(masks: list, grid, atlases: tuple = ()) -> tuple:
    """One neuroid per (region mask, voxel) of a voxel grid: ``(neuroid table, grid column of each)``.
    ``atlases`` keeps the masks of these atlases only, in this order of preference: a region takes
    the mask of the first atlas that has it, so each region has one mask set (empty: every mask)."""
    import nibabel
    tables, columns, taken = [], [], set()
    rank = {atlas: place for place, atlas in enumerate(atlases)}
    named = sorted((mask_region(path), path) for path in masks)
    for (atlas, region), path in sorted(named, key=lambda item: rank.get(item[0][0], -1)):
        if atlases and (atlas not in rank or region in taken):
            continue
        taken.add(region)
        inside = region_columns(nibabel.load(path), grid)
        tables.append(voxel_rows(atlas, region, inside, grid))
        columns.append(inside)
    if not tables:
        raise FileNotFoundError(f'no mask of {", ".join(atlases) or "any atlas"} among {len(masks)} masks')
    return pd.concat(tables, ignore_index=True), np.concatenate(columns)


def voxel_rows(atlas: str, region: str, inside: np.ndarray, grid) -> pd.DataFrame:
    """The neuroid rows of the grid columns ``inside`` of one region: region, atlas, voxel (grid column)
    and i, j, k (the voxel's place in the grid, C order: the column order of the grid)."""
    ijk = np.argwhere(np.asarray(grid.dataobj) > 0)[inside]
    return pd.DataFrame({'neuroid_id': [f'{atlas}.{region}.{c}' for c in inside], 'region': region,
                         'atlas': atlas, 'voxel': inside, 'i': ijk[:, 0], 'j': ijk[:, 1], 'k': ijk[:, 2]})


def mask_region(path: Path) -> tuple:
    """``(atlas, region)`` of a mask file name: ``..._atlas-fLocVisionTask_label-faceFFA_mask`` -> FFA."""
    found = entities(path.name.split('.')[0])
    label = found['label']
    for prefix in _PREFIXES:
        if label.startswith(prefix) and len(label) > len(prefix):
            label = label[len(prefix):]
            break
    return found['atlas'], label


def atlas_parcels(dseg: Path, labels: Path) -> pd.DataFrame:
    """The parcels present in a participant's atlas, in column order (ascending index, no 0)."""
    import nibabel
    present = np.unique(np.asarray(nibabel.load(dseg).dataobj).astype(int))
    names = read_tsv(labels).set_index('index')['name']
    present = present[present > 0]
    return pd.DataFrame({'parcel': present, 'parcel_name': [names.get(i, '') for i in present]})


def tr_events(volumes: np.ndarray, tr_seconds: float, events: pd.DataFrame) -> np.ndarray:
    """For each row (BOLD volume ``volumes[i]``, which starts at volume x TR), the row of ``events`` whose
    ``[onset, onset + duration)`` holds the volume start, else -1. A later event wins."""
    starts = np.asarray(volumes) * tr_seconds
    rows = np.full(len(starts), -1)
    for row, (onset, duration) in enumerate(zip(events['onset'], events['duration'])):
        rows[(starts >= onset - 1e-6) & (starts < onset + duration - 1e-6)] = row
    return rows


# --- writing -------------------------------------------------------------------------------

def run_table(module, clone: Path, participant: str, run: dict) -> pd.DataFrame:
    """One run's event table from the module, sorted by onset, with the run's coordinates."""
    events = module.run_events(clone, participant, run).sort_values('onset', kind='stable').reset_index(drop=True)
    for column in ('onset', 'duration', 'stimulus_id', 'repetition'):
        if column not in events:
            raise ValueError(f'{module.NAME}.run_events gave no {column!r} column')
    front = pd.DataFrame({'participant': participant, 'session': run['session'], 'task': run['task'],
                          'run': run['run'], 'event': range(len(events))})
    return pd.concat([front, events.drop(columns=[c for c in front if c in events])], axis=1)


def stimulus_table(module, events: pd.DataFrame) -> pd.DataFrame:
    """Identifiers only: one row per stimulus, the module's stimulus columns and the source to get it from."""
    columns = ['stimulus_id', *[c for c in module.STIMULUS_COLUMNS if c in events]]
    table = events[columns].drop_duplicates('stimulus_id').sort_values('stimulus_id').reset_index(drop=True)
    table = table[table['stimulus_id'] != '']
    table['source'] = module.STIMULUS_SOURCE
    return table


def read_tsv(path: Path) -> pd.DataFrame:
    """A tab-separated source table. ``round_trip``: each number is read exactly, so it is written
    back with the digits of the source; the default parser of some pandas versions changes the last
    digit, and the written bytes would then depend on the pandas version."""
    return pd.read_csv(path, sep='\t', float_precision='round_trip')


def write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def write_assembly(values: np.ndarray, presentation: pd.DataFrame, neuroid: pd.DataFrame, path: Path) -> None:
    """A netCDF data array, dims ``presentation`` x ``neuroid``, one coordinate per column.

    Written so that the same content gives the same bytes and a rebuild has the same sha1:
    netCDF3 (the scipy engine; netCDF4 differs in a few layout bytes) and a sorted list of
    coordinate names.
    """
    import xarray as xr
    coords = {**{f'{name}': ('presentation', _plain(presentation[name])) for name in presentation},
              **{f'{name}': ('neuroid', _plain(neuroid[name])) for name in neuroid}}
    path.parent.mkdir(parents=True, exist_ok=True)
    array = xr.DataArray(values.astype(np.float32), dims=('presentation', 'neuroid'), coords=coords, name='responses')
    array.encoding['coordinates'] = ' '.join(sorted(coords))    # else a set order, which varies by process
    array.to_netcdf(path, engine='scipy')


def _plain(column: pd.Series) -> np.ndarray:
    """A column netCDF can store: numbers as they are, everything else as text."""
    values = column.to_numpy()
    return values if values.dtype.kind in 'biuf' else column.fillna('').astype(str).to_numpy()


def timeseries_assembly(module, clone: Path, participant: str, h5: Path, neuroid: pd.DataFrame,
                        columns: np.ndarray | None, session: int | None, out: Path) -> list:
    """Write the per-TR assembly and the event table of each session of one participant's time
    series; ``columns`` selects voxel columns (None: all, the atlas form). The written paths."""
    written, by_session = [], {}
    for run, array in read_timeseries(h5):
        if session is not None and run['session'] != session:
            continue
        events = run_table(module, clone, participant, run)
        rows = tr_events(run['volumes'], module.TR_SECONDS, events)
        starts = run['volumes'] * module.TR_SECONDS
        onset_of = events['onset'].to_numpy(dtype=float)
        onsets = np.array([onset_of[row] if row >= 0 else np.nan for row in rows])
        presentation = pd.DataFrame({
            'participant': participant, 'session': run['session'], 'task': run['task'], 'run': run['run'],
            'tr': run['volumes'], 'time': starts, 'event': rows,
            'stimulus_id': [events['stimulus_id'].iloc[r] if r >= 0 else '' for r in rows],
            'time_in_event': starts - onsets})
        values = array if columns is None else array[:, columns]
        by_session.setdefault(run['session'], []).append((values, presentation, events))
    for number_, parts in sorted(by_session.items()):
        folder = out / participant / f'ses-{number_:02d}'
        values = np.concatenate([p[0] for p in parts])
        presentation = pd.concat([p[1] for p in parts], ignore_index=True)
        events = pd.concat([p[2] for p in parts], ignore_index=True)
        write_assembly(values, presentation, neuroid, folder / f'timeseries_{h5_form(h5)}.nc')
        write_csv(events, folder / 'events.csv')
        write_csv(stimulus_table(module, events), folder / 'stimuli.csv')
        written.append(folder)
    return written


def h5_form(h5: Path) -> str:
    return h5.parent.parent.name     # .../timeseries/<form>/sub-0X/<file>.h5


def copy_files(paths: list, folder: Path) -> list:
    """Copies of source files (masks, atlas) as their own data; the bytes are kept."""
    folder.mkdir(parents=True, exist_ok=True)
    return [Path(shutil.copyfile(path, folder / path.name)) for path in paths]
