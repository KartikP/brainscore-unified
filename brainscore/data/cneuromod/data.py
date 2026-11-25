"""Read the packaged CNeuroMod THINGS files: the trial responses of one participant and region, and
the images. ``brainscore.load_dataset('cneuromod-things', ...)`` and
``brainscore.load_stimulus_set('cneuromod-things')`` call these."""
from pathlib import Path

import pandas as pd

NAME = 'cneuromod-things'
# The sha1 of each THINGS image of the sessions, from images_THINGS.zip (https://osf.io/jum2f), read 2026-11-24.
IMAGE_SHA1 = Path(__file__).parent / 'things_images_sha1.csv'


def packaged(root=None) -> Path:
    """The folder ``package_data('cneuromod-things')`` wrote: ``root``, or its default place."""
    from brainscore.data.local import LocalDataMissing, data_home
    folder = Path(root) if root is not None else data_home() / 'packaged' / NAME
    if not folder.is_dir():
        raise LocalDataMissing(f'{folder} does not exist. Package the data first: '
                               f"`brainscore.data.package_data('{NAME}', part='sub-01')`, one call per participant "
                               f'(see brainscore/data/cneuromod/README.md).')
    return folder


def load_trials(participant: str, region: str, root=None):
    """The trial responses of ``participant`` in ``region`` (the packaged name: ``hV4``, ``FFA``,
    ``cortex``), every session in time order: a ``presentation`` x ``neuroid`` array with the
    presentation coordinates of ``trials.nc`` (``stimulus_id``, ``response``, ...) and ``viewing``, the
    place of each trial among the answered trials of its image (``things.answered_viewing``; 0: no response)."""
    import xarray as xr
    folder = packaged(root) / participant
    paths = sorted(folder.glob('ses-*/trials.nc'))
    if not paths:
        raise FileNotFoundError(f'no ses-*/trials.nc under {folder}: package {participant} '
                                f"(`package_data('{NAME}', part='{participant}')`).")
    sessions = []
    for path in paths:
        session = xr.open_dataarray(path)
        keep = session['region'].values == region
        if not keep.any():
            raise ValueError(f'{participant} has no voxel of region {region!r} in {path}; its regions are '
                             f'{sorted(set(session["region"].values))}.')
        sessions.append(session.isel(neuroid=keep).load())
        session.close()
    from brainscore.data.cneuromod.data_packaging.things import answered_viewing
    trials = xr.concat(sessions, dim='presentation')
    return trials.assign_coords(viewing=('presentation', answered_viewing(trials['stimulus_id'].values,
                                                                          trials['response'].values)))


def load_stimulus_set(root=None):
    """The images the packaged participants saw, from the THINGS images the user put at the
    ``things-images`` place: one row per image, ``stimulus_id``, ``category``, ``image_path``.
    Each image must have the bytes of the THINGS database (the sha1 of ``things_images_sha1.csv``);
    a missing or different file is refused by name."""
    from brainscore.data import local
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    folder = packaged(root)
    ids = sorted(set().union(*(pd.read_csv(path)['stimulus_id'].dropna().astype(str)
                               for path in folder.glob('sub-*/ses-*/stimuli.csv'))))
    images = local.path('things-images')
    table = pd.DataFrame({'stimulus_id': ids, 'category': [i.rsplit('_', 1)[0] for i in ids]})
    table['image_path'] = [str(images / category / f'{i}.jpg') for i, category in zip(ids, table['category'])]
    missing = [path for path in table['image_path'] if not Path(path).is_file()]
    if missing:
        raise local.LocalDataMissing(f'{len(missing)} of {len(table)} THINGS images are not in {images} '
                                     f'(for example {missing[0]}). {local.REGISTRY["things-images"].instructions()}')
    import hashlib
    recorded = pd.read_csv(IMAGE_SHA1).set_index('stimulus_id')['sha1']
    unknown = sorted(set(ids) - set(recorded.index))
    if unknown:
        raise ValueError(f'{len(unknown)} packaged images have no recorded sha1 (for example {unknown[0]}): '
                         'the packaged files are not the CNeuroMod THINGS sessions.')
    wrong = [path for i, path in zip(ids, table['image_path'])
             if hashlib.sha1(Path(path).read_bytes()).hexdigest() != recorded[i]]
    if wrong:
        raise ValueError(f'{len(wrong)} of {len(table)} images in {images} differ from the THINGS database bytes '
                         f'(for example {wrong[0]}): use the files of images_THINGS.zip (https://osf.io/jum2f) '
                         'unchanged, not a resized or converted copy.')
    stimuli = StimulusSet(table)
    stimuli.identifier = NAME
    stimuli.stimulus_paths = dict(zip(table['stimulus_id'], table['image_path']))
    return stimuli

