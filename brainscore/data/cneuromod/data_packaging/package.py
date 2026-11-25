"""Package one CNeuroMod data set; ``package_data('cneuromod-<dataset>...')`` runs this.

    python -m brainscore.data.cneuromod.data_packaging.package --dataset things --form glmsingle \
        --source <folder> --out <new folder> [--part sub-01 | sub-01/ses-01]

``--form``: ``glmsingle`` (THINGS trial responses, public) or a time series form (``cneuromod2026``,
``voxel_native``). The source is a folder of CNeuroMod DataLad clones (``things/``, ``anat.atlases/``,
...), read where it is. When it does not exist, the clones are made and the needed files downloaded in
``$BRAINSCORE_DATA_HOME/sources/cneuromod``, kept there so that a stopped run resumes; a time series
is then made from the public fMRIPrep outputs (``extraction.py``), as the release made it.
"""
import argparse
import glob
import importlib
from pathlib import Path

import pandas as pd

from brainscore.data.cneuromod.data_packaging import extraction, shared
from brainscore.data.local import data_home


def _mask_patterns(participant: str) -> list:
    short = participant.replace('-', '')
    return [f'tpl-{short}T1w/tpl-{short}T1w_res-func_atlas-*_mask.nii.gz']


# The group cneuromod2026 atlas (1 mm, before the grey-matter mask of each participant): public.
# The masked atlas of each participant is not public; the time series files carry their own.
_ATLAS = ['tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_atlas-Schaefer2018TianS3NettekovenAsym_*_dseg.nii.gz',
          shared.LABELS]


def _files(root: Path, repository: str, patterns: list) -> list:
    # glob.glob, not Path.glob: an annexed file whose content is not here yet is a dangling link.
    found = [Path(path) for pattern in patterns for path in sorted(glob.glob(str(root / repository / pattern)))]
    if not found:
        raise FileNotFoundError(f'nothing in {root / repository} matches {patterns}')
    return found


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--form', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--part', default='')
    args = parser.parse_args(argv)
    module = importlib.import_module(f'brainscore.data.cneuromod.data_packaging.{args.dataset}')
    participants, session = shared.scope(module, args.part)
    fetching = not Path(args.source).exists()
    root = data_home() / 'sources' / 'cneuromod' if fetching else Path(args.source)
    form_repository = module.GLMSINGLE if args.form == 'glmsingle' else f'{module.NAME}.timeseries'
    if fetching:
        shared.check_prerequisites()
        anatomy = (module.ANATOMY,) if args.form == 'glmsingle' else ()
        for repository in (module.REPOSITORY, shared.ATLASES, form_repository, *anatomy):
            shared.clone(root, repository)

    needed = {}       # participant -> {kind: paths}
    for participant in participants:
        names = shared.sessions(root / module.REPOSITORY, participant, session)
        form = (module.glmsingle_patterns(participant) if args.form == 'glmsingle'
                else [f'timeseries/{args.form}/{participant}/*'])
        needed[participant] = {
            'events': _files(root, module.REPOSITORY, sorted({p for name in names
                                                               for p in module.event_patterns(participant, name)})),
            'masks': _files(root, shared.ATLASES, _mask_patterns(participant)),
            # fetched: made from the public fMRIPrep outputs; the released files need an account
            'form': (extraction.build(module, args.form, root, participant, session)
                     if fetching and args.form != 'glmsingle' else _files(root, form_repository, form))}
        if args.form == 'glmsingle':
            needed[participant]['cortex'] = _files(root, module.ANATOMY, module.cortex_patterns(participant))[:1]
    atlas = _files(root, shared.ATLASES, _ATLAS)
    paths = atlas + [path for kinds in needed.values() for group in kinds.values() for path in group]
    if fetching:
        shared.plan(paths, root)
        shared.get(paths)
    shared.existing(paths)

    out = Path(args.out)
    number = None if session is None else shared.number(session)
    dseg, labels = atlas
    shared.copy_files(atlas, out / 'atlas')
    shared.write_csv(shared.atlas_parcels(dseg, labels), out / 'atlas' / 'parcels.csv')
    for participant, kinds in needed.items():
        masks = shared.copy_files(kinds['masks'], out / participant / 'masks')
        if args.form == 'glmsingle':
            module.glmsingle_trials(root, participant, number, masks, kinds['cortex'][0], out)
            continue
        h5 = next(path for path in kinds['form'] if path.name.endswith('_timeseries.h5'))
        neuroid, columns = _timeseries_neuroids(args.form, kinds['form'], masks, labels)
        if columns is None:         # the parcels present in the participant's own atlas
            shared.write_csv(neuroid, out / participant / 'atlas' / 'parcels.csv')
        shared.timeseries_assembly(module, root / module.REPOSITORY, participant, h5, neuroid, columns, number, out)


def _timeseries_neuroids(form: str, files: list, masks: list, labels: Path) -> tuple:
    """``(neuroid table, columns)`` of a time series form; ``columns`` None keeps every column."""
    import nibabel
    if form == 'cneuromod2026':
        dseg = next(path for path in files if path.name.endswith('_dseg.nii.gz'))
        return shared.atlas_parcels(dseg, labels), None
    if form == 'voxel_native':
        grid = nibabel.load(next(path for path in files if path.name.endswith('_mask.nii.gz')))
        return shared.region_neuroids(masks, grid)
    raise ValueError(f'form {form!r}: the per-TR packaging names the columns of cneuromod2026 and voxel_native')


if __name__ == '__main__':
    main()
