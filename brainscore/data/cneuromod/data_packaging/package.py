"""Package one CNeuroMod data set; ``package_data('cneuromod-<dataset>...')`` runs this.

    python -m brainscore.data.cneuromod.data_packaging.package --dataset things --form glmsingle \
        --source <folder> --out <new folder> [--part sub-01 | sub-01/ses-01]

``--form``: ``glmsingle`` (THINGS trial responses, public) or a time series form
(``cneuromod2026``, ``schaefer1000``, ``voxel_native``, ``voxel_mni``; not public yet).
The source is a folder of CNeuroMod DataLad clones (``things/``, ``anat.atlases/``, ...). When it
does not exist, the clones are made and the needed files downloaded in
``$BRAINSCORE_DATA_HOME/sources/cneuromod``, kept there so that a stopped run resumes.
"""
import argparse
import importlib
from pathlib import Path

import pandas as pd

from brainscore.data.cneuromod.data_packaging import shared
from brainscore.data.local import data_home


def _mask_patterns(participant: str) -> list:
    short = participant.replace('-', '')
    return [f'tpl-{short}T1w/tpl-{short}T1w_res-func_atlas-*_mask.nii.gz']


def _atlas_patterns(participant: str) -> list:
    return [f'tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_{participant}_res-func_atlas-'
            'Schaefer2018TianS3NettekovenAsym_*_dseg.nii.gz', shared.LABELS]


def _files(root: Path, repository: str, patterns: list) -> list:
    found = [path for pattern in patterns for path in sorted((root / repository).glob(pattern))]
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
        for repository in (module.REPOSITORY, shared.ATLASES, form_repository):
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
            'atlas': _files(root, shared.ATLASES, _atlas_patterns(participant)),
            'form': _files(root, form_repository, form)}
    paths = [path for kinds in needed.values() for group in kinds.values() for path in group]
    if fetching:
        shared.plan(paths, root)
        shared.get(paths)
    shared.existing(paths)

    out = Path(args.out)
    number = None if session is None else shared.number(session)
    for participant, kinds in needed.items():
        masks = shared.copy_files(kinds['masks'], out / participant / 'masks')
        dseg, labels = kinds['atlas']
        shared.copy_files([dseg, labels], out / participant / 'atlas')
        shared.write_csv(shared.atlas_parcels(dseg, labels), out / participant / 'atlas' / 'parcels.csv')
        if args.form == 'glmsingle':
            module.glmsingle_trials(root, participant, number, masks, out)
            continue
        h5 = next(path for path in kinds['form'] if path.name.endswith('_timeseries.h5'))
        neuroid, columns = _timeseries_neuroids(args.form, kinds['form'], masks, labels)
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
