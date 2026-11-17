"""Build every file the registered Algonauts 2025 benchmarks read, from the competition files.

    python -m brainscore.data.algonauts2025.data_packaging.package --source <competition root> --out <new folder>

``python -m brainscore.data package algonauts2025`` runs it. It fails when an input
of a participant or split is missing, so a finished build is complete.
"""
import argparse
from pathlib import Path
from typing import List, Optional

from brainscore.data.algonauts2025 import PARTICIPANTS, SPLITS
from brainscore.data.algonauts2025.data_packaging import prepare_assembly


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source', type=Path, required=True, help='the competition root (fmri/, stimuli/)')
    parser.add_argument('--out', type=Path, required=True, help='a new folder for the prepared files')
    args = parser.parse_args(argv)
    # Convert the fMRI of every participant and split, and the film tables.
    prepare_assembly.main(['--algonauts-root', str(args.source), '--output-root', str(args.out)])
    # A complete build has the assembly of every participant and split.
    expected = [args.out / f'algonauts2025_{split}_sub{participant:02d}.nc'
                for split in SPLITS for participant in PARTICIPANTS]
    missing = [path.name for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} of {len(expected)} assemblies were not built ({", ".join(missing)}): '
            'the source lacks their fMRI files or their target_sample_number files '
            '(the [skip] lines above name each one)')


if __name__ == '__main__':
    main()
