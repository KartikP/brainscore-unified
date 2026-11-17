"""Build every file the registered Algonauts 2025 benchmarks read, from the competition files.

    python -m brainscore.data.algonauts2025.data_packaging.package --source <competition root> --out <new folder>

``python -m brainscore.data package algonauts2025`` runs it. It writes the assemblies
and film tables (``prepare_assembly``), then cuts the training films into the media
that the registered benchmarks show (``prepare_stimuli``): frames for every
participant, clips and sound for participant 1. It fails when an input of a
participant or split is missing, or when a film that an assembly names is absent or
was not cut, so a finished build is complete.
"""
import argparse
from pathlib import Path
from typing import List, Optional

from brainscore.data.algonauts2025 import (
    PARTICIPANTS, SPLITS, assembly_identifier, per_scan_stimuli, scans_per_stimulus, stimulus_set_identifier)
from brainscore.data.algonauts2025.data_packaging import prepare_assembly, prepare_stimuli

# The media of each participant's registered benchmarks: Algonauts2025-friends-sub0X shows
# frames; -sub01-video shows clips; -sub01-video-audio shows clips and sound.
REGISTERED_MEDIA = {1: ('frames', 'clips', 'sound'), 2: ('frames',), 3: ('frames',), 5: ('frames',)}


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
    # Every film that an assembly names is in its film table, and its file is in the source.
    _require_films(args.out)
    # Cut the training films into one file per scan for the media each participant's benchmarks show.
    for participant, media in REGISTERED_MEDIA.items():
        prepare_stimuli.main(['--split', 'friends', '--participant', str(participant),
                              '--media', *media, '--prepared-root', str(args.out)])
    # Read the media back as the benchmarks do; a scan without its file raises FileNotFoundError.
    from brainscore import load_dataset, load_stimulus_set
    films = load_stimulus_set(stimulus_set_identifier('friends'), root=args.out)
    for participant, media in REGISTERED_MEDIA.items():
        scans = scans_per_stimulus(load_dataset(assembly_identifier('friends', participant), root=args.out))
        for kind in media:
            per_scan_stimuli(films, scans, [kind], 'friends', participant, args.out)


def _require_films(root: Path) -> None:
    """Raise ``FileNotFoundError`` naming each film an assembly names that the source lacks."""
    from brainscore import load_dataset, load_stimulus_set
    absent = set()
    for split in SPLITS:
        table = load_stimulus_set(stimulus_set_identifier(split), root=root)
        films = dict(zip(table['stimulus_id'], table['video_path']))
        for participant in PARTICIPANTS:
            named = scans_per_stimulus(load_dataset(assembly_identifier(split, participant), root=root))
            absent |= {film for film in named if film not in films or not Path(films[film]).exists()}
    if absent:
        raise FileNotFoundError(
            f'the assemblies name {len(absent)} films that the source lacks: {", ".join(sorted(absent))} '
            '(stimuli/movies/...)')


if __name__ == '__main__':
    main()
