"""Assets a user must obtain themselves, and how to convert them.

Some benchmarks cannot ship their inputs. Stimuli are usually the reason —
films, copyrighted text, face photographs — and occasionally the neural data or
an atlas carries its own agreement. Brain-Score can still *score* those
benchmarks; it just cannot hand over the bytes.

The pattern is the same every time: the user downloads something from the
provider, puts it somewhere, and a one-off script converts it into the form the
benchmark reads. What differs is the URL, the licence reason, and the command.
This module holds those three things per asset so that a missing file produces
instructions instead of a stack trace, and so a user can ask what they are
missing before starting a run rather than during one.

Declaring an asset::

    register(LocalAsset(
        name='lebel2023-pickle',
        env_var='BRAINSCORE_LEBEL_PICKLE',
        default_path='Downloads/assembly_lebel_uts03.pkl',
        kind='neural data',
        why_local='Redistributed per-subject by the provider, not by Brain-Score.',
        source='https://github.com/GT-LIT-Lab/litcoder_release',
        obtain='Download the UTS03 assembly pickle.',
    ))

Reading it::

    path = local.path('lebel2023-pickle')      # raises LocalDataMissing if absent

Seeing what is missing, without running anything::

    python -m brainscore.data.local
"""

import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class LocalDataMissing(FileNotFoundError):
    """A required local asset is absent. The message says how to get it."""


@dataclass(frozen=True)
class LocalAsset:
    """One file or directory the user supplies.

    :param name: stable identifier, referenced by :func:`path`
    :param env_var: environment variable overriding the location
    :param default_path: looked for under the home directory when unset
    :param kind: ``stimuli``, ``neural data`` or ``atlas`` — what is withheld
    :param why_local: why Brain-Score cannot ship it; shown to the user
    :param source: where to get it
    :param obtain: what to download, in a sentence
    :param prepare: shell command converting it into the benchmark's format,
        if conversion is needed. Downloads that are read directly leave this
        empty.
    :param used_by: benchmark identifiers that need it
    """

    name: str
    env_var: str
    default_path: str
    kind: str
    why_local: str
    source: str
    obtain: str
    prepare: str = ''
    used_by: List[str] = field(default_factory=list)

    def resolved(self) -> pathlib.Path:
        """Where it is expected, honouring the environment override."""
        override = os.environ.get(self.env_var)
        if override:
            return pathlib.Path(override).expanduser()
        return pathlib.Path.home() / self.default_path

    def present(self) -> bool:
        return self.resolved().exists()

    def instructions(self) -> str:
        lines = [
            f'{self.name} ({self.kind}) is required but was not found at',
            f'    {self.resolved()}',
            '',
            f'Why you have to fetch it: {self.why_local}',
            f'Get it from: {self.source}',
            f'What to download: {self.obtain}',
            '',
            f'Then either put it at the path above, or set {self.env_var} to '
            f'where it already is.',
        ]
        if self.prepare:
            lines += ['', 'Then convert it:', f'    {self.prepare}']
        if self.used_by:
            lines += ['', f'Needed by: {", ".join(self.used_by)}']
        return '\n'.join(lines)


REGISTRY: Dict[str, LocalAsset] = {}


def register(asset: LocalAsset) -> LocalAsset:
    """Declare an asset. Re-registering the same name replaces it."""
    REGISTRY[asset.name] = asset
    return asset


def path(name: str) -> pathlib.Path:
    """Resolve an asset, raising instructions if it is not there."""
    try:
        asset = REGISTRY[name]
    except KeyError:
        raise KeyError(
            f'no local asset named {name!r}; '
            f'known: {sorted(REGISTRY)}') from None
    resolved = asset.resolved()
    if not resolved.exists():
        raise LocalDataMissing(asset.instructions())
    return resolved


def status(name: Optional[str] = None) -> List[dict]:
    """Presence of every declared asset, for a pre-run check."""
    assets = [REGISTRY[name]] if name else list(REGISTRY.values())
    return [{'name': a.name, 'kind': a.kind, 'present': a.present(),
             'path': str(a.resolved()), 'used_by': a.used_by}
            for a in sorted(assets, key=lambda a: a.name)]


def format_status() -> str:
    rows = status()
    if not rows:
        return 'no local assets are declared'
    width = max(len(r['name']) for r in rows)
    lines = [f"{'':2s} {'asset'.ljust(width)}  {'kind':12s} path", '']
    for row in rows:
        lines.append(f"{'ok' if row['present'] else '--'} "
                     f"{row['name'].ljust(width)}  {row['kind']:12s} {row['path']}")
    missing = [r['name'] for r in rows if not r['present']]
    lines.append('')
    lines.append(f'{len(rows) - len(missing)}/{len(rows)} present'
                 + (f"; missing: {', '.join(missing)}" if missing else ''))
    if missing:
        lines.append(f'Run  python -m brainscore.data {missing[0]}  '
                     f'for how to obtain one.')
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# The manifest. Declared here rather than beside each reader so that asking
# what you need never requires importing the code that needs it — `python -m
# brainscore.data local` works on a bare checkout.
# --------------------------------------------------------------------------

LEBEL_PICKLE = register(LocalAsset(
    name='lebel2023-pickle',
    env_var='BRAINSCORE_LEBEL_PICKLE',
    default_path='Downloads/assembly_lebel_uts03.pkl',
    kind='neural data',
    why_local='Per-subject assembly redistributed by the provider under their '
              'terms, not by Brain-Score.',
    source='https://github.com/GT-LIT-Lab/litcoder_release',
    obtain='The UTS03 assembly pickle for LeBel et al. 2023.',
    used_by=['LeBel2023-UTS03-encoding',
             'LeBel2023-UTS03-encoding-languagemask'],
))

LANA_ATLAS = register(LocalAsset(
    name='lana-atlas',
    env_var='BRAINSCORE_LANA_ATLAS',
    default_path='Downloads/20425209/FS',
    kind='atlas',
    why_local='Fedorenko-lab resource distributed under its own terms.',
    source='https://osf.io/kzwbh/  (doi:10.17605/OSF.IO/KZWBH)',
    obtain='The "FS Atlas" archive - the FreeSurfer-surface arm, which needs no '
           'volume-to-surface projection. Unzip it; the directory holding '
           'LH_LanA_n804.nii.gz and RH_LanA_n804.nii.gz is what to point at.',
    used_by=['LeBel2023-UTS03-encoding-languagemask'],
))

ALGONAUTS_ROOT = register(LocalAsset(
    name='algonauts2025-root',
    env_var='BRAINSCORE_ALGONAUTS_ROOT',
    default_path='algonauts_2025',
    kind='stimuli',
    why_local='The stimuli are the Friends and Movie10 films; the fMRI is '
              'released under a data-use agreement the downloader signs.',
    source='https://github.com/courtois-neuromod/algonauts_2025.competitors',
    obtain='The competition dataset via DataLad (~130 GB: 129 GB stimuli, ~1 GB '
           'fMRI per subject). Ubuntu 22.04 ships a git-annex too old for '
           'current DataLad; install it from conda-forge instead.',
    prepare='python -m brainscore.data.algonauts2025.prepare_assembly '
            '--algonauts-root <root> --subjects 1',
    used_by=['Algonauts2025-friends-sub01'],
))
