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

Each data plugin declares its own assets in its ``__init__.py``, beside its
registry lines. Plugin ``__init__`` files import no ML library, so listing the
assets (``python -m brainscore.data``) stays light. Declaring an asset::

    from brainscore.data import local

    LANA_ATLAS = local.register(local.LocalAsset(
        name='lana-atlas',                     # read at $BRAINSCORE_DATA_HOME/lana-atlas
        kind='atlas',
        why_local='Fedorenko-lab resource distributed under its own terms.',
        source='https://osf.io/kzwbh/',
        obtain='The "FS Atlas" archive; unzip it.',
    ))

Reading it::

    path = local.path(LANA_ATLAS.name)         # raises LocalDataMissing if absent

Seeing what is missing, without running anything::

    python -m brainscore.data
"""

import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class LocalDataMissing(FileNotFoundError):
    """A required local asset is absent. The message says how to get it."""


HOME_VARIABLE = 'BRAINSCORE_DATA_HOME'


def data_home() -> pathlib.Path:
    """The one folder of local unified data: ``$BRAINSCORE_DATA_HOME``, else ``~/brainscore-data``.

    It holds ``sources/<name>`` (the original source of ``package_data(name)``) and one folder
    for each local asset, ``<asset name>``. A data set on another drive: put a link at its place.
    """
    return pathlib.Path(os.environ.get(HOME_VARIABLE) or '~/brainscore-data').expanduser().absolute()


@dataclass(frozen=True)
class LocalAsset:
    """One file or directory the user supplies, at ``$BRAINSCORE_DATA_HOME/<name>``.

    :param name: stable identifier, referenced by :func:`path`, and the folder name of its place
    :param kind: ``stimuli``, ``neural data`` or ``atlas`` — what is withheld
    :param why_local: why Brain-Score cannot ship it; shown to the user
    :param source: where to get it
    :param obtain: what to download, in a sentence
    :param prepare: shell command converting it into the benchmark's format,
        if conversion is needed. Downloads that are read directly leave this empty.
    :param used_by: registered benchmark identifiers that need it; empty when
        only settings that no registered benchmark uses need it
    """

    name: str
    kind: str
    why_local: str
    source: str
    obtain: str
    prepare: str = ''
    used_by: List[str] = field(default_factory=list)

    def resolved(self) -> pathlib.Path:
        """Where it is expected: ``$BRAINSCORE_DATA_HOME/<name>``."""
        return data_home() / self.name

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
            f'Then put it at the path above ({HOME_VARIABLE}/{self.name}), or put a link there '
            f'to where it already is.',
        ]
        if self.prepare:
            lines += ['', 'Then convert it:', f'    {self.prepare}']
        if self.used_by:
            lines += ['', f'Needed by: {", ".join(self.used_by)}']
        return '\n'.join(lines)


REGISTRY: Dict[str, LocalAsset] = {}


def register(asset: LocalAsset) -> LocalAsset:
    """Declare an asset. A second asset with the same name is refused."""
    if asset.name in REGISTRY:
        raise ValueError(f'a local asset named {asset.name!r} is already declared')
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
    rows = []
    for asset in sorted(assets, key=lambda a: a.name):
        rows.append({'name': asset.name, 'kind': asset.kind, 'present': asset.present(),
                     'path': str(asset.resolved()), 'used_by': asset.used_by})
    return rows


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


def check_benchmark_assets(identifier: str) -> None:
    """Check declared local assets without constructing a model or reading data."""
    for asset in REGISTRY.values():
        if identifier in asset.used_by:
            path(asset.name)
