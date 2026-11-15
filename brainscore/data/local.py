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

    LEBEL_PICKLE = local.register(local.LocalAsset(
        name='lebel2023-pickle',
        env_var='BRAINSCORE_LEBEL_PICKLE',
        default_path='Downloads/assembly_lebel_uts03.pkl',
        kind='neural data',
        why_local='Redistributed per-subject by the provider, not by Brain-Score.',
        source='https://github.com/GT-LIT-Lab/litcoder_release',
        obtain='Download the UTS03 assembly pickle.',
    ))

Reading it::

    path = local.path(LEBEL_PICKLE.name)       # raises LocalDataMissing if absent

Seeing what is missing, without running anything::

    python -m brainscore.data
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
    :param used_by: registered benchmark identifiers that need it; empty when
        only settings that no registered benchmark uses need it
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
        """Where it is expected, honouring the environment override.

        A leading ``~`` in the variable means the home folder. When a folder with
        that literal name (for example ``./~/frames``) also exists under the current
        folder, the value is ambiguous and is refused.
        """
        override = os.environ.get(self.env_var)
        if override:
            expanded = pathlib.Path(override).expanduser()
            literal = pathlib.Path.cwd() / override
            if override.startswith('~') and literal.exists() and literal != expanded:
                raise ValueError(
                    f'{self.env_var}={override!r} is ambiguous: it means {expanded} (the '
                    f'home folder), but {literal} also exists. Set {self.env_var} to the '
                    f'absolute path of the folder you mean.')
            return expanded
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
        try:
            where, present = str(asset.resolved()), asset.present()
        except ValueError as refused:            # a variable that names no single folder
            where, present = f'refused: {refused}', False
        rows.append({'name': asset.name, 'kind': asset.kind, 'present': present,
                     'path': where, 'used_by': asset.used_by})
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
