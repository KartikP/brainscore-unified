"""Check the lock of the UMI peers and the scripts that read it.

Unified's ``install/peer-revisions.json`` is the one lock: it names the core,
vision and language commits that every UMI check installs. Core, vision and
language follow the unified release branch, and the infrastructure Jenkins
scripts read the lock at the unified release tip. This script fails (status 1)
when:

- a lock entry is not a full commit SHA, or is not a commit of the peer's
  ``unified-model-interface-v2`` branch (read on GitHub), so the lock never names
  a commit that only a deletable branch holds;
- the vision or language commit of the lock has other code than the vision or
  language source it reads (only the peer files ``requirements-v2.txt`` and
  ``.github/workflows/umi-v2.yml`` may differ);
- an infrastructure script (``ci/core``, ``ci/vision``, ``ci/language``) names a
  unified commit instead of the release branch. A script that cannot be read
  (no token) is a warning.

Each repository is a local checkout (a directory; its working tree is read) or
a Git ref on GitHub (a commit SHA or branch name). Without an argument, the
script uses the sibling directory under --root when it exists and the
unified-model-interface-v2 branch on GitHub otherwise.

    python install/check_pins.py --root /path/to/siblings
    python install/check_pins.py --vision my-branch --language ../language

Only the standard library is used. GitHub reads use GH_TOKEN or GITHUB_TOKEN,
or `gh auth token` when neither is set.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


REPOSITORIES = {
    'core': 'brain-score/core',
    'vision': 'brain-score/vision',
    'language': 'brain-score/language',
    'unified': 'KartikP/brainscore-unified',
}
INFRASTRUCTURE = 'brain-score/infrastructure'
DEFAULT_BRANCH = 'unified-model-interface-v2'
FULL_SHA = re.compile('[0-9a-f]{40}')
# A Jenkins script names a unified commit (unified_revision=<sha>) or the release
# branch it fetches (unified_ref=<branch>); a value computed at run time ($(...)) is not a pin.
JENKINS_UNIFIED = re.compile(r'^\s*unified_(?:revision|ref)=([^\s$]\S*)', re.MULTILINE)
NOT_ON_BRANCH = 'not on the release branch: '   # a lock entry that the peer's release branch lacks
PINS_A_COMMIT = 'names a commit, not the release branch: '   # a Jenkins script with a fixed unified commit
CODE_DIFFERS = 'code differs: '    # the value of a peer source whose code is not the named commit's
# The files of a peer that only move pins. A peer commit that changes only these
# has the code of the commit before it, so the lock may name that earlier commit.
PEER_PIN_FILES = frozenset({'requirements-v2.txt', '.github/workflows/umi-v2.yml'})


def _token() -> Optional[str]:
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if token or shutil.which('gh') is None:
        return token
    result = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True)
    return result.stdout.strip() or None if result.returncode == 0 else None


def github_file(repository: str, ref: str, path: str, api: bool = False) -> Optional[str]:
    """Return one file at one ref, or None when GitHub does not serve it.

    ``api=True`` reads ``path`` as a JSON endpoint of the repository instead
    (for example ``git/trees/<ref>``).
    """
    url = (f'https://api.github.com/repos/{repository}/{quote(path, safe="/.")}' if api else
           f'https://api.github.com/repos/{repository}/contents/{quote(path)}?ref={quote(ref)}')
    headers = {'Accept': 'application/vnd.github+json' if api else 'application/vnd.github.raw',
               'User-Agent': 'umi-check-pins'}
    token = _token()
    if token:
        headers['Authorization'] = f'Bearer {token}'
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            return response.read().decode()
    except (HTTPError, URLError):
        return None


def _tree_difference(repository: str, base: str, head: str, prefix: str = '') -> Optional[set[str]]:
    """The paths whose content differs between two trees (or commits) on GitHub; None when unknown.

    Content, not history, decides: equal tree objects are equal code, so a
    squash merge or a commit on a kept branch with the same files gives no path.
    Only subtrees whose object differs are read.
    """
    if base == head:
        return set()
    entries = []
    for ref in (base, head):
        text = github_file(repository, ref, f'git/trees/{ref}', api=True)
        if text is None:
            return None
        entries.append({entry['path']: (entry['type'], entry['sha']) for entry in json.loads(text)['tree']})
    old, new = entries
    changed = set()
    for name in old.keys() | new.keys():
        if old.get(name) == new.get(name):
            continue
        if old.get(name, ('',))[0] == new.get(name, ('',))[0] == 'tree':
            inner = _tree_difference(repository, old[name][1], new[name][1], f'{prefix}{name}/')
            if inner is None:
                return None
            changed |= inner
        else:
            changed.add(f'{prefix}{name}')
    return changed


@dataclass
class Source:
    """One repository: a local checkout or a ref on GitHub."""
    name: str
    repository: str
    path: Optional[Path] = None
    ref: Optional[str] = None

    @property
    def label(self) -> str:
        if self.path is None:
            return f'{self.repository}@{self.ref}'
        head = subprocess.run(['git', '-C', str(self.path), 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True).stdout.strip()
        return f'{self.path} ({head or "no git"})'

    def read(self, path: str) -> Optional[str]:
        if self.path is None:
            return github_file(self.repository, self.ref, path)
        file = self.path / path
        return file.read_text() if file.is_file() else None

    def changed_since(self, revision: str) -> Optional[set[str]]:
        """The files that differ between ``revision`` and what this source reads; None when unknown."""
        if self.path is not None:
            result = subprocess.run(['git', '-C', str(self.path), 'diff', '--name-only', revision, '--'],
                                    capture_output=True, text=True)
            return set(result.stdout.split()) if result.returncode == 0 else None
        return _tree_difference(self.repository, revision, self.ref)


def source(name: str, value: Optional[str], root: Path, default_path: Optional[Path] = None) -> Source:
    repository = REPOSITORIES.get(name, INFRASTRUCTURE)
    if value is None:
        local = default_path or root / name
        if local.is_dir():
            return Source(name, repository, path=local.resolve())
        return Source(name, repository, ref=DEFAULT_BRANCH)
    if Path(value).is_dir():
        return Source(name, repository, path=Path(value).resolve())
    return Source(name, repository, ref=value)


@dataclass
class Row:
    pin: str
    site: str
    value: Optional[str]
    required: bool = True
    status: str = ''


def collect(sources: dict[str, Source], infrastructure: Optional[Source]) -> list[Row]:
    """Read the lock and the scripts that read it. A missing file or entry gives a row with value None."""
    rows = []
    manifest = sources['unified'].read('install/peer-revisions.json')
    peers = json.loads(manifest) if manifest else {}
    for name in ('core', 'vision', 'language'):
        named = peers.get(name)
        rows.append(Row(name, 'unified install/peer-revisions.json', named))
        rows.append(Row(name, f'{name} lock entry on the release branch',
                        on_release_branch(REPOSITORIES[name], named) if _valid(named) else named))
    for name in ('vision', 'language'):
        # The peer commit the lock names must have the code that the peer source holds.
        named = peers.get(name)
        changed = sources[name].changed_since(named) if _valid(named) else None
        other = sorted(changed - PEER_PIN_FILES) if changed is not None else None
        code = None if other is None else f'{CODE_DIFFERS}{other[0]}' if other else named
        rows.append(Row(name, f'{name} source: code of the named commit', code))
    if infrastructure is not None:
        for name in ('core', 'vision', 'language'):
            script = infrastructure.read(f'ci/{name}/run_tests.sh')
            values = sorted(set(JENKINS_UNIFIED.findall(script))) if script else []
            for value in values or [None]:
                follows = value == DEFAULT_BRANCH
                rows.append(Row('unified', f'infrastructure ci/{name}/run_tests.sh',
                                value if value is None or follows else f'{PINS_A_COMMIT}{value}',
                                required=script is not None))
    return rows


def on_release_branch(repository: str, revision: str) -> Optional[str]:
    """``revision`` when the release branch of ``repository`` holds it; None when unknown."""
    text = github_file(repository, revision, f'compare/{revision}...{DEFAULT_BRANCH}', api=True)
    if text is None:
        return None
    on_branch = json.loads(text).get('status') in ('ahead', 'identical')
    return revision if on_branch else f'{NOT_ON_BRANCH}{revision}'


def _valid(value: Optional[str]) -> bool:
    return isinstance(value, str) and FULL_SHA.fullmatch(value) is not None


def judge(rows: list[Row]) -> bool:
    """Mark each row; return True when every required row is ok.

    A lock entry must be a full 40-character SHA on the peer's release branch;
    an infrastructure script must name the release branch.
    """
    ok = True
    for row in rows:
        if row.value is None:
            row.status = 'UNREADABLE'
        elif row.value.startswith((CODE_DIFFERS, NOT_ON_BRANCH, PINS_A_COMMIT)):
            row.status = 'MISMATCH'
        elif _valid(row.value) or (row.pin == 'unified' and row.value == DEFAULT_BRANCH):
            row.status = 'ok'
        else:
            row.status = 'INVALID'
        if row.status != 'ok':
            if row.required:
                ok = False
            else:
                row.status = f'warning: {row.status.lower()}'
    return ok


def table(rows: list[Row]) -> str:
    header = ('pin', 'site', 'value', 'status')
    body = [(row.pin, row.site, row.value[:12] if _valid(row.value) else str(row.value or "-")[:96],
             row.status) for row in rows]
    widths = [max(len(line[i]) for line in [header, *body]) for i in range(4)]
    lines = ['  '.join(cell.ljust(width) for cell, width in zip(line, widths)) for line in [header, *body]]
    lines.insert(1, '  '.join('-' * width for width in widths))
    return '\n'.join(lines)


def markdown(rows: list[Row], sources: list[Source]) -> str:
    lines = ['### UMI lock check', '', '| pin | site | value | status |', '|---|---|---|---|']
    lines += [f'| {row.pin} | {row.site} | `{row.value or "-"}` | {row.status} |' for row in rows]
    lines += ['', 'Sources: ' + ', '.join(f'{s.name}={s.label}' for s in sources), '']
    return '\n'.join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path,
                        help='Directory holding sibling checkouts (default: parent of this checkout)')
    for name in REPOSITORIES:
        parser.add_argument(f'--{name}', metavar='PATH|REF', help=f'{name} checkout or Git ref')
    parser.add_argument('--infrastructure', default='main', metavar='PATH|REF',
                        help='brain-score/infrastructure checkout or ref for the Jenkins scripts, or "skip"')
    args = parser.parse_args(argv)
    this_checkout = Path(__file__).resolve().parents[1]
    root = args.root or this_checkout.parent
    sources = {name: source(name, getattr(args, name), root,
                            this_checkout if name == 'unified' else None)
               for name in REPOSITORIES}
    infrastructure = None
    if args.infrastructure != 'skip':
        infrastructure = source('infrastructure', args.infrastructure, root)
    rows = collect(sources, infrastructure)
    ok = judge(rows)
    used = [*sources.values(), *([infrastructure] if infrastructure else [])]
    print('\n'.join(f'{s.name}: {s.label}' for s in used) + '\n')
    print(table(rows))
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a') as handle:
            handle.write(markdown(rows, used))
    print('\nPASS: the lock and the scripts that read it are sound' if ok else
          '\nFAIL: move the lock to commits of the peer release branches (install/peer-revisions.json), '
          'or make the infrastructure script fetch the unified release branch')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
