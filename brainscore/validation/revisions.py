"""Resolve and verify CI checkouts before installing Brain-Score packages.

Run this file directly: CI needs only Python's standard library at this stage.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


REPOSITORIES = {
    'core': 'brain-score/core',
    'vision': 'brain-score/vision',
    'language': 'brain-score/language',
    'unified': 'KartikP/brainscore-unified',
}


def _revision(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{40}', value):
        raise ValueError('Every revision must be a full lowercase commit SHA')
    return value


def _repository(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', value):
        raise ValueError('Expected a GitHub owner/repository name')
    return value


def parse_override(value: str) -> tuple[str, dict]:
    """Parse ``name=sha`` or ``name=owner/repository@sha`` into one checkout."""
    name, separator, target = value.partition('=')
    if not separator or name not in ('core', 'vision', 'language'):
        raise ValueError(f'Override must be core, vision or language: {value!r}')
    repository, _, revision = target.rpartition('@')
    return name, {'repository': _repository(repository or REPOSITORIES[name]),
                  'revision': _revision(revision)}


def resolve(manifest: dict, *, candidate: str, repository: str, revision: str,
            unified_repository: str, unified_revision: str,
            overrides: dict | None = None) -> dict:
    """Replace the repository under test; retain all other peer pins.

    ``overrides`` replaces further peers, so several branch tips can be
    tested together before any of them is merged or pinned.
    """
    if candidate not in REPOSITORIES:
        raise ValueError(f'Unknown candidate: {candidate}')
    if set(manifest) != {'core', 'vision', 'language'}:
        raise ValueError('Peer manifest must contain core, vision and language')
    checkouts = {
        name: {'repository': REPOSITORIES[name], 'revision': _revision(sha)}
        for name, sha in manifest.items()
    }
    checkouts['unified'] = {
        'repository': _repository(unified_repository),
        'revision': _revision(unified_revision),
    }
    requested = {'repository': _repository(repository), 'revision': _revision(revision)}
    if candidate == 'unified' and requested != checkouts['unified']:
        raise ValueError('Unified checkout does not match the candidate under test')
    checkouts[candidate] = requested
    for name, checkout in (overrides or {}).items():
        if name not in ('core', 'vision', 'language'):
            raise ValueError(f'Only core, vision or language can be overridden: {name}')
        if name == candidate and checkout != requested:
            raise ValueError(f'Override for {name} conflicts with the candidate under test')
        checkouts[name] = {'repository': _repository(checkout['repository']),
                           'revision': _revision(checkout['revision'])}
    record = {'candidate': candidate, 'checkouts': checkouts}
    if overrides:
        record['overrides'] = sorted(overrides)
    return record


def verify(record: dict, root: Path, names: list[str]) -> None:
    """Reject a checkout that differs from the requested commit before building."""
    for name in names:
        if name not in REPOSITORIES:
            raise ValueError(f'Unknown checkout: {name}')
        expected = _revision(record['checkouts'][name]['revision'])
        actual = subprocess.check_output(
            ['git', '-C', str(root / name), 'rev-parse', 'HEAD'], text=True,
        ).strip()
        if actual != expected:
            raise ValueError(f'{name}: requested {expected}, checked out {actual}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--verify', nargs='*', choices=tuple(REPOSITORIES))
    parser.add_argument('--override', action='append', default=[], metavar='NAME=[OWNER/REPO@]SHA',
                        help='Also replace this peer pin (repeat for several peers)')
    args = parser.parse_args()
    if args.verify is not None:
        verify(json.loads(args.record.read_text()), args.root,
               args.verify or list(REPOSITORIES))
        return
    if args.manifest is None:
        parser.error('--manifest is required when resolving revisions')
    record = resolve(
        json.loads(args.manifest.read_text()),
        candidate=os.environ['CANDIDATE'],
        repository=os.environ['CANDIDATE_REPOSITORY'],
        revision=os.environ['CANDIDATE_REVISION'],
        unified_repository=os.environ['UNIFIED_REPOSITORY'],
        unified_revision=os.environ['UNIFIED_REVISION'],
        overrides=dict(parse_override(value) for value in args.override),
    )
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + '\n')
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            for name, checkout in record['checkouts'].items():
                for key, value in checkout.items():
                    print(f'{name}_{key}={value}', file=output)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
