"""Resolve and verify CI checkouts before installing Brain-Score packages.

Run this file directly: CI needs only Python's standard library at this stage.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.request import Request, urlopen


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


# One line of a pull request body names the partner pull requests of one change,
# for example "Candidate set: core#254 language#463". Each listed peer is tested
# at that pull request's head (its merge commit once it is merged) instead of the lock.
SET_FORM = 'Candidate set: core#<number> vision#<number> language#<number> (any of the three, once each)'
SET_LINE = re.compile(r'Candidate set:((?: +(?:core|vision|language)#[1-9][0-9]*)+) *')
SET_KEY = re.compile(r'candidate\s*set\s*:', re.IGNORECASE)


def _github_pull(repository: str, number: int) -> dict:
    headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'umi-candidate-set'}
    token = os.environ.get('GH_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    url = f'https://api.github.com/repos/{repository}/pulls/{number}'
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def candidate_set(body: str | None, lookup=_github_pull) -> dict:
    """Read the candidate set line of a pull request body into peer overrides.

    A body without the line gives no override. A line that starts with the key
    but does not have the form is refused, so a typing error never tests the lock
    in silence. ``lookup(repository, number)`` returns the GitHub pull request.
    """
    lines = [line.strip() for line in (body or '').splitlines()
             if SET_KEY.match(line.strip().lstrip('*_`'))]
    if not lines:
        return {}
    match = SET_LINE.fullmatch(lines[0]) if len(lines) == 1 else None
    entries = match.group(1).split() if match else []
    names = [entry.partition('#')[0] for entry in entries]
    if not match or len(set(names)) != len(names):
        raise ValueError(f'Malformed candidate set {lines!r}: write one line of the form '
                         f'"{SET_FORM}", or delete the line to test with the lock')
    overrides = {}
    for entry in entries:
        name, _, number = entry.partition('#')
        pull = lookup(REPOSITORIES[name], int(number))
        if pull.get('merged'):
            checkout = {'repository': REPOSITORIES[name], 'revision': pull['merge_commit_sha']}
        elif pull.get('state') == 'open' and (pull['head'].get('repo') or {}).get('full_name'):
            checkout = {'repository': pull['head']['repo']['full_name'],
                        'revision': pull['head']['sha']}
        else:
            raise ValueError(f'{entry} is closed without a merge or its source is gone: '
                             'remove it from the candidate set')
        overrides[name] = {'repository': _repository(checkout['repository']),
                           'revision': _revision(checkout['revision'])}
    return overrides


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
    overrides = dict(parse_override(value) for value in args.override)
    stated = candidate_set(os.environ.get('PULL_REQUEST_BODY'))
    if overrides.keys() & stated.keys():
        raise ValueError('Give a peer either as a workflow input or in the candidate set, not both')
    record = resolve(
        json.loads(args.manifest.read_text()),
        candidate=os.environ['CANDIDATE'],
        repository=os.environ['CANDIDATE_REPOSITORY'],
        revision=os.environ['CANDIDATE_REVISION'],
        unified_repository=os.environ['UNIFIED_REPOSITORY'],
        unified_revision=os.environ['UNIFIED_REVISION'],
        overrides={**overrides, **stated},
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
