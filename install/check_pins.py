"""Check that every UMI pin site names the same core and unified commits.

The coordinated v2 repositories pin each other in several files. A pin that
differs between sites makes a change pass in one CI system and fail in another.
This script reads every site and prints one table. It exits with status 1 when
a required site disagrees or cannot be read. The infrastructure Jenkins scripts
and the core layout workflow are reported as warnings only.

Each repository is a local checkout (a directory; its working tree is read) or
a Git ref on GitHub (a commit SHA or branch name). Without an argument, the
script uses the sibling directory under --root when it exists and the
unified-model-interface-v2 branch on GitHub otherwise.

    python install/check_pins.py --root /path/to/siblings
    python install/check_pins.py --core 9e421f6... --vision my-branch --language ../language

Only the standard library is used. GitHub reads use GH_TOKEN or GITHUB_TOKEN,
or `gh auth token` when neither is set.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
CALLERS = ('core', 'vision', 'language')
FULL_SHA = re.compile('[0-9a-f]{40}')
# Each pattern captures the complete ref token of an active (uncommented) line.
# judge() then requires a full 40-character SHA, so a partial or decorated ref
# such as "deadbeef-other" is reported as INVALID instead of matching a prefix.
REQUIREMENT_LINE = re.compile(r'^\s*brainscore-core\b.*$', re.MULTILINE)
REQUIREMENT_REF = re.compile(r'\.git@([^\s;]+)')
USES_UNIFIED = re.compile(
    r'^\s*(?:-\s*)?uses:\s*KartikP/brainscore-unified/\.github/workflows/offline-tests\.yml@(\S+)',
    re.MULTILINE)
INPUT_UNIFIED = re.compile(r'^\s*unified_revision:\s*(\S+)', re.MULTILINE)
LAYOUT_UNIFIED = re.compile(r'repository:\s*KartikP/brainscore-unified\s*\n\s*ref:\s*(\S+)')
JENKINS_UNIFIED = re.compile(r'^\s*unified_revision=(\S+)', re.MULTILINE)


def _token() -> Optional[str]:
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if token or shutil.which('gh') is None:
        return token
    result = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True)
    return result.stdout.strip() or None if result.returncode == 0 else None


def github_file(repository: str, ref: str, path: str) -> Optional[str]:
    """Return one file at one ref, or None when GitHub does not serve it."""
    url = f'https://api.github.com/repos/{repository}/contents/{quote(path)}?ref={quote(ref)}'
    headers = {'Accept': 'application/vnd.github.raw', 'User-Agent': 'umi-check-pins'}
    token = _token()
    if token:
        headers['Authorization'] = f'Bearer {token}'
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            return response.read().decode()
    except (HTTPError, URLError):
        return None


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

    def read_at(self, revision: str, path: str) -> Optional[str]:
        """Read a file at another commit of this repository."""
        if self.path is not None:
            result = subprocess.run(['git', '-C', str(self.path), 'show', f'{revision}:{path}'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
        return github_file(self.repository, revision, path)


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


def _active(text: str) -> str:
    """Drop comment lines and trailing comments, so a comment cannot set a pin."""
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith('#'):
            continue
        lines.append(re.sub(r'\s+#.*$', '', line))
    return '\n'.join(lines)


def _clean(value: str) -> str:
    return value.strip().strip('\'"')


def _matches(text: Optional[str], pattern: re.Pattern) -> list[Optional[str]]:
    """Return every distinct active ref; [None] when the file or entry is absent."""
    values = sorted({_clean(v) for v in pattern.findall(_active(text))}) if text is not None else []
    return values or [None]


def _requirement_refs(text: Optional[str]) -> list[Optional[str]]:
    """Return the ref of every active brainscore-core requirement.

    An active requirement without a ``.git@<ref>`` part is returned whole, so it
    is reported as INVALID rather than hidden behind another entry.
    """
    if text is None:
        return [None]
    values = set()
    for line in REQUIREMENT_LINE.findall(_active(text)):
        match = REQUIREMENT_REF.search(line)
        values.add(_clean(match.group(1)) if match else line.strip())
    return sorted(values) or [None]


def collect(sources: dict[str, Source], infrastructure: Optional[Source]) -> list[Row]:
    """Read every pin site. A missing file or pattern gives a row with value None."""
    rows = []
    manifest = sources['unified'].read('install/peer-revisions.json')
    rows.append(Row('core', 'unified install/peer-revisions.json',
                    json.loads(manifest).get('core') if manifest else None))
    for name in ('vision', 'language'):
        for value in _requirement_refs(sources[name].read('requirements-v2.txt')):
            rows.append(Row('core', f'{name} requirements-v2.txt', value))
    caller_pins = set()
    for name in CALLERS:
        workflow = sources[name].read('.github/workflows/umi-v2.yml')
        for pattern, field in ((USES_UNIFIED, 'uses'), (INPUT_UNIFIED, 'unified_revision')):
            for value in _matches(workflow, pattern):
                rows.append(Row('unified', f'{name} umi-v2.yml {field}', value))
                if value and FULL_SHA.fullmatch(value):
                    caller_pins.add(value)
    for revision in sorted(caller_pins):
        # The peer manifest of the pinned unified commit decides the core that
        # GitHub Actions installs for core, vision and language PRs.
        text = sources['unified'].read_at(revision, 'install/peer-revisions.json')
        rows.append(Row('core', f'unified@{revision[:12]} peer-revisions.json (caller CI)',
                        json.loads(text).get('core') if text else None))
    layout = sources['core'].read('.github/workflows/core-layout.yml')
    for value in _matches(layout, LAYOUT_UNIFIED):
        rows.append(Row('unified', 'core core-layout.yml checkout ref', value, required=False))
    if infrastructure is not None:
        for name in ('vision', 'language'):
            script = infrastructure.read(f'ci/{name}/run_tests.sh')
            for value in _matches(script, JENKINS_UNIFIED):
                rows.append(Row('unified', f'infrastructure ci/{name}/run_tests.sh', value,
                                required=False))
    return rows


def _valid(value: Optional[str]) -> bool:
    return isinstance(value, str) and FULL_SHA.fullmatch(value) is not None


def judge(rows: list[Row]) -> bool:
    """Mark each row; return True when every required site agrees.

    Every pin must be a full 40-character SHA, and pins agree only when they
    are identical.
    """
    ok = True
    for pin in ('core', 'unified'):
        group = [row for row in rows if row.pin == pin]
        votes = Counter(row.value for row in group if row.required and _valid(row.value))
        expected = votes.most_common(1)[0][0] if votes else None
        for row in group:
            if row.value is None:
                row.status = 'UNREADABLE'
            elif not _valid(row.value):
                row.status = 'INVALID'
            elif expected is None or row.value == expected:
                row.status = 'ok'
            else:
                row.status = 'MISMATCH'
            if row.status != 'ok':
                if row.required:
                    ok = False
                else:
                    row.status = f'warning: {row.status.lower()}'
    return ok


def table(rows: list[Row]) -> str:
    header = ('pin', 'site', 'value', 'status')
    body = [(row.pin, row.site, row.value[:12] if _valid(row.value) else str(row.value or "-")[:40],
             row.status) for row in rows]
    widths = [max(len(line[i]) for line in [header, *body]) for i in range(4)]
    lines = ['  '.join(cell.ljust(width) for cell, width in zip(line, widths)) for line in [header, *body]]
    lines.insert(1, '  '.join('-' * width for width in widths))
    return '\n'.join(lines)


def markdown(rows: list[Row], sources: list[Source]) -> str:
    lines = ['### UMI pin agreement', '', '| pin | site | value | status |', '|---|---|---|---|']
    lines += [f'| {row.pin} | {row.site} | `{row.value or "-"}` | {row.status} |' for row in rows]
    lines += ['', 'Sources: ' + ', '.join(f'{s.name}={s.label}' for s in sources), '']
    return '\n'.join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path,
                        help='Directory holding sibling checkouts (default: parent of this checkout)')
    for name in REPOSITORIES:
        parser.add_argument(f'--{name}', metavar='PATH|REF', help=f'{name} checkout or Git ref')
    parser.add_argument('--infrastructure', default='main', metavar='REF',
                        help='brain-score/infrastructure ref for the Jenkins warnings, or "skip"')
    args = parser.parse_args(argv)
    this_checkout = Path(__file__).resolve().parents[1]
    root = args.root or this_checkout.parent
    sources = {name: source(name, getattr(args, name), root,
                            this_checkout if name == 'unified' else None)
               for name in REPOSITORIES}
    infrastructure = None
    if args.infrastructure != 'skip':
        infrastructure = Source('infrastructure', INFRASTRUCTURE, ref=args.infrastructure)
    rows = collect(sources, infrastructure)
    ok = judge(rows)
    used = [*sources.values(), *([infrastructure] if infrastructure else [])]
    print('\n'.join(f'{s.name}: {s.label}' for s in used) + '\n')
    print(table(rows))
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a') as handle:
            handle.write(markdown(rows, used))
    print('\nPASS: required pin sites agree' if ok else
          '\nFAIL: required pin sites disagree or cannot be read; update them together')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
