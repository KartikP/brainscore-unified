"""``python -m brainscore.doctor`` — is this environment able to score?

Two things silently produce wrong or missing results: dependency versions
outside the range scoring is verified against, and user-supplied data that is
not where the code expects. Both are reported here, in one command, before a
run rather than during one.

This exists because several environments tend to accumulate on one machine and
the drifted one is not obviously distinguishable from the good one at a glance.
The interpreter path is printed first for that reason: knowing *which* env
answered is usually the whole question.
"""

import sys


def _dependency_rows():
    from brainscore_core._env_check import describe_env
    return describe_env()


def _asset_rows():
    from brainscore.data import local
    return local.status()


def report():
    """Human-readable environment report. Returns (text, n_problems)."""
    lines, problems = [], 0

    lines.append(f'interpreter  {sys.executable}')
    # conda names the env in the path; a venv usually does too.
    parts = sys.executable.split('/')
    if 'envs' in parts:
        lines.append(f'environment  {parts[parts.index("envs") + 1]}')
    lines.append('')

    lines.append('dependencies (versions scoring is verified against)')
    for package, installed, expected, ok in _dependency_rows():
        if installed is None:
            mark, detail = '--', f'not installed        {expected}'
            problems += 1
        elif ok:
            mark, detail = 'ok', f'{installed:<20s} {expected}'
        else:
            mark, detail = 'XX', f'{installed:<20s} {expected}  <- outside'
            problems += 1
        lines.append(f'  {mark} {package:<14s} {detail}')

    lines.append('')
    lines.append('user-supplied data (benchmarks needing it will say so)')
    for row in _asset_rows():
        lines.append(f"  {'ok' if row['present'] else '--'} {row['name']:<20s} "
                     f"{row['kind']:<12s} {row['path']}")
    lines.append('')

    if problems:
        lines.append(f'{problems} dependency problem(s). Scoring may be wrong '
                     f'rather than merely broken: install the pinned '
                     f'environment (environment-unified.yml).')
    else:
        lines.append('Dependencies are within bounds. Missing data assets above '
                     'are only needed by the benchmarks that name them; run '
                     '`python -m brainscore.data <name>` for how to obtain one.')
    return '\n'.join(lines), problems


def main(argv=None):
    text, problems = report()
    print(text)
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
