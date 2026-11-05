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
    from brainscore_core.environment_check import describe_env
    return describe_env()


# Imported, not just version-checked: a wheel built for another NumPy (pyarrow
# 26 under NumPy 1.26) installs cleanly and passes `pip check`, then fails here.
IMPORT_CHECKS = ('numpy', 'pandas', 'pyarrow', 'scipy', 'sklearn', 'xarray', 'torch', 'transformers')


def _import_rows():
    import importlib
    rows = []
    for name in IMPORT_CHECKS:
        try:
            importlib.import_module(name)
            rows.append((name, None))
        except Exception as error:      # ImportError, or an ABI error raised during import
            rows.append((name, f'{type(error).__name__}: {error}'.splitlines()[0]))
    return rows


# Each folder can be moved with its environment variable, set before Python starts.
STORAGE = (('result cache', 'RESULTCACHING_HOME', '~/.result_caching', 'results'),
           ('downloads', 'BRAINSCORE_HOME', '~/.brain-score', 'downloads'),
           ('datasets', 'BRAINIO_HOME', '~/.brainio', 'datasets'))


def _storage_rows():
    """(name, variable, path, problem or None) for each storage folder."""
    import os
    import tempfile
    from pathlib import Path
    rows = []
    for name, variable, default, folder in STORAGE:
        if variable == 'RESULTCACHING_HOME' and os.environ.get('RESULTCACHING_DISABLE') == '1':
            rows.append((name, variable, 'disabled (RESULTCACHING_DISABLE=1)', None))
            continue
        path = Path(os.environ.get(variable, default)).expanduser()
        fix = (f'Mount the drive it points to, or set {variable} to a folder on this disk '
               f'(e.g. export {variable}=$HOME/brainscore-cache/{folder}) before starting Python.')
        if path.is_symlink() and not path.exists():
            rows.append((name, variable, path,
                         f'{path} links to {os.readlink(path)}, which does not exist (an unmounted drive?). {fix}'))
            continue
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=path):
                pass
            rows.append((name, variable, path, None))
        except OSError as error:
            rows.append((name, variable, path, f'{path} is not writable ({error.strerror}). {fix}'))
    return rows


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
    lines.append('imports')
    for package, error in _import_rows():
        if error is None:
            lines.append(f'  ok {package}')
        else:
            lines.append(f'  XX {package:<14s} {error}')
            problems += 1

    lines.append('')
    lines.append('user-supplied data (benchmarks needing it will say so)')
    for row in _asset_rows():
        lines.append(f"  {'ok' if row['present'] else '--'} {row['name']:<20s} "
                     f"{row['kind']:<12s} {row['path']}")
    lines.append('')

    dependency_problems = problems
    lines.append('storage (where downloads and cached results go)')
    for name, variable, path, error in _storage_rows():
        if error is None:
            lines.append(f'  ok {name:<14s} {path}')
        else:
            lines.append(f'  XX {name:<14s} {error}')
            problems += 1
    lines.append('')

    if problems:
        lines.append(f'{problems} environment problem(s). Resolve the errors above before scoring.')
        if dependency_problems:
            lines.append('Dependency mismatches can make scoring wrong rather than merely broken: '
                         'install the pinned environment (environment-unified.yml).')
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
