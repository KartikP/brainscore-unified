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

import platform
import re
import sys
from pathlib import Path


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


# The platforms with a full lock (every package at one version): (sys.platform, machine,
# Python major.minor) -> the lock file in unified/install. Elsewhere only the bounds apply.
LOCKS = {('linux', 'x86_64', '3.11'): 'lock-linux-py311-cu124.txt'}
# Installed from their checkouts, not from the lock; pip's own tools are not in a freeze.
_NOT_LOCKED = {'brainscore', 'brainscore-core', 'brainscore-vision', 'brainscore-language',
               'pip', 'setuptools', 'wheel'}


def _normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def read_lock(path):
    """``{normalized name: version}`` of a lock file; None as the version of a package
    pinned to a git commit (``name @ git+...@<commit>``). Option and comment lines are skipped."""
    lock = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '-')):
            continue
        if ' @ ' in line:
            lock[_normalized(line.split(' @ ', 1)[0])] = None
        else:
            name, version = line.split('==', 1)
            lock[_normalized(name)] = version
    return lock


def lock_differences(lock, installed):
    """One line for each package that differs from the lock: absent, another version, or not
    in the lock. ``installed`` is ``{normalized name: version}``."""
    lines = []
    for name in sorted(set(lock) | set(installed)):
        if name in _NOT_LOCKED:
            continue
        if name not in installed:
            lines.append(f'{name:<28s} not installed        lock {lock[name] or "(git commit)"}')
        elif name not in lock:
            lines.append(f'{name:<28s} {installed[name]:<20s} not in the lock')
        elif lock[name] is not None and installed[name] != lock[name]:
            lines.append(f'{name:<28s} {installed[name]:<20s} lock {lock[name]}')
    return lines


def _lock_file():
    """The lock of this platform in the unified checkout, or None (no lock for the platform,
    or an install without the checkout's install folder)."""
    key = (sys.platform, platform.machine(), f'{sys.version_info[0]}.{sys.version_info[1]}')
    name = LOCKS.get(key)
    if name is None:
        return None
    import brainscore
    path = Path(brainscore.__file__).resolve().parents[1] / 'install' / name
    return path if path.is_file() else None


def _installed_distributions():
    """``{normalized name: version}``: the fingerprint's inventory (startup entries, first one counts)."""
    from brainscore_core.environment_check import environment_distributions
    return {name: version for name, (version, _distribution) in environment_distributions().items()}


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

    from brainscore_core.environment_check import environment_fingerprint
    lines.append('')
    # The value every score's software carries: scores compare bit for bit only when it is equal.
    lines.append(f'environment fingerprint  {environment_fingerprint()}')

    lock = _lock_file()
    lines.append('')
    if lock is None:
        lines.append('full lock    none for this platform; the bounds above apply '
                     f'(locked platforms: {", ".join(" ".join(key) for key in LOCKS)})')
    else:
        differences = lock_differences(read_lock(lock), _installed_distributions())
        lines.append(f'full lock    {lock.name}: {len(differences)} package(s) differ')
        for difference in differences:
            lines.append(f'  XX {difference}')
        problems += len(differences)

    lines.append('')
    lines.append('user-supplied data (benchmarks needing it will say so)')
    for row in _asset_rows():
        lines.append(f"  {'ok' if row['present'] else '--'} {row['name']:<20s} "
                     f"{row['kind']:<12s} {row['path']}")
    lines.append('')

    dependency_problems = problems
    from brainscore_core.preflight import check_cache_directory
    try:
        cache = check_cache_directory()
        lines.append(f'result cache  {cache if cache is not None else "disabled"}')
    except OSError as error:
        lines.append(f'result cache  XX {error}')
        problems += 1
    lines.append('')

    if problems:
        lines.append(f'{problems} environment problem(s). Resolve the errors above before scoring.')
        if dependency_problems:
            lines.append('Dependency mismatches can make scoring wrong rather than merely broken: '
                         'install with the pinned versions (unified/install/setup.sh, or pip install '
                         'with -c unified/install/v2-constraints.txt; see the unified README).')
        if lock is not None:
            lines.append('On this platform `bash setup.sh --lock` installs every package at the '
                         'version of the full lock (unified/install/SETUP.md).')
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
