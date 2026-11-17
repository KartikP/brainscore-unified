"""The built package ships every file the code reads by path at run time: the pinned
requirements, the worker, its validation module, the supervisor, and the README whose code
blocks ``test_outside_library.py`` runs.

From a source checkout the wheel is built from a copy of the source with the installed build
backend, offline, and read as a ZIP. Against an installed package (no source tree next to the
tests) the installed distribution's record of its files is read. Nothing is installed.
"""
import importlib.metadata
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = ('brainscore/harnesses/coggym_requirements.txt', 'brainscore/harnesses/_coggym_worker.py',
                 'brainscore/harnesses/coggym_validation.py', 'brainscore/harnesses/_supervise.py',
                 'brainscore/harnesses/README.md')


def _packaged_files(tmp_path) -> set:
    if not (ROOT / 'pyproject.toml').exists():
        return {str(path) for path in importlib.metadata.files('brainscore')}
    source = tmp_path / 'source'
    shutil.copytree(ROOT / 'brainscore', source / 'brainscore',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', 'README.md', 'LICENSE'):
        if (ROOT / name).exists():
            shutil.copy(ROOT / name, source / name)
    build = ('import sys; from setuptools import build_meta; '
             'print(build_meta.build_wheel(sys.argv[1]))')
    completed = subprocess.run([sys.executable, '-c', build, str(tmp_path / 'dist')], cwd=source,
                               capture_output=True, text=True, timeout=600,
                               env={'PATH': '/usr/bin:/bin', 'PIP_NO_INDEX': '1', 'HOME': str(tmp_path)})
    assert completed.returncode == 0, completed.stderr[-2000:]
    wheel = tmp_path / 'dist' / completed.stdout.strip().splitlines()[-1]
    return set(zipfile.ZipFile(wheel).namelist())


def test_the_package_contains_every_file_the_code_reads_by_path(tmp_path):
    names = _packaged_files(tmp_path)
    assert [name for name in RUNTIME_FILES if name not in names] == []
