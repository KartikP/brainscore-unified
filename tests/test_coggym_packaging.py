"""The built package ships what CogGym's environment needs: its pinned requirements and the worker.

The wheel is built from a copy of the source with the installed build backend, offline, and
read as a ZIP; nothing is installed.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


def test_the_wheel_contains_the_coggym_requirements_and_worker(tmp_path):
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
    names = set(zipfile.ZipFile(wheel).namelist())
    assert 'brainscore/harnesses/coggym_requirements.txt' in names
    assert 'brainscore/harnesses/_coggym_worker.py' in names
    assert 'brainscore/harnesses/coggym_validation.py' in names
    assert 'brainscore/harnesses/_supervise.py' in names
