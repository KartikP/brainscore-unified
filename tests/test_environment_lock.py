"""The full lock of the scoring platform and the doctor's comparison with it (offline)."""
import re
from pathlib import Path

import pytest

from brainscore import doctor

pytestmark = pytest.mark.unit

LOCK = Path(__file__).resolve().parents[1] / 'install' / 'lock-linux-py311-cu124.txt'


def test_the_lock_parses_with_every_line_pinned_and_no_duplicate():
    names, packages = [], 0
    for line in LOCK.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        if line.startswith('-'):
            assert line == '--extra-index-url https://download.pytorch.org/whl/cu124'
            continue
        packages += 1
        if ' @ ' in line:     # a package pinned to a git commit
            name, url = line.split(' @ ', 1)
            assert re.fullmatch(r'git\+https://\S+@[0-9a-f]{40}', url), line
        else:
            name, version = line.split('==', 1)
            assert re.fullmatch(r'[0-9][0-9A-Za-z.+!-]*', version), line
        names.append(doctor._normalized(name))
    assert len(names) == len(set(names)) == packages == 227
    lock = doctor.read_lock(LOCK)
    assert lock['torch'] == '2.6.0+cu124' and lock['quickbrain'] is None


def test_a_made_up_difference_is_reported():
    lock = {'numpy': '1.26.4', 'torch': '2.6.0+cu124', 'quickbrain': None}
    installed = {'numpy': '1.26.4', 'torch': '2.6.0', 'quickbrain': '0.1', 'extra-tool': '3.0',
                 'brainscore': '0.0', 'pip': '25.0'}
    differences = doctor.lock_differences(lock, installed)
    assert len(differences) == 2
    assert differences[0].startswith('extra-tool') and 'not in the lock' in differences[0]
    assert differences[1].startswith('torch') and 'lock 2.6.0+cu124' in differences[1]
    assert 'not installed' in doctor.lock_differences(lock, {'numpy': '1.26.4', 'torch': '2.6.0+cu124'})[0]


def test_the_doctor_reports_lock_differences_on_the_locked_platform(monkeypatch):
    lock = doctor.read_lock(LOCK)
    installed = dict(lock, numpy='1.26.3')
    installed['quickbrain'] = '0.1'
    monkeypatch.setattr(doctor, '_lock_file', lambda: LOCK)
    monkeypatch.setattr(doctor, '_installed_distributions', lambda: installed)
    monkeypatch.setattr(doctor, '_dependency_rows', lambda: [])
    monkeypatch.setattr(doctor, '_import_rows', lambda: [])
    monkeypatch.setattr(doctor, '_asset_rows', lambda: [])
    text, problems = doctor.report()
    assert 'lock-linux-py311-cu124.txt: 1 package(s) differ' in text
    assert re.search(r'XX numpy\s+1\.26\.3\s+lock 1\.26\.4', text)
    assert problems >= 1 and 'setup.sh --lock' in text


@pytest.mark.parametrize('platform_key', [('darwin', 'arm64', '3.11'), ('linux', 'aarch64', '3.11'),
                                          ('linux', 'x86_64', '3.12')])
def test_a_platform_without_a_lock_keeps_only_the_bounds(monkeypatch, platform_key):
    import sys
    system, machine, python = platform_key
    monkeypatch.setattr(sys, 'platform', system)
    monkeypatch.setattr(doctor.platform, 'machine', lambda: machine)
    monkeypatch.setattr(doctor.sys, 'version_info', tuple(int(p) for p in python.split('.')) + (0,))
    assert doctor._lock_file() is None
    monkeypatch.setattr(doctor, '_dependency_rows', lambda: [])
    monkeypatch.setattr(doctor, '_import_rows', lambda: [])
    monkeypatch.setattr(doctor, '_asset_rows', lambda: [])
    text, problems = doctor.report()
    assert 'full lock    none for this platform' in text
    assert 'XX' not in text.split('full lock', 1)[1].split('user-supplied', 1)[0]


# The one difference left between the CPU test profile and the reference lock: datasets is 2.21.0
# (the legacy wikitext plugin requires datasets 2.*); the lock still names 5.0.1 until it is made
# again from the reference machine. Remove the entry then.
OPEN_POINTS = {'datasets'}


def _pins(path):
    import re
    pins = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        match = re.match(r'^([A-Za-z0-9_.\-]+)==([^\s;#]+)', line.split('#')[0].strip())
        if match:
            pins[match.group(1).lower().replace('_', '-')] = match.group(2)
    return pins


def test_every_constraint_names_the_lock_version_without_its_local_tag():
    from pathlib import Path
    install = Path(__file__).resolve().parents[1] / 'install'
    constraints, lock = _pins(install / 'v2-constraints.txt'), _pins(install / 'lock-linux-py311-cu124.txt')
    differing = {name: (version, lock.get(name)) for name, version in constraints.items()
                 if (lock.get(name) or '').split('+')[0] != version}
    assert set(differing) == OPEN_POINTS, differing
    script = (install / 'run_candidate.sh').read_text(encoding='utf-8')
    assert f"TORCH_VERSION={constraints['torch']}\n" in script
    assert f"TORCHVISION_VERSION={constraints['torchvision']}\n" in script
