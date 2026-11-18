"""The per-pull-request selection leaves out every qualification test.

``tests/markers.py`` marks a test without a tier ``unit``, so most qualification tests
are ``unit`` tests too. ``unit or integration and not qualification`` reads as
``unit or (integration and not qualification)`` and keeps them; the selection
must bracket the tiers.
"""
import subprocess
import sys

import pytest

from brainscore.validation.workspace import SUITES, suite_arguments

STAND_INS = '''
import pytest

@pytest.mark.unit
def test_plain_unit():
    pass

@pytest.mark.unit
@pytest.mark.qualification
def test_unit_qualification():
    pass

@pytest.mark.integration
@pytest.mark.qualification
def test_integration_qualification():
    pass
'''


def _selected(folder, marks):
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', '-p', 'no:cacheprovider',
         '-W', 'ignore::pytest.PytestUnknownMarkWarning', '-m', marks, 'test_stand_ins.py'],
        cwd=folder, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return sorted(line.split('::')[1] for line in result.stdout.splitlines() if '::' in line)


@pytest.mark.parametrize('full, expected', [
    (None, ['test_plain_unit']),
    ('1', ['test_integration_qualification', 'test_plain_unit', 'test_unit_qualification']),
])
def test_the_unified_selection_leaves_out_qualification_tests_unless_full(tmp_path, monkeypatch, full, expected):
    if full is None:
        monkeypatch.delenv('UMI_FULL', raising=False)
    else:
        monkeypatch.setenv('UMI_FULL', full)
    (tmp_path / 'test_stand_ins.py').write_text(STAND_INS)
    arguments = suite_arguments('unified', SUITES['unified'])
    marks = arguments[arguments.index('-m') + 1]
    assert _selected(tmp_path, marks) == expected


def test_the_marking_hook_is_registered_once(request):
    # Both conftest files (tests/, brainscore/benchmarks/) name the marking plugin; a joint
    # collection must still run it once, or every test gets its markers twice.
    from tests import markers
    hooks = request.config.pluginmanager.hook.pytest_collection_modifyitems.get_hookimpls()
    assert [hook.function for hook in hooks].count(markers.pytest_collection_modifyitems) == 1
