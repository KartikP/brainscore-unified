"""Ensure installed qualification cannot pass by importing a source checkout."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from brainscore.validation import installed, workspace


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    package = 'umi_fixture_package'
    site = tmp_path / 'site-packages'
    root = site / package
    root.mkdir(parents=True)
    content = b'value = 1\n'
    module = root / '__init__.py'
    module.write_bytes(content)
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip('=')
    wheel = tmp_path / f'{package}-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr(f'{package}/__init__.py', content)
        archive.writestr(f'{package}-1.0.dist-info/METADATA', 'Name: umi-fixture-package\nVersion: 1.0\n')
        archive.writestr(f'{package}-1.0.dist-info/RECORD',
                         f'{package}/__init__.py,sha256={digest},{len(content)}\n')
    distribution = SimpleNamespace(version='1.0', locate_file=lambda name: site / name,
                                   read_text=lambda name: None)
    monkeypatch.setattr(installed, 'PACKAGES', (package,))
    monkeypatch.setattr(installed.metadata, 'distribution', lambda name: distribution)
    monkeypatch.setattr(installed.sysconfig, 'get_path', lambda name: str(site))
    monkeypatch.setattr(installed.importlib, 'import_module',
                        lambda name: SimpleNamespace(__file__=str(module)))
    return package, module, wheel, distribution


def test_candidate_records_hash_version_and_installed_files(candidate, tmp_path):
    package, module, wheel, _ = candidate
    result = installed.verify_wheels(tmp_path)[package]
    assert result['version'] == '1.0'
    assert result['import_root'] == str(module.parent)
    assert result['verified_files'] == 1
    assert result['wheel_sha256'] == hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_package_replacement_during_tests_fails(candidate, tmp_path, monkeypatch):
    _, module, _, _ = candidate
    identity = tmp_path / 'identity.json'
    monkeypatch.setattr(sys, 'argv', ['installed', '--stage', str(tmp_path),
                                    '--wheels', str(tmp_path), '--identity', str(identity)])

    def test_run(*args, **kwargs):
        module.write_text('replaced = True\n')
        return 0

    monkeypatch.setattr(installed.pytest, 'main', test_run)
    # main() adds the copied test directory to sys.path.
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    assert installed.main() == 1
    report = json.loads(identity.read_text())
    assert not report['complete']
    assert 'differs from candidate wheel' in report['error']


@pytest.mark.parametrize('defect', ['changed_file', 'missing_file', 'editable', 'version', 'source_import', 'duplicate', 'outside_site'])
def test_rejects_wrong_installation(candidate, tmp_path, monkeypatch, defect):
    package, module, wheel, distribution = candidate
    if defect == 'changed_file':
        module.write_text('value = 2\n')
    elif defect == 'missing_file':
        module.unlink()
    elif defect == 'editable':
        distribution.read_text = lambda name: '{"dir_info": {"editable": true}}'
    elif defect == 'version':
        distribution.version = '2.0'
    elif defect == 'source_import':
        monkeypatch.setattr(installed.importlib, 'import_module',
                            lambda name: SimpleNamespace(__file__=str(tmp_path / 'source.py')))
    elif defect == 'duplicate':
        (tmp_path / f'{package}-2.0-py3-none-any.whl').write_bytes(wheel.read_bytes())
    elif defect == 'outside_site':
        distribution.locate_file = lambda name: tmp_path / 'source' / name
    with pytest.raises((ValueError, FileNotFoundError)):
        installed.verify_wheels(tmp_path)


def test_source_import_after_collection_fails_the_run(tmp_path, monkeypatch):
    root = tmp_path / 'site-packages' / 'umi_fixture'
    guard = installed.InstalledImports({'umi_fixture': {'import_root': str(root)}})
    module = SimpleNamespace(__file__=str(root / '__init__.py'))
    monkeypatch.setitem(sys.modules, 'umi_fixture', module)
    guard.pytest_collection_finish(None)
    assert not guard.violations
    monkeypatch.setitem(sys.modules, 'umi_fixture.late',
                        SimpleNamespace(__file__=str(tmp_path / 'checkout' / 'late.py')))
    guard.pytest_runtest_logreport(None)
    # Remember the violation even if a fixture removes the source module later.
    monkeypatch.delitem(sys.modules, 'umi_fixture.late')
    session = SimpleNamespace(exitstatus=0)
    guard.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 1
    assert len(guard.violations) == 1


def test_staging_keeps_fixtures_but_excludes_package_sources(tmp_path):
    checkout, stage = tmp_path / 'checkout', tmp_path / 'stage'
    stage.mkdir()
    for name in ('tests/fixtures/input.csv', 'examples/demo.py', 'install/config.json', '.github/workflows/tests.yml',
                 'notebooks/demo.ipynb', 'templates/new_model/model.py',
                 'brainscore/__init__.py', 'tests/__pycache__/test_demo.pyc'):
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    (checkout / 'pyproject.toml').write_text('')
    workspace.stage_tests(checkout, stage)
    assert (stage / 'tests/fixtures/input.csv').is_file()
    assert (stage / 'examples/demo.py').is_file()
    assert (stage / 'pyproject.toml').is_file()
    assert (stage / '.github/workflows/tests.yml').is_file()
    assert (stage / 'notebooks/demo.ipynb').is_file()
    assert (stage / 'templates/new_model/model.py').is_file()
    assert not (stage / 'brainscore').exists()
    assert not (stage / 'tests/__pycache__').exists()


UNIFIED = Path(__file__).resolve().parents[1]


@pytest.fixture
def libero_checkout(tmp_path, monkeypatch):
    """A checkout whose helper differs from a fake installed brainscore distribution."""
    checkout = tmp_path / 'checkout'
    evaluator = checkout / 'examples/libero/evaluate.py'
    evaluator.parent.mkdir(parents=True)
    evaluator.write_bytes((UNIFIED / 'examples/libero/evaluate.py').read_bytes())
    source = checkout / workspace.EVALUATOR_HELPER
    source.parent.mkdir(parents=True)
    source.write_text('raise RuntimeError("source checkout helper")\n')
    (checkout / 'brainscore/__init__.py').write_text('')
    site = tmp_path / 'site-packages'
    installed_helper = site / workspace.EVALUATOR_HELPER
    installed_helper.parent.mkdir(parents=True)
    content = (UNIFIED / workspace.EVALUATOR_HELPER).read_bytes() + b'INSTALLED = True\n'
    installed_helper.write_bytes(content)
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip('=')
    dist_info = site / 'brainscore-1.0.dist-info'
    dist_info.mkdir()
    (dist_info / 'METADATA').write_text('Name: brainscore\nVersion: 1.0\n')
    (dist_info / 'RECORD').write_text(f'{workspace.EVALUATOR_HELPER},sha256={digest},{len(content)}\n')
    distribution = workspace.metadata.Distribution.at(dist_info)
    monkeypatch.setattr(workspace.metadata, 'distribution',
                        lambda name: distribution if name == 'brainscore' else None)
    return checkout, installed_helper


def test_staging_supplies_installed_evaluator_helper_without_package_source(tmp_path, libero_checkout):
    checkout, installed_helper = libero_checkout
    stage = tmp_path / 'stage'
    stage.mkdir()
    workspace.stage_tests(checkout, stage)
    assert (stage / workspace.EVALUATOR_HELPER).read_bytes() == installed_helper.read_bytes()
    assert sorted(str(p.relative_to(stage)) for p in (stage / 'brainscore').rglob('*')) == [
        'brainscore/experiments', 'brainscore/experiments/backends', workspace.EVALUATOR_HELPER]
    # The staged evaluator loads that copy by path and never imports brainscore.
    probe = ('import importlib.util, sys\n'
             f'spec = importlib.util.spec_from_file_location("evaluate", {str(stage / "examples/libero/evaluate.py")!r})\n'
             'module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n'
             'assert module.socket_deadline.INSTALLED\n'
             'assert not [name for name in sys.modules if name.split(".")[0] == "brainscore"]\n'
             'print(module.socket_deadline.__file__)\n')
    run = subprocess.run([sys.executable, '-I', '-c', probe], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert Path(run.stdout.strip()) == (stage / workspace.EVALUATOR_HELPER).resolve()


def test_staging_rejects_helper_that_differs_from_its_record(tmp_path, libero_checkout):
    checkout, installed_helper = libero_checkout
    installed_helper.write_bytes(installed_helper.read_bytes() + b'# edited\n')
    stage = tmp_path / 'stage'
    stage.mkdir()
    with pytest.raises(ValueError, match='differs from its RECORD'):
        workspace.stage_tests(checkout, stage)


def test_staging_rejects_helper_resolving_to_the_checkout(tmp_path, libero_checkout, monkeypatch):
    checkout, _ = libero_checkout
    # An editable or path install would locate the helper inside the source checkout.
    class Entry(str):
        hash = SimpleNamespace(mode='sha256', value='x')
    monkeypatch.setattr(workspace.metadata, 'distribution', lambda name: SimpleNamespace(
        files=[Entry(workspace.EVALUATOR_HELPER)],
        locate_file=lambda entry: checkout / workspace.EVALUATOR_HELPER))
    stage = tmp_path / 'stage'
    stage.mkdir()
    with pytest.raises(ValueError, match='source checkout'):
        workspace.stage_tests(checkout, stage)


@pytest.mark.parametrize('identity', [None, {'complete': False, 'returncode': 0},
    {'complete': True, 'returncode': 0, 'import_violations': ['source import']}])
def test_workspace_requires_installed_evidence(tmp_path, monkeypatch, identity):
    checkout = tmp_path / 'repos' / 'unified'
    (checkout / 'tests').mkdir(parents=True)
    monkeypatch.setattr(workspace, 'SUITES', {'unified': ['tests']})
    monkeypatch.setenv('PYTHONPATH', str(checkout))
    monkeypatch.setenv('PYTEST_ADDOPTS', '--ignore=tests')
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'test-only-value')

    def git(command, **kwargs):
        if command[1] == 'rev-parse':
            return 'a' * 40
        return '' if kwargs.get('text') else b''

    def run(command, *, cwd, env, **kwargs):
        assert command[1:4] == ['-I', '-m', 'brainscore.validation.installed']
        assert cwd != checkout
        assert 'PYTHONPATH' not in env and 'PYTEST_ADDOPTS' not in env
        assert 'AWS_ACCESS_KEY_ID' not in env
        assert env['AWS_SHARED_CREDENTIALS_FILE'] == os.devnull
        assert Path(env['HOME']).is_relative_to(cwd)
        checks = Path(command[command.index('--umi-check-report') + 1])
        checks.write_text(json.dumps({'returncode': 0, 'passed': ['tests/test_ok.py::test_ok']}))
        if identity is not None:
            Path(command[command.index('--identity') + 1]).write_text(json.dumps(identity))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(workspace.subprocess, 'check_output', git)
    monkeypatch.setattr(workspace.subprocess, 'run', run)
    output = tmp_path / 'report.json'
    assert workspace.qualify(checkout.parent, output, mode='installed', wheels=tmp_path) == 1
    assert json.loads(output.read_text())['tier'] == 'installed-offline'
