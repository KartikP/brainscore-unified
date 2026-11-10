"""The pin check must fail when required pin sites disagree."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'install' / 'check_pins.py'
CORE = 'a' * 40
UNIFIED = 'b' * 40


@pytest.fixture
def pins(monkeypatch):
    spec = importlib.util.spec_from_file_location('check_pins', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through sys.modules.
    monkeypatch.setitem(sys.modules, 'check_pins', module)
    spec.loader.exec_module(module)
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
    served = {UNIFIED: json.dumps({'core': CORE, 'vision': 'c' * 40, 'language': 'd' * 40})}
    monkeypatch.setattr(module, 'github_file',
                        lambda repository, ref, path: served.get(ref))
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _workflow(unified: str, field: str = None) -> str:
    return ('jobs:\n  coordinated:\n'
            f'    uses: KartikP/brainscore-unified/.github/workflows/offline-tests.yml@{unified}\n'
            f'    with:\n      unified_revision: {field or unified}\n')


def _repositories(root: Path, *, vision_core: str = CORE, language_unified: str = UNIFIED,
                  layout_unified: str = UNIFIED) -> Path:
    _write(root / 'unified/install/peer-revisions.json',
           json.dumps({'core': CORE, 'vision': 'c' * 40, 'language': 'd' * 40}))
    for name, core in (('vision', vision_core), ('language', CORE)):
        _write(root / name / 'requirements-v2.txt',
               f'brainscore-core @ git+https://github.com/brain-score/core.git@{core}\n')
    for name, unified in (('core', UNIFIED), ('vision', UNIFIED), ('language', language_unified)):
        _write(root / name / '.github/workflows/umi-v2.yml', _workflow(unified))
    _write(root / 'core/.github/workflows/core-layout.yml',
           f'      - uses: actions/checkout@v4\n        with:\n'
           f'          repository: KartikP/brainscore-unified\n          ref: {layout_unified}\n')
    return root


def _run(module, root: Path) -> int:
    arguments = []
    for name in ('core', 'vision', 'language', 'unified'):
        arguments += [f'--{name}', str(root / name)]
    return module.main([*arguments, '--infrastructure', 'skip'])


def test_agreeing_pins_pass(pins, tmp_path, capsys):
    assert _run(pins, _repositories(tmp_path)) == 0
    assert 'PASS' in capsys.readouterr().out


def test_core_pin_disagreement_fails_with_a_table(pins, tmp_path, capsys):
    assert _run(pins, _repositories(tmp_path, vision_core='e' * 40)) == 1
    out = capsys.readouterr().out
    assert 'vision requirements-v2.txt' in out and 'MISMATCH' in out


def test_caller_unified_disagreement_fails(pins, tmp_path, capsys):
    assert _run(pins, _repositories(tmp_path, language_unified='f' * 40)) == 1
    out = capsys.readouterr().out
    assert 'language umi-v2.yml uses' in out
    # The manifest of an unknown unified commit cannot be read.
    assert 'UNREADABLE' in out


def test_warning_sites_do_not_fail(pins, tmp_path, capsys):
    assert _run(pins, _repositories(tmp_path, layout_unified='9' * 40)) == 0
    assert 'warning: mismatch' in capsys.readouterr().out


def test_abbreviated_sha_is_invalid(pins):
    rows = [pins.Row('core', 'one', CORE), pins.Row('core', 'two', CORE[:7])]
    assert not pins.judge(rows)
    assert [row.status for row in rows] == ['ok', 'INVALID']


def test_decorated_ref_does_not_match_its_prefix(pins, tmp_path, capsys):
    root = _repositories(tmp_path, vision_core=CORE + '-other')
    assert _run(pins, root) == 1
    assert 'INVALID' in capsys.readouterr().out


def test_comment_cannot_hide_an_active_requirement(pins, tmp_path, capsys):
    root = _repositories(tmp_path)
    _write(root / 'vision/requirements-v2.txt',
           f'# brainscore-core @ git+https://github.com/brain-score/core.git@{CORE}\n'
           'brainscore-core @ git+https://github.com/brain-score/core.git@main\n')
    assert _run(pins, root) == 1
    out = capsys.readouterr().out
    assert 'INVALID' in out and 'main' in out


def test_commented_workflow_pin_is_ignored(pins, tmp_path):
    root = _repositories(tmp_path)
    _write(root / 'vision/.github/workflows/umi-v2.yml',
           '# uses: KartikP/brainscore-unified/.github/workflows/offline-tests.yml@' + 'c' * 40
           + '\n' + _workflow(UNIFIED))
    assert _run(pins, root) == 0


def test_missing_required_site_fails(pins, tmp_path):
    root = _repositories(tmp_path)
    (root / 'vision/requirements-v2.txt').unlink()
    assert _run(pins, root) == 1


def test_mismatch_inside_one_workflow_fails(pins, tmp_path):
    root = _repositories(tmp_path)
    _write(root / 'core/.github/workflows/umi-v2.yml', _workflow(UNIFIED, field='c' * 40))
    assert _run(pins, root) == 1
