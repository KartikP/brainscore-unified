"""The evaluation method is ``participate``; the provisional ``interact`` is gone.

A static check over maintained executable paths: the four runtime packages'
contract code, the domain adapters, examples, templates and notebook code.
Files are parsed, never imported, so this needs no Torch or model weights.
Legacy plugins, vendored code and dated qualification reports are not covered.
"""
import ast
import json
import re
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
MAINTAINED = [
    ('brainscore_core', ''),
    ('brainscore_vision', 'compat'),
    ('brainscore_vision', '__init__.py'),
    ('brainscore_language', 'compat'),
    ('brainscore_language', '__init__.py'),
    ('brainscore', ''),
]
# The old method name in prose or messages; ordinary English ("interact with",
# "interaction", "interactive") is fine.
OLD_WORD = re.compile(r'\binteract(?!ion|ive|ing| with)(\b|_)')


# Stable record-format values of response meta["driver"]; saved records compare
# them exactly. Allowed only as a ``driver`` default, keyword or dict value.
RECORD_LABELS = {'interact', 'interact_streaming'}


def _record_labels(tree):
    """ids of the Constant nodes that are ``driver`` record labels."""
    allowed = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional = node.args.posonlyargs + node.args.args
            pairs = list(zip(positional[len(positional) - len(node.args.defaults):],
                             node.args.defaults))
            pairs += list(zip(node.args.kwonlyargs, node.args.kw_defaults))
            allowed += [value for arg, value in pairs if arg.arg == 'driver']
        elif isinstance(node, ast.keyword) and node.arg == 'driver':
            allowed.append(node.value)
        elif isinstance(node, ast.Dict):
            allowed += [value for key, value in zip(node.keys, node.values)
                        if isinstance(key, ast.Constant) and key.value == 'driver']
    return {id(value) for value in allowed
            if isinstance(value, ast.Constant) and value.value in RECORD_LABELS}


def old_name_uses(source):
    """(line, text) for each use of the old evaluation name in Python source."""
    found = []
    tree = ast.parse(source)
    labels = _record_labels(tree)
    for node in ast.walk(tree):
        if id(node) in labels:
            continue
        name = (node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else
                node.attr if isinstance(node, ast.Attribute) else
                node.arg if isinstance(node, ast.keyword) else
                node.id if isinstance(node, ast.Name) else None)
        if name is not None and (name == 'interact' or name.startswith('interact_')):
            found.append((node.lineno, name))
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and OLD_WORD.search(node.value)):
            found.append((node.lineno, node.value[:60]))
    return found


def _python_files():
    for package, inside in MAINTAINED:
        spec = find_spec(package)  # a top-level lookup does not import the package
        assert spec is not None and spec.submodule_search_locations, package
        target = Path(list(spec.submodule_search_locations)[0]) / inside
        assert target.exists(), target
        yield from [target] if target.is_file() else sorted(target.rglob('*.py'))
    for folder in ('examples', 'templates'):
        yield from sorted((REPO / folder).rglob('*.py'))


def test_maintained_code_never_uses_the_old_evaluation_name():
    offenders = [f'{path}:{line} {what}' for path in _python_files()
                 for line, what in old_name_uses(path.read_text(encoding='utf-8'))]
    assert not offenders, 'old evaluation name:\n' + '\n'.join(offenders)


def test_notebook_code_never_uses_the_old_evaluation_name():
    offenders = []
    for path in sorted((REPO / 'notebooks').glob('*.ipynb')):
        for index, cell in enumerate(json.loads(path.read_text(encoding='utf-8'))['cells']):
            source = ''.join(cell['source'])
            if cell['cell_type'] == 'code' and OLD_WORD.search(source):
                offenders.append(f'{path.name} cell {index}')
    assert not offenders, offenders


@pytest.mark.parametrize('source', [
    'class S:\n    def interact(self, session): pass',
    'subject.interact(session)',
    'SimpleNamespace(interact=lambda session: None)',
    'driver = "interact_streaming"',
    'raise TypeError("implement interact(session)")',
    'getattr(subject, "interact")(session)',
    'def drive(subject, mode="interact"): pass',
    'meta = {"driver": "interact_v2"}',
])
def test_guard_catches_the_old_name(source):
    assert old_name_uses(source)


@pytest.mark.parametrize('source', [
    'subject.participate(session)',
    '"""Benchmarks only interact with models through the interface."""',
    '"two agents interacting"; x = "interactive"',
    'def drive(subject, session, driver: str = "interact"): pass',
    'def drive(subject, *, driver="interact_streaming"): pass',
    'StreamEvent("motor", 1, None, meta={"driver": "interact", "step_num": 0})',
    'drive_neural_session(subject, session, driver="interact")',
])
def test_guard_allows_participate_and_ordinary_words(source):
    assert not old_name_uses(source)
