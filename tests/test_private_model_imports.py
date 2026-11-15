"""Enforcement check E3: benchmarks, tools and adapters use no private model code.

Contract rules 2.2 and 7.4; House Style pattern 2; Architecture and Ownership
Map, section 6, E3.

Reads (with ``ast``, never imported) every file under ``brainscore/`` folders
``benchmarks``, ``tools``, ``metrics``, ``data``, ``experiments`` and
``harnesses``. Signals:

- an import of ``brainscore._model`` or a module below it;
- an import of a private name (leading underscore) from
  ``brainscore.capabilities``, ``brainscore.brainscore_model`` or
  ``brainscore_core.legacy_facade``;
- an import of a private module or private name of another part of the
  package: a file in ``brainscore/<part>/`` that reaches a private module or
  name in ``brainscore.<other>`` (private names inside the own part are the
  part's own business);
- an attribute read ``subject._x``, ``model._x`` or ``candidate._x``;
- a call of ``.process``, ``.start_recording`` or ``.start_task`` on any
  object (the typed calling convention, House Style pattern 2).

Cannot detect: private access through ``getattr`` with a string, through a
variable with another name than the three listed, or through a public helper
that forwards to private code; imports through ``importlib``. A second call of
the same method on the same receiver inside a file that is already
allow-listed is not reported (findings are per file).

The allow-list may only shrink: an entry that no longer violates fails.
"""
import ast

import pytest

from tests.architecture_support import (compare, imports, is_private, module_name,
                                        package_root, python_files)

FOLDERS = ('benchmarks', 'tools', 'metrics', 'data', 'experiments', 'harnesses')
MODEL_PRIVATE = ('brainscore.capabilities', 'brainscore.brainscore_model',
                 'brainscore_core.legacy_facade')
SUBJECT_NAMES = {'subject', 'model', 'candidate'}
TYPED_METHODS = {'process', 'start_recording', 'start_task'}

_TYPED = 'the typed calling convention of the -unified MajajHong/Pereira pair (R21); needs a parity run to change'
# (path inside brainscore, finding): reason
ALLOWED = {
    ('experiments/task.py', 'import brainscore._model.intervention'):
        'the Intervene tool reads INTERVENTION_CHANNELS; named in the Ownership Map E3',
    ('benchmarks/majajhong2015/benchmark.py', 'call candidate.start_recording'): _TYPED,
    ('benchmarks/majajhong2015/benchmark.py', 'call candidate.process'): _TYPED,
    ('benchmarks/pereira2018/benchmark.py', 'call candidate.start_recording'): _TYPED,
    ('benchmarks/pereira2018/benchmark.py', 'call candidate.process'): _TYPED,
    ('experiments/tools.py', 'private brainscore.activation_window._first_tensor_out'):
        'TorchInstrumentation shares the tensor unwrapping of ActivationWindow',
}


def findings(source, module):
    """Set of E3 findings for one file whose dotted name is ``module``."""
    tree = ast.parse(source)
    own_part = module.split('.')[1]
    found = set()
    for imported, names in imports(tree, module):
        parts = imported.split('.')
        if parts[:2] == ['brainscore', '_model']:
            found.add(f'import {imported}')
            continue
        private_names = [name for name in names if is_private(name)]
        if imported in MODEL_PRIVATE:
            found.update(f'private {imported}.{name}' for name in private_names)
        if parts[0] == 'brainscore' and len(parts) > 1 and parts[1] != own_part:
            depth = next((i for i, part in enumerate(parts) if is_private(part)), None)
            if depth is not None:
                found.add(f"private {'.'.join(parts[:depth + 1])}")
            else:
                found.update(f'private {imported}.{name}' for name in private_names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in SUBJECT_NAMES and is_private(node.attr):
            found.add(f'read {node.value.id}.{node.attr}')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in TYPED_METHODS:
            found.add(f'call {ast.unparse(node.func.value)}.{node.func.attr}')
    return found


def violations(root):
    return {(path, finding)
            for path, source in python_files(root, FOLDERS)
            for finding in findings(source, module_name('brainscore', path))}


def test_no_private_model_code_in_benchmarks_tools_and_adapters():
    compare(violations(package_root('brainscore')), ALLOWED, 'private model access')


def _tree(tmp_path, files):
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    return tmp_path


def test_check_fails_on_a_new_violation(tmp_path):
    root = _tree(tmp_path, {'tools/new.py': 'from brainscore._model.recording import x\n'})
    with pytest.raises(AssertionError, match='new private model access'):
        compare(violations(root), {}, 'private model access')


def test_check_fails_on_a_stale_entry(tmp_path):
    root = _tree(tmp_path, {'tools/clean.py': 'from brainscore import BrainScoreModel\n'})
    with pytest.raises(AssertionError, match='no longer violate'):
        compare(violations(root), {('tools/clean.py', 'call candidate.process'): 'stale'},
                'private model access')


@pytest.mark.parametrize('source', [
    'from brainscore._model.intervention import INTERVENTION_CHANNELS',
    'import brainscore._model',
    'from brainscore.capabilities import _claim',
    'from brainscore_core.legacy_facade import _resolve',
    'from brainscore.model_helpers._extractors import LiveRecording',
    'from brainscore.harnesses.coggym import _PROMPT_NOTE',
    'from ...harnesses import _private_helper',
    'subject._extractors',
    'candidate.start_recording("IT")',
    'self.subject.process(stimuli)',
])
def test_signal_catches(source):
    assert findings(source, 'brainscore.benchmarks.example.benchmark')


@pytest.mark.parametrize('source', [
    'from brainscore import BrainScoreModel',
    'from brainscore.benchmarks._score import benchmark_score',
    'from ._util import helper',
    'from .benchmark import _build',
    'from brainscore_core.legacy_facade import subject_for_benchmark',
    'self._cache',
    'session.respond(event)',
])
def test_signal_allows(source):
    assert not findings(source, 'brainscore.benchmarks.example.benchmark')
