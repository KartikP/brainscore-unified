"""The concrete subject and its extensions have one implementation owner."""
import ast
import json
import pickle
from pathlib import Path

import pytest

from brainscore import BrainScoreModel, LayerSelector
from brainscore._model.behavior import ProbabilitiesClassifier
from brainscore._model.intervention import Interventions

pytestmark = pytest.mark.unit

MOVED_MODULES = {
    'brainscore_model', 'selection', 'extractor', 'capabilities', '_model',
    'behavior', 'behavioral', 'capability_config', 'dispatch', 'perturbation', 'recording',
}
MOVED_NAMES = {
    'BrainScoreModel', 'UnitSelector', 'LayerSelector', 'CompositeSelector',
    'UnitSelection', 'FunctionalSelection', 'RandomSelection', 'IndexSelection',
    'selection_unit_indices', 'selection_coordinates', 'Extractor',
    'Capability', 'register_capability', 'capability_registry',
}


def stale_imports(source):
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            parts = (node.module or '').split('.')
            if parts[0] != 'brainscore_core':
                continue
            if (len(parts) > 1 and parts[1] in MOVED_MODULES
                    or any(a.name in MOVED_NAMES | MOVED_MODULES for a in node.names)):
                found.append(node.lineno)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split('.')
                if len(parts) > 1 and parts[0] == 'brainscore_core' and parts[1] in MOVED_MODULES:
                    found.append(node.lineno)
    return found


def test_maintained_sources_use_the_unified_implementation():
    root = Path(__file__).resolve().parents[1]
    failures = []
    for folder in ('brainscore', 'tests', 'examples', 'templates', 'notebooks'):
        for path in sorted((root / folder).rglob('*')):
            if 'archive' in path.relative_to(root).parts or not path.is_file():
                continue
            if path.suffix == '.py':
                sources = [path.read_text(encoding='utf-8')]
            elif path.suffix == '.ipynb':
                sources = [''.join(c.get('source', [])) for c in json.loads(path.read_text())['cells']
                           if c.get('cell_type') == 'code']
            else:
                continue
            for index, source in enumerate(sources):
                if path.suffix == '.ipynb':
                    # Notebook magic lines are commands, not Python imports.
                    source = '\n'.join(line for line in source.splitlines()
                                       if not line.lstrip().startswith(('%', '!', '?')))
                failures.extend(f'{path}:{index}:{line}' for line in stale_imports(source))
    assert not failures, '\n'.join(failures)


@pytest.mark.parametrize('definition', [ProbabilitiesClassifier, Interventions, LayerSelector])
def test_moved_classes_pickle_to_the_unified_definition(definition):
    assert definition.__module__.startswith('brainscore.')
    assert pickle.loads(pickle.dumps(definition)) is definition


def test_selector_instance_pickle_round_trip():
    selector = LayerSelector('layer', modality='vision')
    restored = pickle.loads(pickle.dumps(selector))
    assert type(restored) is LayerSelector
    assert restored == selector


def test_built_in_capabilities_keep_their_order():
    """One registry routes events and sessions. Event capabilities come first, in
    this order; the session capabilities come after any added capability."""
    from brainscore.capabilities import capability_registry
    order = {identifier: capability.order for identifier, capability in capability_registry.items()}
    assert [identifier for identifier in sorted(order, key=lambda i: (order[i], i))
            if order[identifier] < 10000] == [
        'intervention', 'behavioral-generation', 'behavioral-readout', 'closed-loop',
        'neural-encoding']
    assert {order[i] for i in ('neural-sessions', 'behavior-sessions', 'motor-sessions',
                               'intervention-sessions')} == {10000}


@pytest.mark.parametrize('source', [
    'from brainscore_core import BrainScoreModel',
    'from brainscore_core.model_interface import LayerSelector, Subject',
    'import brainscore_core.capabilities.registry',
    'from brainscore_core import brainscore_model as module',
])
def test_guard_rejects_core_implementation_imports(source):
    assert stale_imports(source)


@pytest.mark.parametrize('source', [
    'from brainscore import BrainScoreModel',
    'from brainscore.selection import LayerSelector',
    'from brainscore_core.model_interface import Subject, Selection',
])
def test_guard_accepts_contract_and_unified_imports(source):
    assert not stale_imports(source)


def test_moved_definitions_report_their_real_module():
    import importlib
    import brainscore.brainscore_model as model_module

    root = Path(model_module.__file__).parent
    paths = [root / name for name in ('brainscore_model.py', 'selection.py', 'extractor.py')]
    paths += list((root / '_model').glob('*.py'))
    paths += list((root / 'capabilities').glob('*.py'))
    for path in paths:
        if path.name == '__init__.py':
            continue
        name = '.'.join(path.relative_to(root).with_suffix('').parts)
        module = importlib.import_module('brainscore.' + name)
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert vars(module)[node.name].__module__ == module.__name__
