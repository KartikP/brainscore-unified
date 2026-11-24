"""Enforcement check E3: benchmarks, tools and adapters use no private model code.

The folders ``models`` and ``validation`` get the import signal only: a file there
that reaches a private module or private name of another part fails (a model
plugin is what an outside author copies, so it uses public helpers).

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

Rule 2.2 in core and in the library code of unified (all of ``brainscore_core``; in
``brainscore``, every file and folder but the model plugin folders): no ability is
discovered by a read of a private name with ``getattr`` or ``hasattr`` and a text
argument on another object than ``self`` or ``cls``, or by ``isinstance`` against a
subject class or a part class (a subclass of ``Subject``, ``CachingExtractor`` or
``Capability`` defined in core or unified; ``Subject`` itself is the contract type and
is allowed; classes are matched by module and class name, so a plugin class with
the name of a core class is another class). These places have their own short list, ``DISCOVERY_ALLOWED``.

Cannot detect: private access through a variable with another name than the three
listed, or through a public helper that forwards to private code; imports through
``importlib``. A second call of
the same method on the same receiver inside a file that is already
allow-listed is not reported (findings are per file).

The allow-list may only shrink: an entry that no longer violates fails.
"""
import ast
from pathlib import Path

import pytest

from tests.architecture_support import (compare, imports, is_private, module_name,
                                        package_root, python_files)

FOLDERS = ('benchmarks', 'tools', 'metrics', 'data', 'experiments', 'harnesses')
MODEL_PRIVATE = ('brainscore.capabilities', 'brainscore.brainscore_model',
                 'brainscore_core.legacy_facade')
SUBJECT_NAMES = {'subject', 'model', 'candidate'}
TYPED_METHODS = {'process', 'start_recording', 'start_task'}

# (path inside brainscore, finding): reason. The target is empty.
ALLOWED = {}

IMPORT_ONLY_FOLDERS = ('models', 'validation')

# Rule 2.2 places: all of core; all of unified but the model plugin folders.
DISCOVERY_PACKAGES = ('brainscore_core', 'brainscore')
PLUGIN_FOLDERS = ('models',)
PART_BASES = {'brainscore_core.contract.Subject',
              'brainscore.model_helpers.extractors.CachingExtractor',
              'brainscore.capabilities.base.Capability'}

# (package/path, finding): reason. Each is a stated place; a new place fails.
_HINT = ('N/A reason text only: the hint reads the wiring of the model class; a hint from '
         'declarations needs a new public name (owner decision, audit 2026-11-22 finding 27)')
_FACTS = ('provenance facts of a caching wrapper (network, batch size, token limit); public '
          'names would be new (open, finding 29); no ability is chosen by it')
_DEVICE = ('the cache key of the vision PytorchWrapper (the one caller, pytorch.py) reads the '
           'input device of the wrapper; a public name would be new there (open, finding 29)')
_HF_ATTENTION = ('a fact for the parity record from a Hugging Face config; transformers has no '
                 'public name for the attention implementation; no subject is read')
_PLACEMENTS = ('the integration runner keeps its own record on the pytest config and reads it '
               'back; the config of the offline test is a SimpleNamespace (no weak key, no '
               'stash), so a move needs a new public name or a private read in the test')
DISCOVERY_ALLOWED = {
    ('brainscore/__init__.py', 'class check BrainScoreModel'): _HINT,
    ('brainscore/model_helpers/provenance.py', 'class check CachingExtractor'): _FACTS,
    ('brainscore/model_helpers/provenance.py', 'getattr obj._model'): _FACTS,
    ('brainscore/model_helpers/provenance.py', 'getattr obj._batch_size'): _FACTS,
    ('brainscore/model_helpers/provenance.py', 'getattr obj._max_length'): _FACTS,
    ('brainscore_core/extraction_cache.py', 'getattr wrapper._device'): _DEVICE,
    ('brainscore/validation/benchmark_parity.py', 'class check BrainScoreModel'):
        'the parity tool refuses a subject that does not match its route (native means the '
        'model class); it chooses no ability',
    ('brainscore/validation/benchmark_parity.py',
     "getattr getattr(net, 'config', None)._attn_implementation"): _HF_ATTENTION,
    ('brainscore/validation/integration_checks.py', 'getattr self.config._umi_model_placements'):
        _PLACEMENTS,
    ('brainscore/validation/integration_checks.py', 'getattr config._umi_model_placements'):
        _PLACEMENTS,
    ('brainscore_core/memory.py', 'hasattr benchmark._assembly'):
        'open legacy guess (finding 28), not a rule exception: with no stated metric category, '
        'the memory estimate guesses neural or behavioral from an _assembly; estimate only',
}


def findings(source, module, imports_only=False):
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
    if imports_only:
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in SUBJECT_NAMES and is_private(node.attr):
            found.add(f'read {node.value.id}.{node.attr}')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in TYPED_METHODS:
            found.add(f'call {ast.unparse(node.func.value)}.{node.func.attr}')
    return found


def violations(root):
    found = {(path, finding)
             for path, source in python_files(root, FOLDERS)
             for finding in findings(source, module_name('brainscore', path))}
    found |= {(path, finding)
              for path, source in python_files(root, IMPORT_ONLY_FOLDERS)
              for finding in findings(source, module_name('brainscore', path), imports_only=True)}
    return found


def file_names(tree, module):
    """Name in one file -> the qualified name it stands for: an import (relative ones resolved,
    ``as`` names followed) or a class defined in the file. ``module`` is the dotted file name."""
    parts = module.split('.')
    own = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
    names, classes = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                first = alias.name.split('.')[0]
                names[alias.asname or first] = alias.name if alias.asname else first
        elif isinstance(node, ast.ImportFrom):
            source = node.module or ''
            if node.level:
                base = parts[:len(parts) - node.level]
                source = '.'.join(base + ([node.module] if node.module else []))
            for alias in node.names:
                names[alias.asname or alias.name] = f'{source}.{alias.name}'
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = f'{own}.{node.name}'
    return {**names, **classes}, own


def qualified(node, names, own, tables):
    """Qualified name of a name or dotted name, followed through re-exports in ``tables``."""
    text = ast.unparse(node)
    first, _, rest = text.partition('.')
    name = names.get(first, f'{own}.{first}') + (f'.{rest}' if rest else '')
    for _ in range(20):
        module, _, last = name.rpartition('.')
        target = tables.get(module, {}).get(last, name)
        if target == name:
            break
        name = target
    return name


def part_classes(roots):
    """Qualified names (module and class) of the classes in ``roots`` (package -> folder) that
    derive from a part base, the bases' own subclasses included; ``Subject`` itself is left
    out (the contract type). Also returns the name tables of the files read."""
    tables, bases = {}, {}
    files = [(module_name(package, path), ast.parse(source))
             for package, root in roots.items() for path, source in python_files(root)]
    for module, tree in files:
        names, own = file_names(tree, module)
        tables[own] = names
    for module, tree in files:
        own = file_names(tree, module)[1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases.setdefault(f'{own}.{node.name}', set()).update(
                    qualified(base, tables[own], own, tables) for base in node.bases
                    if isinstance(base, (ast.Name, ast.Attribute)))
    found = set(PART_BASES)
    while new := {name for name, parents in bases.items() if parents & found} - found:
        found |= new
    return found - {'brainscore_core.contract.Subject'}, tables


def class_nodes(node):
    """Class expressions in a class argument: a name, a dotted name, a tuple or a ``|``."""
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [kind for element in node.elts for kind in class_nodes(element)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return class_nodes(node.left) + class_nodes(node.right)
    if isinstance(node, ast.Starred):
        return class_nodes(node.value)
    return [node] if isinstance(node, (ast.Name, ast.Attribute)) else []


def is_type_of(node):
    """``type(x)`` or ``x.__class__``."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'type' and len(node.args) == 1) or (
        isinstance(node, ast.Attribute) and node.attr == '__class__')


def discovery_findings(source, classes, module='brainscore.example', tables=None):
    """Rule 2.2 findings of one file: a private read by text, a class check of a part
    (``isinstance``, ``issubclass``, ``type(x)`` or ``x.__class__`` compared with a class).
    ``classes`` are qualified names; a class argument is resolved through the file's imports."""
    tree = ast.parse(source)
    names, own = file_names(tree, module)

    def checked(node):
        return {f"class check {name.rpartition('.')[2]}" for kind in class_nodes(node)
                if (name := qualified(kind, names, own, tables or {})) in classes}

    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(is_type_of(operand) for operand in operands):
                found.update(*(checked(operand) for operand in operands))
            continue
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and len(node.args) >= 2):
            continue
        target, what = node.args[:2]
        if node.func.id in ('getattr', 'hasattr') and isinstance(what, ast.Constant) \
                and isinstance(what.value, str) and is_private(what.value) \
                and not (isinstance(target, ast.Name) and target.id in ('self', 'cls')):
            found.add(f'{node.func.id} {ast.unparse(target)}.{what.value}')
        if node.func.id in ('isinstance', 'issubclass'):
            found |= checked(what)
    return found


def discovery_places(package, root):
    """Folders and files read for rule 2.2. A model plugin folder stays out: it is an author's
    plugin, read by the import signal above; the files beside the plugin folders
    (``models/_shared.py``) are library code and are read. A benchmark plugin is read."""
    if package != 'brainscore':
        return ('',)
    places = []
    for path in sorted(Path(root).iterdir()):
        if path.name in PLUGIN_FOLDERS:
            places += [f'{path.name}/{file.name}' for file in sorted(path.glob('*.py'))]
        elif path.suffix == '.py' or (path.is_dir() and path.name != '__pycache__'):
            places.append(path.name)
    return places


def discovery_violations(roots, classes, tables=None):
    return {(f'{package}/{path}', finding)
            for package, root in roots.items()
            for path, source in python_files(root, discovery_places(package, root))
            for finding in discovery_findings(source, classes, module_name(package, path),
                                              tables)}


def test_no_ability_is_discovered_by_private_read_or_class_check_in_core_and_helpers():
    """Rule 2.2 in core and in the unified library code.

    Limit: a source scan cannot be complete, and this one does not see a class held in a
    variable, ``getattr`` or ``hasattr`` with a name held in a variable, a private attribute
    read with a dot, signature inspection, or a class of vision or language.
    """
    roots = {package: package_root(package) for package in DISCOVERY_PACKAGES}
    compare(discovery_violations(roots, *part_classes(roots)), DISCOVERY_ALLOWED,
            'discovery by private read or class check (rule 2.2)')


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


IMPORTS = ('from brainscore import BrainScoreModel\n'
           'from brainscore.model_helpers import wrappers\n'
           'from brainscore.model_helpers.extractors import CachingExtractor\n')
CLASSES = {'brainscore.BrainScoreModel', 'brainscore.model_helpers.wrappers.TextWrapper',
           'brainscore.model_helpers.extractors.CachingExtractor'}


@pytest.mark.parametrize('source', [
    "getattr(wrapper, '_device', None)",
    "hasattr(benchmark, '_assembly')",
    "getattr(self.extractor, '_model')",
    'isinstance(subject, BrainScoreModel)',
    'isinstance(part, (str, wrappers.TextWrapper))',
    'isinstance(subject, BrainScoreModel | str)',
    'issubclass(type(part), CachingExtractor)',
    'type(subject) is BrainScoreModel',
    'type(subject) != BrainScoreModel',
    'subject.__class__ is BrainScoreModel',
    'type(part) in (str, wrappers.TextWrapper)',
])
def test_discovery_signal_catches(source):
    assert discovery_findings(IMPORTS + source, CLASSES)


@pytest.mark.parametrize('source', [
    "getattr(self, '_cache', None)",
    "getattr(subject, 'identifier', None)",
    "getattr(module, '__name__')",
    'isinstance(event, StreamEvent)',
    'isinstance(subject, Subject)',
    'type(event) is StreamEvent',
    'subject.identifier == BrainScoreModel',
])
def test_discovery_signal_allows(source):
    assert not discovery_findings(IMPORTS + source, CLASSES)


def test_part_classes_follow_subclasses(tmp_path):
    _tree(tmp_path, {'a.py': 'from brainscore_core.contract import Subject as S\n'
                             'from .model_helpers.extractors import CachingExtractor\n'
                             'class Wrap(CachingExtractor): pass\nclass Deep(Wrap): pass\n'
                             'class Agent(S): pass\nclass Other: pass\n'})
    classes, _ = part_classes({'brainscore': tmp_path})
    assert {'brainscore.a.Wrap', 'brainscore.a.Deep', 'brainscore.a.Agent'} <= classes
    assert not {'brainscore.a.Other', 'brainscore_core.contract.Subject'} & classes


def test_a_plugin_class_with_the_name_of_a_core_class_does_not_fail(tmp_path):
    roots = {'brainscore_core': tmp_path / 'brainscore_core', 'brainscore': tmp_path / 'brainscore'}
    _tree(tmp_path, {
        'brainscore_core/contract.py': 'class Subject: pass\n',
        'brainscore_core/streaming.py': 'class Session: pass\n',
        'brainscore_core/check.py': 'from .streaming import Session\nok = isinstance(x, Session)\n',
        'brainscore/models/plug/model.py': ('from brainscore_core.contract import Subject\n'
                                            'class Session(Subject): pass\n'),
        'brainscore/tool.py': ('from brainscore.models.plug.model import Session as Plug\n'
                               'ok = isinstance(x, Plug)\n')})
    assert discovery_violations(roots, *part_classes(roots)) == {
        ('brainscore/tool.py', 'class check Session')}


def test_discovery_reads_library_folders_and_not_plugin_folders(tmp_path):
    root = _tree(tmp_path, {'harnesses/new.py': IMPORTS + 'isinstance(subject, BrainScoreModel)\n',
                            'validation/check.py': "getattr(config, '_marker', [])\n",
                            'top.py': "hasattr(benchmark, '_assembly')\n",
                            'benchmarks/_score.py': "getattr(model, '_extractors')\n",
                            'benchmarks/plugin/benchmark.py':
                                IMPORTS + 'isinstance(subject, BrainScoreModel)\n',
                            'models/plugin/model.py':
                                IMPORTS + 'isinstance(subject, BrainScoreModel)\n'})
    found = {path for path, _ in discovery_violations({'brainscore': root}, CLASSES)}
    assert {'brainscore/harnesses/new.py', 'brainscore/validation/check.py',
            'brainscore/top.py', 'brainscore/benchmarks/_score.py',
            'brainscore/benchmarks/plugin/benchmark.py'} <= found
    assert 'brainscore/models/plugin/model.py' not in found


def test_discovery_check_fails_on_a_new_place(tmp_path):
    root = _tree(tmp_path, {'new.py': IMPORTS + 'isinstance(subject, BrainScoreModel)\n'})
    with pytest.raises(AssertionError, match='new discovery'):
        compare(discovery_violations({'brainscore_core': root}, CLASSES), {}, 'discovery')


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
