"""Enforcement check E4: one definition site for each exclusive responsibility.

Guideline "one owner for each responsibility"; Architecture and Ownership Map,
sections 3, 3a and 6 (E4).

Reads (with ``ast``, never imported) the library code of core, unified, vision
and language. Tests are not read. The legacy plugin folders of vision and
language (``models``, ``benchmarks``, ``data``, ``metrics``) are not read:
legacy plugins are permanent and own nothing of the contract.

A static check sees names, not meaning. For each responsibility it checks one
concrete signal, listed with what it cannot detect:

R1 channel matching. Signal: a function named ``channel_declared`` is defined.
    Cannot detect: channel names compared inline with ``==`` or ``in``.
R2 declaration validation. Signal: a function named
    ``check_channel_compatibility``, ``check_compatibility`` or
    ``form_problem`` is defined. Cannot detect: the same rules checked under
    another name (the early ``required_channels`` check of ``BrainScoreModel``
    is such a check; it is an accepted exception).
R4 the N/A decision. Signal: ``na_score`` is defined or called. Cannot detect:
    an N/A score built without ``na_score``.
R6 one ``Session`` class. Signal: a class lists a base named ``Session``.
    Cannot detect: a class that copies the session behaviour without the base.
R7 response validation. Signal: ``check_event`` (the catalog check of one
    event) is called. Cannot detect: a payload checked by other code.
R9 extractor choice for a region. Signal: a function named ``extractor_keys``
    or ``region_extractor_keys`` is defined. Cannot detect: an extractor
    picked inline from a mapping.
R11 hooks on a network. Signal: a call of ``register_forward_hook``,
    ``register_forward_pre_hook`` or ``register_full_backward_hook``. Cannot
    detect: hooks installed through a third-party helper.
R12 intervention apply and undo. Signal: an in-place write to a tensor's
    ``.data`` (``.data.copy_``, ``.data.index_copy_``, ``.data.mul_``,
    ``.data.add_``, ``.data.zero_``, ``.data.fill_``) in core or unified
    outside the model plugins (weight initialisation of a null model is not an
    intervention). Cannot detect: interventions made with hooks (R11 reports
    those, for example the ``Ablate`` tool family) or with ``torch.no_grad``
    assignments to parameters.
R14 regression for encoding scores. Signal: an import from
    ``sklearn.linear_model`` or ``sklearn.cross_decomposition``. Cannot detect:
    a regression written with numpy, or a fit through a third package.
R15 score construction. Signal: a call of ``Score(`` in unified
    ``benchmarks``, ``metrics`` or ``harnesses``. Cannot detect: a score built
    by a legacy helper (``explained_variance``, ``ceiling_normalize`` in the
    ``-unified`` MajajHong and Pereira pair) or by ``Score.__new__``.
R17 score metadata stamp. Signal: ``stamp_score_metadata`` is defined or
    called. Cannot detect: score attributes written directly.
R18 activation cache validity. Signal: a write to the
    ``RESULTCACHING_DISABLE`` variable of ``os.environ`` (not an environment
    built for a child process), or a definition of
    ``extraction_fingerprint`` or ``weight_fingerprint_scope``. Cannot detect:
    a cache key built under another name.
R20 registry assignment. Signal: a module-level name ending in ``_registry``
    is assigned, or a module-level function ``register_*`` is defined. Cannot
    detect: a registry held under another name.

Each finding is one (responsibility, file). The allow-list holds the second
sites of today with a reason; it may only shrink: an entry that no longer
violates fails the test.
"""
import ast

import pytest

from tests.architecture_support import compare, package_root, python_files

PACKAGES = {'core': 'brainscore_core', 'unified': 'brainscore',
            'vision': 'brainscore_vision', 'language': 'brainscore_language'}
LEGACY_PLUGIN_FOLDERS = ('models/', 'benchmarks/', 'data/', 'metrics/')
ENTRY_POINTS = {'unified:__init__.py', 'vision:__init__.py', 'language:__init__.py'}


def _calls(tree, names):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
            if name in names:
                return True
    return False


def _defines(tree, names, module_level=False):
    nodes = tree.body if module_level else ast.walk(tree)
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
               for node in nodes)


def _subclasses_session(tree):
    return any(isinstance(node, ast.ClassDef) and
               any(ast.unparse(base).split('.')[-1] == 'Session' for base in node.bases)
               for node in ast.walk(tree))


_IN_PLACE = {'copy_', 'index_copy_', 'mul_', 'add_', 'zero_', 'fill_'}


def _writes_tensor_data(tree):
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr in _IN_PLACE and isinstance(node.func.value, ast.Attribute)
               and node.func.value.attr == 'data'
               for node in ast.walk(tree))


def _imports_sklearn_regression(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module] + [f'{node.module}.{alias.name}' for alias in node.names]
        else:
            continue
        if any(module.startswith(('sklearn.linear_model', 'sklearn.cross_decomposition'))
               for module in modules):
            return True
    return False


def _writes_cache_switch(tree):
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ('pop', 'setdefault', 'update', 'putenv', 'unsetenv'):
            targets = [node]
        # ponytail: text match on the unparsed target; a renamed alias of os.environ is not seen
        if any('RESULTCACHING_DISABLE' in text and 'environ' in text
               for text in map(ast.unparse, targets)):
            return True
    return _defines(tree, {'extraction_fingerprint', 'weight_fingerprint_scope'})


def _defines_registry(tree):
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else \
            [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id.endswith('_registry') for target in targets):
            return True
    return any(isinstance(node, ast.FunctionDef) and node.name.startswith('register_')
               for node in tree.body)


def _everywhere(site):
    return True


def _core_and_unified_library(site):
    return site.startswith('core:') or (site.startswith('unified:')
                                        and not site.startswith('unified:models/'))


def _unified_score_code(site):
    return site.startswith(('unified:benchmarks/', 'unified:metrics/', 'unified:harnesses/'))


# id: (signal, scope, allowed definition sites)
RESPONSIBILITIES = {
    'R1 channel matching': (lambda tree: _defines(tree, {'channel_declared'}), _everywhere,
                            {'core:streaming.py'}),
    'R2 declaration validation': (
        lambda tree: _defines(tree, {'check_channel_compatibility', 'check_compatibility',
                                     'form_problem'}),
        _everywhere, {'core:compatibility.py', 'core:io_catalog.py'}),
    'R4 N/A decision': (lambda tree: _defines(tree, {'na_score'}) or _calls(tree, {'na_score'}),
                        _everywhere, {'core:score_status.py'} | ENTRY_POINTS),
    'R6 one Session class': (_subclasses_session, _everywhere, set()),
    'R7 response validation': (lambda tree: _calls(tree, {'check_event'}), _everywhere,
                               {'core:streaming.py'}),
    'R9 extractor choice': (lambda tree: _defines(tree, {'extractor_keys', 'region_extractor_keys'}),
                            _everywhere, {'unified:_model/dispatch.py', 'unified:_model/recording.py'}),
    'R11 hooks on a network': (
        lambda tree: _calls(tree, {'register_forward_hook', 'register_forward_pre_hook',
                                   'register_full_backward_hook'}),
        _everywhere, {'unified:model_helpers/_extractors.py', 'unified:perturbation.py'}),
    'R12 intervention apply and undo': (_writes_tensor_data, _core_and_unified_library,
                                        {'unified:perturbation.py'}),
    'R14 encoding regression': (_imports_sklearn_regression, _everywhere,
                                {'unified:metrics/encoding.py', 'unified:_model/behavior.py'}),
    'R15 score construction': (lambda tree: _calls(tree, {'Score'}), _unified_score_code,
                               {'unified:benchmarks/_score.py'}),
    'R17 metadata stamp': (
        lambda tree: _defines(tree, {'stamp_score_metadata'}) or _calls(tree, {'stamp_score_metadata'}),
        _everywhere, {'core:score_metadata.py'} | ENTRY_POINTS),
    'R18 cache validity': (_writes_cache_switch, _everywhere, {'core:extraction_cache.py'}),
    'R20 registry assignment': (_defines_registry, _everywhere,
                                {'unified:__init__.py', 'core:io_catalog.py'}),
}

_HOOK_TOOL = 'an observer or tool, not a score path; a second way to attach hooks (second audit P5)'
_LEGACY = 'legacy Brain-Score 2.0 helper, permanent'
_NO_CEILING = '"no ceiling" or a benchmark ceiling built as Score(...) beside benchmark_score (project record decision 29)'
# (responsibility, site): reason
ALLOWED = {
    ('R11 hooks on a network', 'unified:activation_window.py'): _HOOK_TOOL,
    ('R11 hooks on a network', 'unified:percept_window.py'): _HOOK_TOOL,
    ('R11 hooks on a network', 'unified:experiments/tools.py'):
        'TorchInstrumentation: the Ablate tool family intervenes through its own hooks (R12, decision 14)',
    ('R11 hooks on a network', 'vision:model_helpers/activations/pytorch.py'): _LEGACY,
    ('R11 hooks on a network', 'vision:model_helpers/activations/temporal/model/pytorch.py'): _LEGACY,
    ('R11 hooks on a network', 'language:model_helpers/huggingface.py'): _LEGACY,
    ('R11 hooks on a network', 'language:model_helpers/localize.py'): _LEGACY,
    ('R14 encoding regression', 'unified:tools/layer_mapping.py'):
        'layer mapping fits its own Ridge/RidgeCV (second audit)',
    ('R14 encoding regression', 'unified:tools/linear_probe.py'):
        'the linear probe tool fits its own sklearn model',
    ('R14 encoding regression', 'vision:model_helpers/brain_transformation/behavior.py'): _LEGACY,
    ('R15 score construction', 'unified:metrics/direct_comparison/metric.py'):
        'registered metric builds Score directly (second score builder)',
    ('R15 score construction', 'unified:metrics/topographic/metric.py'):
        'registered metric builds Score directly (second score builder)',
    ('R15 score construction', 'unified:benchmarks/grid_game/benchmark.py'): _NO_CEILING,
    ('R15 score construction', 'unified:benchmarks/lahner2024/benchmark.py'): _NO_CEILING,
    ('R15 score construction', 'unified:benchmarks/lahner2024/benchmark_timeresolved.py'): _NO_CEILING,
    ('R15 score construction', 'unified:benchmarks/roar_yeatman2021/benchmark.py'): _NO_CEILING,
    ('R15 score construction', 'unified:harnesses/human_replay.py'): _NO_CEILING,
    ('R18 cache validity', 'vision:model_helpers/check_submission/check_models.py'): _LEGACY,
    ('R20 registry assignment', 'vision:__init__.py'): 'domain registries of vision, permanent',
    ('R20 registry assignment', 'language:__init__.py'): 'domain registries of language, permanent',
    ('R20 registry assignment', 'unified:tools/_registry.py'):
        'register_tool: a second registration style (R20)',
    ('R20 registry assignment', 'unified:capabilities/registry.py'):
        'register_capability and capability_registry: a second registration style (R20)',
    ('R20 registry assignment', 'unified:data/preparation.py'):
        'data_builder_registry: a sixth registry, not named in the Ownership Map',
}


def _library_files():
    for repo, package in PACKAGES.items():
        for path, source in python_files(package_root(package)):
            if repo in ('vision', 'language') and path.startswith(LEGACY_PLUGIN_FOLDERS):
                continue
            yield f'{repo}:{path}', source


def violations(files, responsibilities=RESPONSIBILITIES):
    found = set()
    for site, source in files:
        tree = ast.parse(source)
        for name, (signal, scope, allowed) in responsibilities.items():
            if site not in allowed and scope(site) and signal(tree):
                found.add((name, site))
    return found


def test_each_exclusive_responsibility_has_one_definition_site():
    compare(violations(_library_files()), ALLOWED, 'second definition sites')


def test_every_allowed_site_still_holds_its_signal():
    """An owner that loses its signal makes the check blind for that responsibility."""
    files = dict(_library_files())
    missing = [(name, site) for name, (signal, scope, allowed) in RESPONSIBILITIES.items()
               for site in allowed if site not in files or not signal(ast.parse(files[site]))]
    assert not missing, f'owner sites without their signal: {missing}'


def test_check_fails_on_a_new_violation():
    files = [('unified:tools/new_tool.py', 'from sklearn.linear_model import Ridge\n'),
             ('unified:benchmarks/new/benchmark.py', 'score = Score(0.5)\n'),
             ('core:new.py', 'class Mine(Session):\n    pass\n')]
    with pytest.raises(AssertionError, match='new second definition sites'):
        compare(violations(files), ALLOWED, 'second definition sites')
    assert violations(files) == {('R14 encoding regression', 'unified:tools/new_tool.py'),
                                 ('R15 score construction', 'unified:benchmarks/new/benchmark.py'),
                                 ('R6 one Session class', 'core:new.py')}


def test_check_fails_on_a_stale_entry():
    files = [('unified:tools/layer_mapping.py', 'import numpy\n')]
    with pytest.raises(AssertionError, match='no longer violate'):
        compare(violations(files), {('R14 encoding regression', 'unified:tools/layer_mapping.py'): 'x'},
                'second definition sites')


@pytest.mark.parametrize('name, source', [
    ('R1 channel matching', 'def channel_declared(a, b): ...'),
    ('R2 declaration validation', 'def check_compatibility(b, s): ...'),
    ('R4 N/A decision', 'na_score(benchmark, reason)'),
    ('R7 response validation', 'check_event(event, OUTPUT)'),
    ('R9 extractor choice', 'def extractor_keys(m, r): ...'),
    ('R11 hooks on a network', 'module.register_forward_hook(f)'),
    ('R12 intervention apply and undo', 'parameter.data.copy_(original)'),
    ('R17 metadata stamp', 'stamp_score_metadata(score, b, s)'),
    ('R18 cache validity', "os.environ['RESULTCACHING_DISABLE'] = '1'"),
    ('R18 cache validity', "os.environ.pop('RESULTCACHING_DISABLE', None)"),
    ('R20 registry assignment', 'model_registry = {}'),
    ('R20 registry assignment', 'def register_widget(name, w): ...'),
])
def test_signal_catches(name, source):
    signal = RESPONSIBILITIES[name][0]
    assert signal(ast.parse(source))


@pytest.mark.parametrize('name, source', [
    ('R4 N/A decision', 'from brainscore_core.score_status import is_na'),
    ('R12 intervention apply and undo', 'result.data = cast'),
    ('R18 cache validity', "os.environ.get('RESULTCACHING_DISABLE')"),
    ('R18 cache validity', "env.update(RESULTCACHING_DISABLE='1')"),
    ('R20 registry assignment', "model_registry['x'] = build"),
    ('R20 registry assignment', 'class Hooks:\n    def register_hook(self): ...'),
])
def test_signal_allows(name, source):
    signal = RESPONSIBILITIES[name][0]
    assert not signal(ast.parse(source))
