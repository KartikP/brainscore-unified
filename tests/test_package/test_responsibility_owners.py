"""Enforcement check E4: one definition site for each exclusive responsibility.

Guideline "one owner for each responsibility"; Architecture and Ownership Map,
sections 3, 3a and 6 (E4).

Reads (with ``ast``, never imported) the library code of core and unified, and
the UMI code of vision and language: the score entry point (``__init__.py``),
the adapters (``compat/``) and vision's ``benchmark_helpers/test_db_contract.py``.
Tests are not read. The rest of vision and language is Brain-Score 2.0 code
(the plugin folders and the library code such as ``model_helpers``): it is
permanent and owns nothing of the contract (Contract Rule 11.2), so it is not read.

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
    ``register_forward_pre_hook`` or ``register_full_backward_hook``. Owners:
    the extractors (recording), ``model_helpers/intervention.py`` (interventions) and
    ``TorchInstrumentation`` (experiment tools, House Style pattern 4). Cannot
    detect: hooks installed through a third-party helper.
R12 intervention apply and undo. Signal: an in-place write to a tensor's
    ``.data`` (``.data.copy_``, ``.data.index_copy_``, ``.data.mul_``,
    ``.data.add_``, ``.data.zero_``, ``.data.fill_``) in core or unified
    outside the model plugins (weight initialisation of a null model is not an
    intervention). Cannot detect: interventions made with hooks (R11 reports
    those, for example the ``Ablate`` tool family) or with ``torch.no_grad``
    assignments to parameters.
R14 regression for encoding scores. Signal: a regression estimator of
    ``sklearn.linear_model`` or ``sklearn.cross_decomposition`` (``REGRESSORS``)
    is imported or named; a classifier (``LogisticRegression``) is not a
    regression. Cannot detect: a regression written with numpy, or a fit
    through a third package.
R15 score construction. Signal: a call of ``Score(`` in unified
    ``benchmarks`` or ``harnesses`` that is not part of a ``ceiling=``
    argument. A metric returns its raw value as a ``Score`` by the core
    ``Metric`` contract, and a ceiling handed to the base class or to
    ``benchmark_score``, or returned by a benchmark's ``ceiling`` property
    (a ceiling measured at first use), is a reference, not a score; neither
    is score construction. Cannot detect: a score built by a legacy helper
    (``explained_variance``, ``ceiling_normalize`` in the ``-unified``
    MajajHong and Pereira pair) or by ``Score.__new__``.
R17 score metadata stamp. Signal: ``stamp_score_metadata`` is defined or
    called. Cannot detect: score attributes written directly.
R18 activation cache validity. Signal: a write to the
    ``RESULTCACHING_DISABLE`` variable of ``os.environ`` (not an environment
    built for a child process), a read of it (``os.environ``, ``os.getenv``, or
    ``is_enabled`` of ``result_caching``), or a definition of
    ``extraction_fingerprint`` or ``weight_fingerprint_scope``. The one reader
    is ``store_enabled``; a read is also looked for in the library code of
    vision (not its plugin folders ``models/``, ``benchmarks/``, ``data/``).
    Cannot detect: a cache key built under another name, or the variable name
    built at run time.
R20 registry assignment. Signal: a module-level name ending in ``_registry``
    is assigned, or a module-level function ``register_*`` is defined. Cannot
    detect: a registry held under another name.

Each finding is one (responsibility, file). ``ALLOWED`` holds the violations
of today with a reason; its target is empty. ``STATED_EXCEPTIONS`` holds the
second sites that a rule states as exceptions, each with the rule sentence.
Both may only shrink: an entry that is no longer found fails the test.
"""
import ast

import pytest

from tests.architecture_support import compare, package_root, python_files

PACKAGES = {'core': 'brainscore_core', 'unified': 'brainscore',
            'vision': 'brainscore_vision', 'language': 'brainscore_language'}
# The UMI code of vision and language; the rest is Brain-Score 2.0 code (Contract Rule 11.2).
UMI_CODE = {'vision': ('__init__.py', 'compat/', 'benchmark_helpers/test_db_contract.py'),
            'language': ('__init__.py', 'compat/')}
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


_REGRESSION_MODULES = ('sklearn.linear_model', 'sklearn.cross_decomposition')
REGRESSORS = {'LinearRegression', 'Ridge', 'RidgeCV', 'Lasso', 'LassoCV', 'LassoLars',
              'ElasticNet', 'ElasticNetCV', 'MultiTaskLasso', 'MultiTaskElasticNet',
              'BayesianRidge', 'HuberRegressor', 'SGDRegressor', 'Lars',
              'PLSRegression', 'PLSCanonical', 'PLSSVD', 'CCA'}


def _module_names(tree):
    """Local name -> dotted module path, for every ``import x[.y] [as z]`` and
    ``from x import y [as z]`` (``y`` may be a module)."""
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    names[alias.asname] = alias.name
                else:
                    root = alias.name.split('.')[0]
                    names[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                names[alias.asname or alias.name] = f'{node.module}.{alias.name}'
    return names


def _dotted(expr, names):
    """The dotted path of ``expr`` with its root name resolved through the imports."""
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if not isinstance(expr, ast.Name):
        return None
    return '.'.join([names.get(expr.id, expr.id)] + parts[::-1])


def _imports_sklearn_regression(tree):
    """A regression estimator imported from, or named through, the two sklearn modules.

    Module aliases are resolved first (``import sklearn.linear_model as lm`` then
    ``lm.Ridge``); a star import from the modules counts, since its names are unknown.
    """
    names = _module_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or '').startswith(_REGRESSION_MODULES):
            if any(alias.name in REGRESSORS or alias.name == '*' for alias in node.names):
                return True
        elif isinstance(node, ast.Attribute) and node.attr in REGRESSORS \
                and (_dotted(node.value, names) or '').endswith(_REGRESSION_MODULES):
            return True
    return False


# The accepted recipients of a ceiling built as Score(...): the score
# builder ``benchmark_score`` and the benchmark base class constructor, reached as
# ``super().__init__`` or ``BenchmarkBase.__init__`` / ``BenchmarkBase(...)``.
CEILING_RECIPIENTS = ('benchmark_score', 'super().__init__', 'BenchmarkBase.__init__',
                      'BenchmarkBase')


def _ceiling_property_returns(cls):
    """The calls a ``ceiling`` property (or cached property) of ``cls`` returns itself, not
    the returns of a function nested in it."""
    for method in cls.body:
        if not (isinstance(method, ast.FunctionDef) and method.name == 'ceiling' and any(
                ast.unparse(decorator).split('.')[-1] in ('property', 'cached_property')
                for decorator in method.decorator_list)):
            continue
        pending = list(method.body)
        while pending:
            statement = pending.pop()
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call):
                yield statement.value
            pending.extend(ast.iter_child_nodes(statement))


def _ceiling_scores(tree):
    """The ``Score(`` calls that are the reference ceiling of an accepted recipient.

    Accepted expressions: the ``ceiling=`` value itself, a branch of a conditional
    expression that is the value (``not_estimated() if c is None else Score(c)``), or
    the value a ``ceiling`` property of a class returns.
    """
    accepted = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            accepted |= {id(value) for value in _ceiling_property_returns(node)}
        if isinstance(node, ast.Call) and any(
                ast.unparse(node.func) in (name, f'self.{name}') or
                ast.unparse(node.func).endswith(f'.{name}') and '(' not in name
                for name in CEILING_RECIPIENTS):
            for keyword in node.keywords:
                if keyword.arg == 'ceiling':
                    value = keyword.value
                    branches = [value.body, value.orelse] if isinstance(value, ast.IfExp) else [value]
                    accepted.update(id(branch) for branch in branches)
    return accepted


def _builds_score(tree):
    """A ``Score(`` call (``Score`` resolved through imports) that is not an accepted ceiling."""
    names = _module_names(tree)
    accepted = _ceiling_scores(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and id(node) not in accepted:
            path = _dotted(node.func, names) or ''
            if path == 'Score' or path.endswith('.Score'):
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
    return _reads_cache_switch(tree) or _defines(tree, {'extraction_fingerprint', 'weight_fingerprint_scope'})


def _reads_cache_switch(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'result_caching' \
                and any(alias.name == 'is_enabled' for alias in node.names):
            return True
        if isinstance(node, ast.Attribute) and node.attr == 'is_enabled' \
                and ast.unparse(node.value).endswith('result_caching'):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ('get', 'getenv') and node.args \
                and ast.unparse(node.args[0]) == "'RESULTCACHING_DISABLE'":
            return True
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) \
                and ast.unparse(node.slice) == "'RESULTCACHING_DISABLE'":
            return True
    return False


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
    return site.startswith(('unified:benchmarks/', 'unified:harnesses/'))


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
        _everywhere, {'unified:model_helpers/extractors.py', 'unified:model_helpers/intervention.py',
                      'unified:experiments/tools.py'}),
    'R12 intervention apply and undo': (_writes_tensor_data, _core_and_unified_library,
                                        {'unified:model_helpers/intervention.py'}),
    'R14 encoding regression': (_imports_sklearn_regression, _everywhere,
                                {'unified:metrics/encoding.py', 'unified:_model/behavior.py'}),
    'R15 score construction': (_builds_score, _unified_score_code,
                               {'unified:benchmarks/_score.py'}),
    'R17 metadata stamp': (
        lambda tree: _defines(tree, {'stamp_score_metadata'}) or _calls(tree, {'stamp_score_metadata'}),
        _everywhere, {'core:score_metadata.py'} | ENTRY_POINTS),
    'R18 cache validity': (_writes_cache_switch, _everywhere, {'core:extraction_cache.py'}),
    'R20 registry assignment': (_defines_registry, _everywhere,
                                {'unified:__init__.py', 'core:io_catalog.py'}),
}

# What the author must do when a rule fails (shown by the failure message).
REMEDIES = {
    'R1 channel matching': 'call channel_declared from brainscore_core.streaming; do not compare channel names yourself',
    'R2 declaration validation': 'let score() and pre-flight check the declarations; do not add a second check',
    'R4 N/A decision': 'do not build an N/A score; declare inputs and outputs, and pre-flight decides',
    'R6 one Session class': 'do not subclass Session; vary the source and give it to Session(source, outputs, task)',
    'R7 response validation': 'respond through session.respond; the session checks each event',
    'R9 extractor choice': 'name the extractor by modality in the extractor= mapping of BrainScoreModel',
    'R11 hooks on a network': 'attach hooks through an extractor or a pytorch_intervention; do not register them yourself',
    'R12 intervention apply and undo': 'write weights through an intervention capability (it applies and undoes the change)',
    'R14 encoding regression': 'fit through the metrics of brainscore.metrics.encoding or load_metric; do not import a regressor',
    'R15 score construction': ('build a ceiling inside the ceiling= argument (or a ceiling property) '
                               'and return the score with benchmark_score; a benchmark builds no Score itself'),
    'R17 metadata stamp': 'score() stamps the metadata; do not stamp a score yourself',
    'R18 cache validity': ('do not switch or read the cache switch yourself; give the extractor a '
                           'cache_config() and ask brainscore_core.extraction_cache.store_enabled'),
    'R20 registry assignment': 'register with a line in your plugin __init__.py (for example model_registry[id] = factory)',
}

# (responsibility, site): reason. Violations only; the target is empty.
ALLOWED = {}

_OBSERVER = ('Ownership Map R11: a user-attached observer (ActivationWindow, PerceptWindow) '
             'installs its own hooks on a network passed to it; it never runs in a score path')
_DOMAIN_REGISTRIES = ('Contract Rule 11.2: the registries of vision and language are '
                      'Brain-Score 2.0 code and permanent; unified loads from them as a fallback')
# (responsibility, site): the rule sentence that states the exception
STATED_EXCEPTIONS = {
    ('R11 hooks on a network', 'unified:activation_window.py'): _OBSERVER,
    ('R11 hooks on a network', 'unified:percept_window.py'): _OBSERVER,
    ('R20 registry assignment', 'vision:__init__.py'): _DOMAIN_REGISTRIES,
    ('R20 registry assignment', 'language:__init__.py'): _DOMAIN_REGISTRIES,
    ('R20 registry assignment', 'unified:tools/_registry.py'):
        'Ownership Map R20: a tool is registered with register_tool(identifier, factory); '
        'a built-in tool is a lazy import path in _TOOLS, which a dict entry cannot hold',
    ('R20 registry assignment', 'unified:capabilities/registry.py'):
        'Ownership Map R20: a capability carries its own identifier and registers with '
        'register_capability(capability), so the identifier is written once; a factory '
        '(model, benchmark, data, metric) registers by a registry line',
}


def _library_files():
    for repo, package in PACKAGES.items():
        for path, source in python_files(package_root(package)):
            if repo in UMI_CODE and not path.startswith(UMI_CODE[repo]):
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
    compare(violations(_library_files()), {**ALLOWED, **STATED_EXCEPTIONS},
            'second definition sites', REMEDIES)


def test_a_stated_exception_is_not_also_a_violation():
    assert not set(ALLOWED) & set(STATED_EXCEPTIONS)


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
    with pytest.raises(AssertionError, match=r'R15 score construction: .*benchmark_score'):
        compare(violations(files), ALLOWED, 'second definition sites', REMEDIES)
    assert violations(files) == {('R14 encoding regression', 'unified:tools/new_tool.py'),
                                 ('R15 score construction', 'unified:benchmarks/new/benchmark.py'),
                                 ('R6 one Session class', 'core:new.py')}


_PATH = 'unified:benchmarks/lazy/benchmark.py'


@pytest.mark.parametrize('decorator', ['property', 'cached_property', 'functools.cached_property'])
def test_a_ceiling_measured_at_first_use_is_a_reference_and_another_score_is_not(decorator):
    lazy = f'class B:\n    @{decorator}\n    def ceiling(self):\n        return Score(self.rate)\n'
    assert not violations([(_PATH, lazy)])
    built = lazy + '    def __call__(self, s):\n        return Score(0.5)\n'
    assert violations([(_PATH, built)]) == {('R15 score construction', _PATH)}


@pytest.mark.parametrize('source', [
    'def ceiling():\n    return Score(0.5)\n',                                   # a free function
    'class B:\n    def ceiling(self):\n        return Score(0.5)\n',             # not a property
    ('class B:\n    @property\n    def ceiling(self):\n'
     '        def inner():\n            return Score(0.5)\n        return inner\n'),  # nested
])
def test_only_the_return_of_a_ceiling_property_is_a_reference(source):
    assert violations([(_PATH, source)]) == {('R15 score construction', _PATH)}


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
    ('R14 encoding regression', 'import sklearn.linear_model\nfit = sklearn.linear_model.Ridge(1.0)'),
    ('R14 encoding regression', 'from sklearn.cross_decomposition import PLSRegression'),
    ('R15 score construction', 'score = Score(raw)'),
    ('R15 score construction', 'benchmark_score(b, raw=r, ceiling=c)\nextra = Score(0.1)'),
    ('R15 score construction', 'def f(ceiling):\n    return ceiling\ns = f(ceiling=Score(0.5))'),
    ('R15 score construction', 'super().__init__(ceiling=max(Score(0.5), c))'),
    ('R15 score construction', 'from brainscore_core.metrics import Score as S\ns = S(0.5)'),
    ('R15 score construction', 'import brainscore_core.metrics as m\ns = m.Score(0.5)'),
    ('R14 encoding regression', 'import sklearn.linear_model as lm\nr = lm.Ridge()'),
    ('R14 encoding regression', 'from sklearn import linear_model as lm\nr = lm.RidgeCV()'),
    ('R14 encoding regression', 'import sklearn\nr = sklearn.linear_model.Ridge()'),
    ('R14 encoding regression', 'from sklearn.linear_model import Ridge as R'),
    ('R14 encoding regression', 'from sklearn.linear_model import *'),
    ('R17 metadata stamp', 'stamp_score_metadata(score, b, s)'),
    ('R18 cache validity', "os.environ['RESULTCACHING_DISABLE'] = '1'"),
    ('R18 cache validity', "os.environ.pop('RESULTCACHING_DISABLE', None)"),
    ('R18 cache validity', "os.environ.get('RESULTCACHING_DISABLE')"),
    ('R18 cache validity', "os.getenv('RESULTCACHING_DISABLE', '0')"),
    ('R18 cache validity', "on = os.environ['RESULTCACHING_DISABLE'] == '1'"),
    ('R18 cache validity', 'from result_caching import is_enabled'),
    ('R18 cache validity', "import result_caching\non = result_caching.is_enabled('')"),
    ('R20 registry assignment', 'model_registry = {}'),
    ('R20 registry assignment', 'def register_widget(name, w): ...'),
])
def test_signal_catches(name, source):
    signal = RESPONSIBILITIES[name][0]
    assert signal(ast.parse(source))


@pytest.mark.parametrize('name, source', [
    ('R4 N/A decision', 'from brainscore_core.score_status import is_na'),
    ('R12 intervention apply and undo', 'result.data = cast'),
    ('R14 encoding regression', 'from sklearn.linear_model import LogisticRegression'),
    ('R14 encoding regression', 'import sklearn.linear_model\nc = sklearn.linear_model.LogisticRegression()'),
    ('R15 score construction', 'super().__init__(identifier=i, ceiling=Score(0.8))'),
    ('R15 score construction', 'super().__init__(ceiling=not_estimated() if c is None else Score(float(c)))'),
    ('R15 score construction', 'benchmark_score(b, raw=r, ceiling=Score(float(c)))'),
    ('R15 score construction', 'BenchmarkBase.__init__(self, identifier=i, ceiling=Score(c))'),
    ('R14 encoding regression', 'import sklearn.linear_model as lm\nc = lm.LogisticRegression()'),
    ('R18 cache validity', "env.update(RESULTCACHING_DISABLE='1')"),
    ('R18 cache validity', "env = {'RESULTCACHING_DISABLE': '1'}"),
    ('R20 registry assignment', "model_registry['x'] = build"),
    ('R20 registry assignment', 'class Hooks:\n    def register_hook(self): ...'),
])
def test_signal_allows(name, source):
    signal = RESPONSIBILITIES[name][0]
    assert not signal(ast.parse(source))


def test_a_metric_returns_its_raw_score_outside_the_r15_scope():
    """A registered metric returns its raw value as a Score (core Metric contract)."""
    assert not violations([('unified:metrics/new/metric.py', 'score = Score(median_r)\n')])
    assert violations([('unified:benchmarks/new/benchmark.py', 'score = Score(median_r)\n')])


def test_only_the_umi_code_of_vision_and_language_is_read():
    """Brain-Score 2.0 library code is permanent and owns nothing of the contract."""
    sites = {site for site, _ in _library_files()}
    assert {'vision:__init__.py', 'language:__init__.py'} <= sites
    assert any(site.startswith('vision:compat/') for site in sites)
    assert any(site.startswith('language:compat/') for site in sites)
    assert not any(site.startswith(('vision:model_helpers/', 'language:model_helpers/'))
                   for site in sites)



def test_no_library_module_of_vision_reads_the_store_switch():
    """The stores of vision ask ``store_enabled`` (R18). Its plugin folders (``models/``,
    ``benchmarks/``, ``data/``) are Brain-Score 2.0 plugins and are not read."""
    root = package_root('brainscore_vision')
    folders = [path.name for path in root.iterdir()
               if path.name not in ('models', 'benchmarks', 'data') and (path.is_dir() or path.suffix == '.py')]
    readers = [path for path, source in python_files(root, folders) if _reads_cache_switch(ast.parse(source))]
    assert not readers, f'{readers}: {REMEDIES["R18 cache validity"]}'
