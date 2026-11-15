"""Enforcement check E5: every score entry point delegates to the same core steps.

Contract rule 11.5 and responsibility R5; Architecture and Ownership Map,
section 6, E5.

Reads (with ``ast``, never imported) the three entry points: unified
``brainscore.score``, vision ``brainscore_vision._run_score`` and language
``brainscore_language._run_score``. Two signals:

1. The core steps, in this order: ``check_compatibility`` inside a ``try``
   whose ``except CompatibilityError`` calls ``na_score`` and returns;
   ``check_memory`` under ``if check_mem``; ``subject_for_benchmark``; the
   benchmark run inside one ``with`` that enters ``weight_fingerprint_scope()``
   and ``record_measures()``; ``check_released``; ``stamp_score_metadata``.
2. The steps of each entry point: calls of Brain-Score functions (names
   imported from a Brain-Score package or defined in the same module),
   ``raise`` of an exception class, and keys written to ``attrs``. A step that
   is not in all three entry points is a difference. ``ALLOWED`` holds the
   differences that are violations (target: empty); ``STATED_EXCEPTIONS`` holds
   those that rule 11.5 states, each with its sentence. Both may only shrink.

Cannot detect: a difference in arguments (for example another
``harness_id``) or in what a step does inside; calls through a variable or a
non-Brain-Score helper; a step reached only through ``score`` instead of
``_run_score``. Same-pair behaviour is not compared (rule 10 of the
Ownership Map list).
"""
import ast

import pytest

from tests.architecture_support import compare, package_root

ENTRY_POINTS = {'unified': ('brainscore', 'score'),
                'vision': ('brainscore_vision', '_run_score'),
                'language': ('brainscore_language', '_run_score')}
BRAINSCORE_PACKAGES = ('brainscore', 'brainscore_core', 'brainscore_vision', 'brainscore_language')
CORE_ORDER = ('check_compatibility', 'check_memory', 'subject_for_benchmark',
              'weight_fingerprint_scope', 'record_measures', 'check_released',
              'stamp_score_metadata')

# (step, entry points that have it): reason. Violations only; the target is empty.
ALLOWED = {
    ('call check_benchmark_assets', 'unified'):
        'only unified checks local benchmark assets before loading (R5 drift)',
    ('call score_benchmark', 'vision'):
        'only vision runs score_benchmark and so the preallocate_memory hook (R5, R19 drift)',
}

# (step, entry points that have it): the rule sentence that states the difference
STATED_EXCEPTIONS = {
    ('call _resolve_identifier', 'unified'):
        'Contract Rule 11.5: unified score() accepts a built subject or benchmark as well as '
        'an identifier, and names a built object by its identifier',
    ('raise AssertionError', 'vision'):
        'Contract Rule 11.5: vision adds a hint about its stale activation cache to an '
        'AssertionError that names a stimulus path',
    ("attrs['comment']", 'vision'):
        'Contract Rule 11.5: vision records the layer commitment of a Brain-Score 2.0 '
        'model on its score',
}


def entry_point(package, function):
    tree = ast.parse((package_root(package) / '__init__.py').read_text(encoding='utf-8'))
    return tree, next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == function)


def _brainscore_names(tree):
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or '').split('.')[0]
                                                 in BRAINSCORE_PACKAGES):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _call_name(node):
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)


def steps(module_tree, function):
    known = _brainscore_names(module_tree)
    found = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and _call_name(node) in known:
            found.add(f'call {_call_name(node)}')
        elif isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            found.add(f'raise {ast.unparse(exc)}')
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) \
                and isinstance(node.value, ast.Attribute) and node.value.attr == 'attrs':
            found.add(f'attrs[{ast.unparse(node.slice)}]')
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ('update', 'setdefault') \
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'attrs':
            keys = [repr(keyword.arg) for keyword in node.keywords]
            if node.func.attr == 'setdefault':
                keys.append(ast.unparse(node.args[0]))
            found.update(f'attrs[{key}]' for key in keys)
    return found


def core_step_problems(function):
    """Problems with the required core steps of one entry point; empty when it conforms."""
    problems, order = [], []
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and _call_name(node) in CORE_ORDER:
            order.append((node.lineno, node.col_offset, _call_name(node)))
    first = {}
    for _, _, name in sorted(order):
        first.setdefault(name, len(first))
    missing = [name for name in CORE_ORDER if name not in first]
    if missing:
        problems.append(f'missing core steps: {missing}')
    present = [name for name in CORE_ORDER if name in first]
    if sorted(present, key=first.get) != present:
        problems.append(f'core steps out of order: {sorted(present, key=first.get)}')

    guarded = False
    for node in ast.walk(function):
        if isinstance(node, ast.Try) and any(
                isinstance(c, ast.Call) and _call_name(c) == 'check_compatibility'
                for s in node.body for c in ast.walk(s)):
            for handler in node.handlers:
                if handler.type is not None and ast.unparse(handler.type).endswith('CompatibilityError'):
                    calls = {_call_name(c) for c in ast.walk(handler) if isinstance(c, ast.Call)}
                    returns = any(isinstance(s, ast.Return) for s in ast.walk(handler))
                    guarded = guarded or ('na_score' in calls and returns)
    if not guarded:
        problems.append('check_compatibility is not in try/except CompatibilityError returning na_score')

    if not any(isinstance(node, ast.If) and 'check_mem' in ast.unparse(node.test)
               and any(isinstance(c, ast.Call) and _call_name(c) == 'check_memory'
                       for c in ast.walk(node))
               for node in ast.walk(function)):
        problems.append('check_memory is not under `if check_mem`')

    if not any(isinstance(node, ast.With) and
               {'weight_fingerprint_scope', 'record_measures'} <=
               {_call_name(item.context_expr) for item in node.items
                if isinstance(item.context_expr, ast.Call)}
               for node in ast.walk(function)):
        problems.append('the run is not inside one `with weight_fingerprint_scope(), record_measures()`')
    return problems


def differences(per_entry):
    every = set.intersection(*per_entry.values())
    union = set.union(*per_entry.values())
    return {(step, '+'.join(name for name in per_entry if step in per_entry[name]))
            for step in union - every}


def _all_steps():
    result = {}
    for name, (package, function) in ENTRY_POINTS.items():
        tree, node = entry_point(package, function)
        result[name] = steps(tree, node)
    return result


@pytest.mark.parametrize('name', ENTRY_POINTS)
def test_entry_point_calls_the_core_steps_in_order(name):
    _, function = entry_point(*ENTRY_POINTS[name])
    assert not core_step_problems(function), core_step_problems(function)


def test_entry_points_differ_only_by_known_steps():
    compare(differences(_all_steps()), {**ALLOWED, **STATED_EXCEPTIONS},
            'differences between score entry points')


def test_a_stated_difference_is_not_also_a_violation():
    assert not set(ALLOWED) & set(STATED_EXCEPTIONS)


_CONFORMING = '''
def score(subject, benchmark, check_mem=True):
    try:
        c = check_compatibility(subject, benchmark)
    except CompatibilityError as error:
        return na_score(error)
    if check_mem:
        check_memory(subject, benchmark)
    received = subject_for_benchmark(subject, benchmark)
    with weight_fingerprint_scope(), record_measures() as measures:
        result = benchmark(received)
    measures.check_released(b, s)
    stamp_score_metadata(result, subject)
    return result
'''


def test_core_step_check_passes_a_conforming_entry_point():
    assert not core_step_problems(ast.parse(_CONFORMING).body[0])


@pytest.mark.parametrize('old, new, problem', [
    ('    if check_mem:\n        check_memory(subject, benchmark)\n', '', 'missing core steps'),
    ('        return na_score(error)\n', '        raise\n', 'returning na_score'),
    ('weight_fingerprint_scope(), record_measures()', 'record_measures()', 'missing core steps'),
    ('    stamp_score_metadata(result, subject)\n', '', 'missing core steps'),
])
def test_core_step_check_fails_on_a_synthetic_violation(old, new, problem):
    source = _CONFORMING.replace(old, new)
    assert source != _CONFORMING
    assert any(problem in p for p in core_step_problems(ast.parse(source).body[0]))


def test_difference_check_fails_on_a_new_violation():
    per_entry = _all_steps()
    per_entry['language'] = per_entry['language'] | {'call reserve_memory'}
    with pytest.raises(AssertionError, match='new differences'):
        compare(differences(per_entry), {**ALLOWED, **STATED_EXCEPTIONS},
                'differences between score entry points')


def test_difference_check_fails_on_a_stale_entry():
    per_entry = _all_steps()
    for name in ('vision', 'language'):
        per_entry[name] = per_entry[name] | {'call check_benchmark_assets'}
    with pytest.raises(AssertionError, match='no longer violate'):
        compare(differences(per_entry), {**ALLOWED, **STATED_EXCEPTIONS},
                'differences between score entry points')
