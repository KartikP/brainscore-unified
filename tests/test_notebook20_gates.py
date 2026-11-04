"""The failure gates in notebook 20, run offline on the notebook's own helper code.

The helpers are read from the notebook source, so these tests check exactly
what the notebook runs. No model is loaded.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

torch = pytest.importorskip('torch')

# ponytail: the blind-discovery section holding require_damage was cut from
# experiments/12; the archived original still defines both gates verbatim.
NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks' / 'archive' / '20_steer_a_language_model.ipynb'


def _helpers(*names):
    cells = json.loads(NOTEBOOK.read_text())['cells']
    source = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
    tree = ast.parse(source)
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert sorted(n.name for n in found) == sorted(names)
    namespace = {'np': np}
    exec(compile(ast.Module(body=found, type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
    return [namespace[name] for name in names]


def _push(before, added):
    """A native bfloat16 push, as the intervention hook computes it."""
    before = torch.tensor(before, dtype=torch.bfloat16)
    return before.float().numpy(), (before + added).float().numpy()


@pytest.fixture
def push_tolerance():
    return _helpers('push_tolerance')[0]


@pytest.fixture
def require_damage():
    return _helpers('require_damage')[0]


def _passes(push_tolerance, before, after, added):
    return bool(np.all(np.abs((after - before) - added) <= push_tolerance(added, after)))


def test_cancellation_to_zero_is_a_correct_push(push_tolerance):
    before, after = _push([-23.125], 23.1)
    assert after[0] == 0.0 and abs((after - before)[0] - 23.1) > 1e-5  # the amount's own rounding
    assert _passes(push_tolerance, before, after, 23.1)


@pytest.mark.parametrize('added', [23.1, -23.1, 0.37, 1e3, 5.0])
def test_normal_pushes_pass(push_tolerance, added):
    before, after = _push(np.random.default_rng(0).normal(0, 3, 500), added)
    assert _passes(push_tolerance, before, after, added)


@pytest.mark.parametrize('applied', [0.0, 23.6, 22.6, 46.2, -23.1])
def test_incorrect_pushes_fail(push_tolerance, applied):
    before, after = _push(np.random.default_rng(1).normal(0, 3, 500), applied)
    assert not _passes(push_tolerance, before, after, 23.1)


def test_valid_damage_passes_and_ratio_is_unchanged(require_damage):
    drops = np.array([0.01, 0.5, 1.3])
    require_damage(drops, 0.01, 'valid')
    np.testing.assert_array_equal(np.array([1., 2., 3.]) / drops, [100., 4., 3. / 1.3])


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, 0.0, -0.2, 0.0099, 1e-12])
def test_unusable_denominators_stop_the_notebook(require_damage, bad):
    with pytest.raises(ValueError, match=r'failing positions \[1\]'):
        require_damage([0.5, bad, 1.0], 0.01, 'neuron pushes')


def test_both_ratios_are_gated_before_division():
    source = '\n'.join(''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells'])
    table_gate = source.index("require_damage(table['disruption']")
    null_gate = source.index("require_damage([r['disruption'] for r in null_results]")
    assert table_gate < source.index("table['discovery'] / table['disruption']")
    assert null_gate < source.index("r['validation'] / r['disruption']")
