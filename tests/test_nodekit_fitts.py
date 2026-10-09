"""Unit tests for the nodekit Fitts benchmark scoring, pointer actions and the
vision-language click adapter. No browser: traces are synthetic."""
import numpy as np
import pytest

from brainscore import benchmark_registry, model_registry
from brainscore.benchmarks.nodekit_fitts.benchmark import load_design, score_trace
from brainscore.harnesses.nodekit_browser import BOARD_SIZE, DOWN, MOVE, UP, click, pointer_action
from brainscore.model_helpers.pointer_policy import build_click_policy, parse_click


def _fitts_trace(design, mt_of_id, n=None):
    events, t = [], 0.0
    for i, spec in enumerate(design['trials'][:n]):
        index = np.log2(spec['distance'] / spec['width'] + 1)
        mt = mt_of_id(index)
        events += [{'event_type': 'NodeStartedEvent', 't': t, 'node_address': [str(i), 'target']},
                   {'event_type': 'PointerSampledEvent', 't': t + mt, 'x': spec['x'], 'y': spec['y'],
                    'kind': 'down'},
                   {'event_type': 'ActionTakenEvent', 't': t + mt, 'node_address': [str(i), 'target'],
                    'action': {}},
                   {'event_type': 'NodeEndedEvent', 't': t + mt, 'node_address': [str(i), 'target']}]
        t += mt + 100
    return {'events': events}


@pytest.fixture(scope='module')
def design():
    return load_design()[0]


def test_design_is_fully_crossed(design):
    trials = design['trials']
    assert len(trials) == 36
    assert {(t['width'], t['distance']) for t in trials} == {
        (w, d) for w in (32, 64, 128) for d in (128, 256, 384)}
    assert load_design()[1].exists()


def test_fitts_conforming_trace_scores_one(design):
    score, info = score_trace(_fitts_trace(design, lambda i: 100 + 150 * i), design)
    assert score == pytest.approx(1.0)
    assert info['slope_ms_per_bit'] == pytest.approx(150)
    assert info['intercept_ms'] == pytest.approx(100)
    assert info['completed'] == 36 and info['per_trial'][0]['hit'] == 1.0


def test_constant_time_and_too_few_trials_score_zero(design):
    assert score_trace(_fitts_trace(design, lambda i: 300), design)[0] == 0.0
    score, info = score_trace(_fitts_trace(design, lambda i: 100 * i, n=2), design)
    assert score == 0.0 and info['completed'] == 2 and np.isnan(info['fitts_r'])


def test_anti_fitts_is_floored(design):
    assert score_trace(_fitts_trace(design, lambda i: 1000 - 100 * i), design)[0] == 0.0


def test_pointer_action_validation():
    np.testing.assert_array_equal(click(3, 4, dt_ms=10), [[10, 3, 4, MOVE], [0, 3, 4, DOWN], [0, 3, 4, UP]])
    with pytest.raises(ValueError):
        pointer_action([(-1, 0, 0, MOVE)])
    with pytest.raises(ValueError):
        pointer_action([(0, 0, 0, 7)])


@pytest.mark.parametrize('text, expected', [
    ('I see it. Click: (512, 100)', (512, 100)),
    ('first 10, 20 then click: (30.5, 40)\nClick: (700, 800)', (700, 800)),
    ('around 300, 400', (300, 400)),
    ('Click: (5000, 10)', None),
    ('no idea', None),
    ('The center is at (512, 51', None),                     # cut off at the token limit
    ('Click: (300, 400)\nso the square is at (512, 51', (300, 400)),
])
def test_parse_click(text, expected):
    assert parse_click(text) == expected


def test_click_policy_maps_pixels_to_board():
    seen = []

    def generate(image, prompt):
        seen.append(prompt)
        return 'Click: (768, 256)'
    policy = build_click_policy(generate, dt_ms=5)
    image = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), np.uint8)
    action = policy({'image': image, 'instruction': 'Hit it.'}, ())
    np.testing.assert_array_equal(action[0], [5, 256, 256, MOVE])  # right and up of centre
    assert 'Hit it.' in seen[0] and '1024 x 1024' in seen[0]
    assert policy.stats == {'calls': 1, 'parse_miss': 0}


def test_click_policy_falls_back_to_seeded_random():
    image = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), np.uint8)
    a = build_click_policy(lambda i, p: 'hmm', seed=3)
    b = build_click_policy(lambda i, p: 'hmm', seed=3)
    np.testing.assert_array_equal(a({'image': image}, ()), b({'image': image}, ()))
    assert a.stats['parse_miss'] == 1


def test_registered():
    assert 'Nodekit-fitts-pointing' in benchmark_registry
    assert {'nodekit-random-pointer', 'nodekit-straight-reach'} <= set(model_registry)
