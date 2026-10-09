"""Unit tests for the nodekit trace reader and cursor metrics (no browser)."""
import numpy as np
import pytest

from brainscore.cursor_traces import read_trials, resample, trajectory_metrics


def _node(address, t0, samples, t_action=None, t_end=None):
    events = [{'event_type': 'NodeStartedEvent', 't': t0, 'node_address': list(address)}]
    events += [{'event_type': 'PointerSampledEvent', 't': t, 'x': x, 'y': y, 'kind': k}
               for t, x, y, k in samples]
    if t_action is not None:
        events.append({'event_type': 'ActionTakenEvent', 't': t_action, 'node_address': list(address),
                       'action': {'action_type': 'SelectAction', 'action_value': address[-1]}})
    if t_end is not None:
        events.append({'event_type': 'NodeEndedEvent', 't': t_end, 'node_address': list(address)})
    return events


def _trace():
    events = [{'event_type': 'TraceStartedEvent', 't': 0},
              {'event_type': 'PointerSampledEvent', 't': 1, 'x': 5, 'y': 5, 'kind': 'move'}]  # before any node
    events += _node(('0', 'target'), 10, [(20, 0, 0, 'move'), (50, 50, 30, 'move'),
                                          (80, 100, 0, 'move'), (80, 100, 0, 'down')], 80, 80)
    events += _node(('1', 'target'), 100, [(130, 10, 0, 'move')])  # never answered
    return {'nodekit_version': 'x', 'events': events}


def test_read_trials_splits_by_node():
    trials = read_trials(_trace())
    assert [t.node_address for t in trials] == [('0', 'target'), ('1', 'target')]
    first, second = trials
    assert first.completed and first.t_action == 80 and first.node_id == 'target'
    assert first.samples.shape == (4, 4) and first.samples[-1, 3] == 1  # 'down'
    assert not second.completed and len(second.samples) == 1


def test_metrics_on_curved_path():
    trial = read_trials(_trace())[0]
    m = trajectory_metrics(trial, start=(0, 0), target=(100, 0), target_size=(20, 20))
    assert m['initiation_time'] == 40       # first sample > 2 px from start is at t=50
    assert m['movement_time'] == 30
    assert m['response_time'] == 70
    assert m['mad'] == pytest.approx(30)    # bulges left of the rightward line
    assert m['auc'] > 0
    assert m['path_length'] == pytest.approx(2 * np.hypot(50, 30))
    assert m['x_flips'] == 0 and m['hit'] == 1.0 and m['endpoint_error'] == 0


def test_metrics_signs_and_flips():
    trial = read_trials(_trace())[0]
    trial.samples[1, 2] = -30               # bulge to the right instead
    trial.samples[2, 1] = 20                # and turn back once in x
    m = trajectory_metrics(trial, start=(0, 0), target=(100, 0))
    assert m['mad'] == pytest.approx(-30)
    assert m['x_flips'] == 2


def test_metrics_miss_and_no_samples():
    trials = read_trials(_trace())
    m = trajectory_metrics(trials[0], start=(0, 0), target=(300, 0), target_size=(20, 20))
    assert m['hit'] == 0.0 and m['endpoint_error'] == pytest.approx(200)
    trials[1].samples = trials[1].samples[:0]
    empty = trajectory_metrics(trials[1], start=(0, 0))
    assert all(np.isnan(v) for v in empty.values())


def test_resample_endpoints():
    path = resample(read_trials(_trace())[0], n=11)
    assert path.shape == (11, 2)
    np.testing.assert_allclose(path[0], (0, 0))
    np.testing.assert_allclose(path[-1], (100, 0))
