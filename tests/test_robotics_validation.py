"""Validate recorded robot observations before using them for evaluation."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from brainscore.robotics import ActionSpec, droid_steps, evaluate_droid_episode, staged_droid_episodes
from brainscore_core.events import EnvironmentResponse


@pytest.fixture
def spec():
    return ActionSpec(('x',), ('normalized',), 'tool', 50., (-1.,), (1.,))


@pytest.fixture
def episode():
    observation = {
        name: np.zeros((2, 3, 3), dtype=np.uint8)
        for name in ('wrist_image_left', 'exterior_image_1_left', 'exterior_image_2_left')
    }
    observation.update(
        joint_position=np.zeros(7),
        cartesian_position=np.zeros(6),
        gripper_position=np.zeros(1),
    )
    return {'steps': [
        {'observation': deepcopy(observation), 'action': np.array([0.5]),
         'is_first': index == 0, 'is_last': index == 1, 'language_instruction': b'pick up cup'}
        for index in range(2)
    ]}


@pytest.mark.parametrize('changes,message', [
    ({'names': ()}, 'nonempty and unique'),
    ({'names': ('x', 'x')}, 'nonempty and unique'),
    ({'names': ('',)}, 'nonempty and unique'),
    ({'units': ()}, 'equal length'),
    ({'units': ('',)}, 'Explicit units'),
    ({'frame': ''}, 'coordinate frame'),
    ({'period_ms': 0}, 'positive and finite'),
    ({'period_ms': np.nan}, 'positive and finite'),
    ({'period_ms': np.inf}, 'positive and finite'),
    ({'lower': (np.nan,)}, 'finite and ordered'),
    ({'upper': (np.inf,)}, 'finite and ordered'),
    ({'lower': (2.,)}, 'finite and ordered'),
])
def test_action_spec_requires_explicit_consistent_semantics(spec, changes, message):
    with pytest.raises(ValueError, match=message):
        replace(spec, **changes)


@pytest.mark.parametrize('field,value,message', [
    ('joint_position', np.zeros(6), 'joint_position'),
    ('cartesian_position', np.full(6, np.nan), 'cartesian_position'),
    ('gripper_position', ['open'], 'gripper_position'),
    ('wrist_image_left', np.zeros((2, 3)), 'uint8 image'),
    ('exterior_image_1_left', np.zeros((2, 3, 4), dtype=np.uint8), 'uint8 image'),
    ('exterior_image_2_left', np.zeros((2, 3, 3)), 'uint8 image'),
])
def test_invalid_first_observation_never_reaches_policy_and_resets(episode, spec, field, value, message):
    episode['steps'][0]['observation'][field] = value
    calls, resets = [], []
    subject = SimpleNamespace(process=lambda step: calls.append(step), reset=lambda: resets.append(True))
    with pytest.raises(ValueError, match=message):
        evaluate_droid_episode(subject, episode, action_spec=spec, action_source=lambda row: row['action'])
    assert calls == [] and resets == [True, True]


@pytest.mark.parametrize('times', [(0., 0.), (1., 0.), (0., np.nan), (0., np.inf)])
def test_recorded_timestamps_must_increase(episode, spec, times):
    for row, time in zip(episode['steps'], times):
        row['observation']['time_ms'] = time
    with pytest.raises(ValueError, match='strictly increasing'):
        list(droid_steps(episode, action_spec=spec, action_source=lambda row: row['action'], timestamp_key='time_ms'))


@pytest.mark.parametrize('damage,message', [
    (lambda rows: rows.clear(), 'nonempty episode'),
    (lambda rows: rows[-1].update(is_last=False), 'ending with is_last'),
    (lambda rows: rows[0].update(is_last=True), 'steps after is_last'),
    (lambda rows: rows[0].update(is_first=False), 'exactly the first step'),
    (lambda rows: rows[1].update(is_first=True), 'exactly the first step'),
    (lambda rows: rows[0].update(language_instruction=123), 'text or UTF-8 bytes'),
])
def test_invalid_episode_boundaries_or_instruction(episode, spec, damage, message):
    damage(episode['steps'])
    with pytest.raises(ValueError, match=message):
        list(droid_steps(episode, action_spec=spec, action_source=lambda row: row['action']))


def test_named_timestamps_and_snapshots_preserve_source_semantics(episode, spec):
    for index, row in enumerate(episode['steps']):
        row['observation']['time_ms'] = 100. + index * 60.
    pairs = list(droid_steps(episode, action_spec=spec, action_source=lambda row: row['action'], timestamp_key='time_ms'))
    step, target = pairs[0]
    assert [pair[0].context['t_ms'] for pair in pairs] == [100., 160.]
    assert step.context['time_source'] == 'time_ms' and not step.context['time_inferred']
    assert step.instruction == 'pick up cup'
    episode['steps'][0]['observation']['wrist_image_left'][:] = 255
    episode['steps'][0]['observation']['joint_position'][:] = 1
    episode['steps'][0]['action'][:] = -1
    assert not step.observation['cameras']['wrist'].rgb.any()
    assert not step.observation['proprioception'].joint_position.any()
    np.testing.assert_array_equal(target, [0.5])


@pytest.mark.parametrize('response,error', [
    ({'action': [0.]}, TypeError),
    (EnvironmentResponse(action=np.array([np.nan])), ValueError),
    (EnvironmentResponse(action=np.array([2.])), ValueError),
])
def test_invalid_prediction_resets_policy(episode, spec, response, error):
    resets = []
    subject = SimpleNamespace(process=lambda step: response, reset=lambda: resets.append(True))
    with pytest.raises(error):
        evaluate_droid_episode(subject, episode, action_spec=spec, action_source=lambda row: row['action'])
    assert resets == [True, True]


def test_staged_data_requires_existing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(staged_droid_episodes(tmp_path / 'missing'))


def test_staged_data_explains_optional_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'tensorflow_datasets', None)
    with pytest.raises(ImportError, match=r'brainscore\[robotics-data\]'):
        list(staged_droid_episodes(tmp_path))


def test_staged_data_uses_local_builder_and_requested_split(tmp_path, monkeypatch):
    calls = []
    builder = SimpleNamespace(as_dataset=lambda **kwargs: calls.append(kwargs) or ['fixture'])
    tfds = SimpleNamespace(
        builder_from_directory=lambda path: calls.append(path) or builder,
        as_numpy=lambda dataset: iter(dataset),
    )
    monkeypatch.setitem(sys.modules, 'tensorflow_datasets', tfds)
    assert list(staged_droid_episodes(tmp_path, split='train[:1]')) == ['fixture']
    assert calls == [str(tmp_path), {'split': 'train[:1]'}]
