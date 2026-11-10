import numpy as np
import pytest
from brainscore.harnesses.droid_policy import DroidPolicy
from brainscore.robotics import ActionSpec

pytestmark = pytest.mark.unit


def spec():
    return ActionSpec(tuple(f'joint_{i}' for i in range(7)) + ('gripper',),
        ('normalized_velocity',)*7 + ('normalized_position',), 'robot_joint_order',
        1000/15, (-1.,)*7+(0.,), (1.,)*8)


def step():
    return {
        'vision:image': {k: np.full((3, 4, 3), i, dtype=np.uint8)
                         for i, k in enumerate(('exterior_1', 'exterior_2', 'wrist'))},
        'proprioception': {'joint_position': np.zeros(7), 'cartesian_position': np.zeros(6),
                           'gripper_position': np.zeros(1)},
        'instruction': 'move the cup'}


def act(adapter, row):
    """The motor event the policy returned for one row."""
    return adapter(row)['motor']


class Policy:
    def __init__(self, actions=None):
        self.requests=[]
        self.actions=np.zeros((3, 8)) if actions is None else actions
    def infer(self, request):
        self.requests.append(request)
        return {'actions': self.actions}


def test_named_mapping_keeps_targets_out_and_copies_observations():
    policy=Policy(); resets=[]
    adapter=DroidPolicy(policy,action_spec=spec(),reset_policy=lambda:resets.append(1),external_camera='exterior_2')
    adapter.reset()
    observation=step()
    response=act(adapter, observation)
    request=policy.requests[0]
    assert set(request)=={'observation/exterior_image_1_left','observation/wrist_image_left',
        'observation/joint_position','observation/gripper_position','prompt'}
    assert request['prompt']=='move the cup'
    assert np.all(request['observation/exterior_image_1_left']==1)
    request['observation/joint_position'][:]=9
    assert np.all(observation['proprioception']['joint_position']==0)
    assert response.payload.shape==(8,) and resets==[1]


def test_chunk_queue_clears_between_episodes_and_on_reset():
    policy=Policy(np.arange(24).reshape(3,8)/24)
    adapter=DroidPolicy(policy,action_spec=spec(),reset_policy=lambda:None,action_horizon=3)
    a=act(adapter, step()).payload
    b=act(adapter, step()).payload
    assert len(policy.requests)==1 and not np.array_equal(a,b)
    adapter.reset()  # BrainScoreModel calls this when an episode begins
    np.testing.assert_array_equal(act(adapter, step()).payload,a)
    assert len(policy.requests)==2
    adapter.reset()
    np.testing.assert_array_equal(act(adapter, step()).payload,a)
    assert len(policy.requests)==3


@pytest.mark.parametrize('actions',[np.zeros(8),np.zeros((1,7)),np.zeros((0,8)),
    np.full((3,8),np.nan),np.full((3,8),2)])
def test_invalid_chunk_is_rejected_without_clipping(actions):
    adapter=DroidPolicy(Policy(actions),action_spec=spec(),reset_policy=lambda:None)
    with pytest.raises(ValueError):adapter(step())
    assert not adapter._pending


def test_policy_reset_failure_propagates():
    def fail():raise RuntimeError('reset failed')
    adapter=DroidPolicy(Policy(),action_spec=spec(),reset_policy=fail)
    with pytest.raises(RuntimeError,match='reset failed'):adapter.reset()


def test_chunk_metadata_declares_predictions_execution_and_owner():
    policy = Policy()
    adapter = DroidPolicy(
        policy, action_spec=spec(), reset_policy=lambda: None, action_horizon=2,
    )
    responses = [act(adapter, step()) for n in range(3)]
    assert len(policy.requests) == 2
    assert [r.meta['chunk_id'] for r in responses] == [0, 0, 1]
    assert [r.meta['chunk_index'] for r in responses] == [0, 1, 0]
    assert [r.meta['step_num'] for r in responses] == [0, 1, 2]
    for response in responses:
        assert response.meta['prediction_horizon'] == 3
        assert response.meta['execution_horizon'] == 2
        assert response.meta['scheduler'] == 'policy_adapter'
        assert response.meta['action_kind'] == 'single'
        assert response.meta['control_period_ms'] == spec().period_ms


def test_bad_camera_frame_is_rejected_before_inference():
    """Moved from core: the robot harness, not core, validates camera frames."""
    policy = Policy()
    adapter = DroidPolicy(policy, action_spec=spec(), reset_policy=lambda: None)
    bad = step()
    bad['vision:image']['wrist'] = np.zeros((3, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match='wrist'):
        adapter(bad)
    assert policy.requests == []
