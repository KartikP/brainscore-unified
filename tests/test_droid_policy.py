import numpy as np
import pytest
from brainscore.harnesses.droid_policy import DroidPolicy
from brainscore.robotics import ActionSpec
from brainscore_core.events import CameraFrame, Proprioception, EnvironmentStep

pytestmark = pytest.mark.unit


def spec():
    return ActionSpec(tuple(f'joint_{i}' for i in range(7)) + ('gripper',),
        ('normalized_velocity',)*7 + ('normalized_position',), 'robot_joint_order',
        1000/15, (-1.,)*7+(0.,), (1.,)*8)


def step(n=0, first=False, last=False):
    return EnvironmentStep(observation={
        'cameras': {k: CameraFrame(rgb=np.full((3, 4, 3), i, dtype=np.uint8))
                    for i, k in enumerate(('exterior_1','exterior_2','wrist'))},
        'proprioception': Proprioception(joint_position=np.zeros(7),
             cartesian_position=np.zeros(6), gripper_position=np.zeros(1))},
        instruction='move the cup', step_num=n, is_first=first, is_last=last)


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
    observation=step(first=True)
    response=adapter(observation)
    request=policy.requests[0]
    assert set(request)=={'observation/exterior_image_1_left','observation/wrist_image_left',
        'observation/joint_position','observation/gripper_position','prompt'}
    assert request['prompt']=='move the cup'
    assert np.all(request['observation/exterior_image_1_left']==1)
    request['observation/joint_position'][:]=9
    assert np.all(observation.observation['proprioception'].joint_position==0)
    assert response.action.shape==(8,) and resets==[1]


def test_chunk_queue_clears_between_episodes_and_on_reset():
    policy=Policy(np.arange(24).reshape(3,8)/24)
    adapter=DroidPolicy(policy,action_spec=spec(),reset_policy=lambda:None,action_horizon=3)
    a=adapter(step(first=True)).action
    b=adapter(step(1,last=True)).action
    assert len(policy.requests)==1 and not np.array_equal(a,b)
    np.testing.assert_array_equal(adapter(step(first=True)).action,a)
    assert len(policy.requests)==2
    adapter.reset()
    np.testing.assert_array_equal(adapter(step()).action,a)
    assert len(policy.requests)==3


@pytest.mark.parametrize('actions',[np.zeros(8),np.zeros((1,7)),np.zeros((0,8)),
    np.full((3,8),np.nan),np.full((3,8),2)])
def test_invalid_chunk_is_rejected_without_clipping(actions):
    adapter=DroidPolicy(Policy(actions),action_spec=spec(),reset_policy=lambda:None)
    with pytest.raises(ValueError):adapter(step())
    assert not adapter._pending


def test_out_of_order_step_does_not_use_stale_action():
    adapter=DroidPolicy(Policy(),action_spec=spec(),reset_policy=lambda:None,action_horizon=3)
    adapter(step())
    with pytest.raises(ValueError,match='consecutive'):adapter(step(2))
    assert not adapter._pending


def test_policy_reset_failure_propagates():
    def fail():raise RuntimeError('reset failed')
    adapter=DroidPolicy(Policy(),action_spec=spec(),reset_policy=fail)
    with pytest.raises(RuntimeError,match='reset failed'):adapter.reset()
