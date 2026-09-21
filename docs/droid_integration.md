# Connect a DROID policy

Use `DroidPolicy` to connect a policy that exposes `infer(request)` to UMI.
Your policy owns model loading, image preprocessing, and inference. UMI supplies
observations and records the returned actions. No robot is controlled by this
helper.

## Choose the action definition first

The OpenPI DROID example returns **eight values**: seven joint velocity commands
and one gripper position. Select `action_dict.joint_velocity` and
`action_dict.gripper_position` from the demonstration. Do not substitute the
seven-value top-level RLDS action; it represents a different action space.
See the [official policy example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py).

Declare units, bounds, joint order, and sampling period in `ActionSpec`. Normalized
commands are not radians per second. Converting them to physical motion belongs
to the robot controller. The example below uses normalized commands at a declared
15 Hz. Its bounds validate recorded actions; they are not hardware safety limits.

```python
import numpy as np
from brainscore.harnesses.droid_policy import DroidPolicy
from brainscore.robotics import ActionSpec, evaluate_droid_episode
from brainscore_core.model_interface import BrainScoreModel

spec = ActionSpec(
    names=tuple(f'joint_velocity_{i}' for i in range(7)) + ('gripper_position',),
    units=('normalized_velocity',) * 7 + ('normalized_position',),
    frame='robot_joint_order', period_ms=1000 / 15,
    lower=(-1.,) * 7 + (0.,), upper=(1.,) * 8,
)
provider = DroidPolicy(
    policy, action_spec=spec, reset_policy=reset_policy,
    external_camera='exterior_1', action_horizon=1,
)
model = BrainScoreModel('my-droid-policy', action_fn=provider)
result = evaluate_droid_episode(
    model, episode, action_spec=spec,
    action_source=lambda row: np.concatenate([
        row['action_dict']['joint_velocity'],
        row['action_dict']['gripper_position'],
    ]),
)
```

Supply `policy`, `episode`, and `reset_policy` before running this example.
`reset_policy` must reset whatever your experiment depends on, including random
state for exact paired comparisons. A no-op is appropriate only for a stateless
policy. The provider clears queued actions before calling it; reset failures
propagate.

## What reaches the policy

The request contains the selected external camera, wrist camera, seven joint
positions, gripper position, and instruction. Images remain RGB uint8 arrays.
The provider copies these inputs and leaves preprocessing to the policy. Targets,
rewards, and demonstration actions never enter its request.

The default uses the first predicted action and queries again at every recorded
observation. A larger `action_horizon` explicitly reuses a chunk. Invalid shapes,
nonfinite values, out-of-bounds actions, and nonconsecutive steps raise errors.
There is no automatic clipping or gripper thresholding. Document any transform
inside your policy and use the same transform in direct-reference comparisons.

## Record, replay, and intervene

Pass a `RunRecorder` to `evaluate_droid_episode` to save observations and actions.
Read `RunRecord.outputs()` to recompute measurements without invoking the policy.
Use `observe` for call traces. If you own a supported PyTorch backend, attach
`ActivationWindow` and `intervene` to it as described in the
[tool guide](tool_authoring.md). A remote policy exposes only the internal
measurements its server explicitly returns.

## Evidence so far

A checksum-verified episode from the official `droid_100/1.0.0` sample contains
119 steps and three 180 x 320 RGB cameras. The local test covers all steps,
119 action outputs, 119 activation captures, exact saved-output replay, and an
intervention followed by exact recovery. The policy is an **untrained test
network**. This verifies data transport and tools, not learned robot behavior.

The source record has no measured timestamp field used here. Time is derived
from the declared 15 Hz period and marked as inferred. The tiny sample does not
establish compatibility with every DROID version or episode.

## Remaining trained-policy check

Run an identified DROID checkpoint in its supported environment. Save checkpoint
and normalization-file hashes, software versions, camera choice, random seed,
action transforms, and action horizon. Compare direct policy calls with UMI on
the same observations, then verify records and a supported intervention.

OpenPI documents Ubuntu/NVIDIA inference. Its dependency set should live in a
separate environment from scientific scoring. A trained checkpoint and that
compute environment have not been qualified here. Recorded action agreement
also does not measure task success under policy control; that requires a
separately qualified closed-loop experiment.
