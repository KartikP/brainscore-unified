# Connect a robotics benchmark: the LIBERO example

For shared specifications, episode records, and clocks, see [environment sessions](environment_sessions.md).

**LIBERO is a collection of simulated robot manipulation tasks.** A policy
receives camera images, robot state, and an instruction, then predicts movement
commands. The simulator applies those commands and returns new observations.
The benchmark counts how often the robot completes the task.

This guide explains how we connected OpenPI's trained π₀.₅-LIBERO policy to UMI.
The same approach can help connect another simulator or evaluation framework.
It does not establish support for arbitrary policies or physical robots.

## Will every benchmark need an adapter?

Usually a new framework needs some integration code. You can reuse it when
benchmarks share the same observation, action, and policy APIs. A new task in
LIBERO does not automatically require a new adapter.

There are two boundaries to check:

| Boundary | What must agree | Our LIBERO implementation |
| --- | --- | --- |
| Policy to UMI | Input fields, action meaning, shapes, and reset behavior | `LiberoChunkPolicy` translates a UMI event into an OpenPI `infer(request)` call. |
| Evaluator to UMI | How observations arrive, actions return, and results are recorded | A websocket bridge lets the official evaluator call the policy through UMI. |

`infer()` is OpenPI's API, not a requirement for every robotics policy. Adapt
another policy's callable method at this boundary. Keep task success rules in
the evaluator; UMI does not define what counts as success for every robot.

## Follow one request

```text
Official LIBERO evaluator prepares images, state, and instruction
  → websocket bridge creates an EnvironmentStep
  → BrainScoreModel.process(step)
  → LiberoChunkPolicy calls the original policy
  → EnvironmentResponse carries the predicted action chunk
  → bridge returns it to the official evaluator
  → evaluator applies actions, advances the simulator, and scores success
```

An **action chunk** is a sequence of predicted future commands. We return the
whole chunk. The official evaluator decides how many commands to execute before
asking again. Changing that schedule would change the experiment.

`BrainScoreModel` supplies the existing UMI machinery here; the policy adapter
is supplied through `action_fn`. We did not change the core contract. This route
uses `process()` and does not test the `EnvironmentSession` driver.

## How we built it

### 1. Keep the reference experiment intact

We pinned OpenPI, its LIBERO dependency, and the checkpoint configuration. We
kept the official task loop, initial states, image transforms, action schedule,
and success scoring. See the [run instructions](../examples/libero/README.md)
for revisions and commands.

For another benchmark, first identify its official evaluator and settings.
Record model weights and normalization-file hashes as well as source revisions.
A URL alone does not identify immutable weights.

### 2. Write down the observation and action meanings

Our adapter accepts the evaluator's already prepared inputs:

| Field | Meaning |
| --- | --- |
| `observation/image` | External camera image, RGB uint8 |
| `observation/wrist_image` | Wrist camera image, RGB uint8 |
| `observation/state` | Eight state values in the upstream evaluator's convention |
| `prompt` | The task instruction |
| Returned `actions` | A nonempty array of shape `(horizon, 7)`: six end-effector commands plus a gripper command |

These seven action values are **not seven joint velocities**. Our DROID helper
uses a different action definition and cannot be substituted based on similar
array shapes. Preserve the checkpoint's conventions and the controller's
scaling; document any conversion your integration needs.

### 3. Implement a small policy adapter

[`LiberoChunkPolicy`](../brainscore/harnesses/libero_policy.py) copies the allowed
input fields, calls the policy, validates the returned shape and finite values,
and packages the result as an `EnvironmentResponse`. It does not clip commands,
rescale images, or expose rewards and target actions to the policy.

Wire it into the existing model helper:

```python
from brainscore_core.model_interface import BrainScoreModel
from brainscore.harnesses.libero_policy import LiberoChunkPolicy

# policy is an initialized OpenPI policy or its websocket client.
subject = BrainScoreModel('pi05-libero', action_fn=LiberoChunkPolicy(policy))
# step is an EnvironmentStep with the fields described above.
response = subject.process(step)
```

For a partner integration, the adapter can live in your own package. Supplying
it as `action_fn` does not require modifying the UMI core or registering a new
built-in capability.

### 4. Connect the evaluator and retain evidence

[`serve_bridge.py`](../examples/libero/serve_bridge.py) offers two routes: a
reference call directly to the policy and a call through UMI. Both record inputs
and outputs with `RunRecorder`. The network boundary also lets LIBERO, OpenPI,
and UMI keep their separate Python environments.

[`evaluate.py`](../examples/libero/evaluate.py) runs the pinned official evaluator
and adds trial records, initial-state hashes, error reporting, and distinct video
filenames. These additions make the result inspectable without redefining its
score. Failed and interrupted trials must remain visible.

Check reset behavior explicitly. OpenPI's websocket client does not reset the
server's random state, so we restart the policy server for matched runs. This
integration does not establish support for recurrent policies that need a
per-episode reset protocol.

### 5. Test the translation and the experiment separately

| Check | What it tells you |
| --- | --- |
| Adapter tests | Inputs and actions survive translation; invalid values fail; targets do not leak into policy inputs. |
| Transport tests | Requests and responses cross the real websocket boundary. A synthetic policy is sufficient for this check. |
| Trained-policy replay | Identical saved inputs produce matching action values through the reference and UMI routes. |
| Paired simulator runs | Both routes complete the same task/trial set, with recorded success rates and any disagreements. |

Start with one trial per task. Then run the official trial count before claiming
a benchmark result. Exact same-input replay and matching task success measure
different things: matching scores alone does not prove identical actions.

Tests are in [test_libero_policy.py](../tests/test_libero_policy.py) and
[test_libero_transport.py](../tests/test_libero_transport.py). The
[run instructions](../examples/libero/README.md#review-the-evidence) identify the
reports to inspect.

## Setup fixes are separate from the adapter

Our pinned LIBERO container needed compatible Python build tools and dependency
installation repairs. These are captured in `prepare_runtime.py`; they do not
change the benchmark's task or scoring code.

Fresh GPU policy servers also produced slightly different actions without UMI.
Reusing GPU compiler autotuning choices restored exact same-input agreement in
the smoke test. Preserve the failed checks and the successful execution profile;
do not assume the saved tuning file works on another GPU or software stack.

## What has been demonstrated

With cached autotuning, the trained-policy smoke test completed **10/10 tasks on
both routes**, and **221 saved inference requests matched exactly** through UMI.
The full Spatial evaluation is still running at the time of writing. This is a
candidate integration, not a completed full-suite qualification.

This example does not demonstrate internal activation recording, interventions,
physical robot control, or a trained DROID evaluation. Each needs its own
integration and evidence. Use the [DROID guide](droid_integration.md) for the
recorded-episode path.
