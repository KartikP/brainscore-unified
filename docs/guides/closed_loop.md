# Closed loop

In a closed-loop task the subject's action decides its next observation: games, simulators, robots.

## Connect an environment

Your environment needs `reset(seed=None, options=None)` and `step(action)`, each returning an `EnvironmentStep`, and optionally `close()`.

```python
from brainscore_core import EnvironmentStep

EnvironmentStep(observation={...}, step_num=3, instruction='pick up the cup',
                is_first=False, is_last=False, is_terminal=False, reward=0.0)
```

`observation` is defined by your environment. Declaring `observation_spec()` and `action_spec()` (with `ArraySpec`, `ActionSpec` and friends from `brainscore_core.environment`) lets the session validate shapes, units and bounds.

## Connect a policy

A policy maps each `EnvironmentStep` to an `EnvironmentResponse(action=...)`. Give it to `BrainScoreModel` as `policy`:

```python
from brainscore_core import BrainScoreModel, EnvironmentResponse

agent = BrainScoreModel(identifier='my-agent', policy=lambda step: EnvironmentResponse(action=choose(step)))
```

`PolicyWrapper` (in `brainscore.model_helpers.policy_wrapper`) adapts existing policy objects. A subject with its own loop can implement `participate` directly instead.

## Run an episode

```python
from brainscore_core.streaming_helpers import EnvironmentSession, run_environment

with EnvironmentSession(MyEnvironment(), seed=0) as session:
    agent.participate(session)
actions = session.collect('motor')
agent.reset()

# or, in one call:
actions = run_environment(agent, MyEnvironment(), seed=0)
```

For long episodes, `EnvironmentSession(history_limit=N)` bounds memory.

## Established robotics evaluators

When a benchmark already has its own evaluation loop (for example LIBERO or a DROID evaluator), keep that loop and wrap it with `CallableProtocol` in an [experiment](experiments.md). The recording and ablation tools still attach to the policy. Robotics integration is experimental; see [Feature status](../reference/features.md).

See [Lesson 9](../../notebooks/lessons/09_closed_loop.ipynb) and [Experiment 8](../../notebooks/experiments/08_learning_in_a_closed_loop.ipynb).
