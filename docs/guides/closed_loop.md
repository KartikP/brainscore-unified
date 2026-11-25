# Closed loop

In a closed-loop task the subject's action decides its next observation: games, simulators, robots.

## Connect an environment

Your environment needs `reset()` and `step(outputs)`, and optionally `close()`.

- `reset()` returns the first **row**: a mapping of channel to payload.
- `step(outputs)` receives what the subject did, a mapping of output channel to payload such as `{'motor': action}`. It returns the next row and a reason: `None` while the episode runs, or `'terminal'`, `'truncated'` or `'failure'` when it ends.

```python
class MyEnvironment:
    """Three steps along a line; the observation is the position."""

    def reset(self, seed=None):
        self.position = 0
        return {'observation': {'x': 0}, 'instruction': 'move right', 'reward': 0.0}

    def step(self, outputs):
        self.position += outputs['motor']
        row = {'observation': {'x': self.position}, 'reward': 0.0}
        return row, ('terminal' if self.position >= 3 else None)  # None: the episode goes on
```

`observation` is defined by your environment. Declaring `observation_spec()` and `action_spec()` (with `ArraySpec`, `ActionSpec` and friends from `brainscore_core.environment`) lets the session validate shapes, units and bounds. [A new kind of output](new_output.md) shows a structured action (a pointer move, a click, a key press) with these specifications.

A task on a screen takes the act forms of a person: `motor:key` (`press`, `release`, `type`, `wait`) and `motor:pointer` (`move`, `down`, `up`, `stroke`, `wait`). Each payload is a list of acts, done in order; positions are pixels of the last picture, origin at the top left; the session checks each act (`MOTOR_ACTS` in `brainscore_core.io_catalog`). The benchmark asks for the form its participants used, and a policy declares the forms it gives with `policy_out_channels`. The NodeKit benchmarks use these forms.

## Connect a policy

A policy maps each row to a mapping of output channel to payload. Give it to `BrainScoreModel` as `policy`:

```python
from brainscore import BrainScoreModel


def choose(row):             # your policy: read the row, return an action
    return 1


subject = BrainScoreModel(identifier='my-agent', policy=lambda row: {'motor': choose(row)})
```

Declare in `policy_in_channels` every row key the policy reads (default `('observation',)`); the session sends only those. A policy that needs earlier steps keeps them itself and clears them in `reset()`, which the model calls when an episode begins ([EXTENDING](../../EXTENDING.md#a-policy-that-keeps-history) has a complete example). A subject with its own loop can implement `participate` directly instead.

## Run an episode

```python
from brainscore_core.streaming_helpers import environment_session, run_environment

# Session that drives the environment
with environment_session(MyEnvironment(), seed=0) as session:
    subject.participate(session)
actions = session.collect('motor')
print(session.source.end_reason)   # 'terminal': the reason the episode ended
subject.reset()

# or, in one call:
actions = run_environment(subject, MyEnvironment(), seed=0)
print(actions)   # [1, 1, 1]
```

`run_environment` returns an empty list when the episode ends at its start.

A benchmark states what an invalid move does. In `GridGame-reach-5x5`, a policy that returns `None` makes an invalid move: the agent stays in place for that step, and the score is marked invalid when more than 10 percent of the steps are invalid.

For long episodes, `environment_session(..., history_limit=N)` bounds memory.

## Established robotics evaluators

When a benchmark already has its own evaluation loop (for example LIBERO or a DROID evaluator), keep that loop and wrap it with `CallableProtocol` in an [experiment](experiments.md). The recording and ablation tools still attach to the policy. Robotics integration is experimental; see [Feature status](../reference/features.md).

See [Lesson 9](../../notebooks/lessons/09_closed_loop.ipynb) and [Experiment 8](../../notebooks/experiments/08_learning_in_a_closed_loop.ipynb).
