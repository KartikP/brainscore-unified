# Environment sessions

An **environment wrapper** supplies observations and applies actions. A **policy
adapter** connects a policy to the subject. An **evaluator** defines the task and
its score. A **session** connects them for one episode.

Use `EnvironmentSession` for a UMI-managed loop. Keep an established evaluator's
loop when reproducing its benchmark. Both routes can use experiment tools.

## Declare observations and actions

Import specifications from `brainscore_core.environment`:

| Specification | Describes |
| --- | --- |
| `ArraySpec` | Array dimensions, numeric type, and optional bounds |
| `DiscreteSpec` | A range of integer choices |
| `TextSpec` | A text field |
| `MappingSpec` | Named fields with their own specifications |
| `ActionSpec` | Continuous commands with names, units, frame, bounds, and period |

An environment wrapper exposes `observation_spec()`, `action_spec()`,
`reset(...)`, and `step(action)`. Reset and step return `EnvironmentStep`.
Specifications implement `validate(value)` and `describe()`; custom specifications
can live in an external package. `brainscore.robotics.ActionSpec` exports the same
core class.

`EnvironmentSession(..., require_specs=True)` checks these declarations before
reset. Observations are validated before inference, and actions before execution.
Validation rejects incompatible shapes, types, or bounds; it does not clip,
round, or wrap invalid commands. Shape declarations alone cannot establish that
two coordinate systems or controllers are compatible.

## Run an episode with tools

This small example uses the existing experiment API. The oracle reads the grid's
privileged coordinates; it is a plumbing example, not a trained robotics result.

```python
from contextlib import contextmanager
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.streaming_helpers import EnvironmentSession
from brainscore.experiments import Experiment, SessionProtocol, RecordInputsOutputs
from brainscore.harnesses.grid_game import (
    GridGameEnv,
    GridGameEnvironment,
    greedy_oracle_policy,
)
from brainscore.model_helpers.policy_wrapper import PolicyWrapper

subject = BrainScoreModel(
    "grid-oracle",
    action_fn=PolicyWrapper(greedy_oracle_policy),
)

# A context manager closes the environment even if the trial fails.
@contextmanager
def session_factory(trial):
    with EnvironmentSession(
        GridGameEnvironment(GridGameEnv(size=4)),
        seed=trial,
        require_specs=True,
    ) as session:
        yield session

# The protocol resets the subject before and after each independent trial.
protocol = SessionProtocol(
    "grid-navigation",
    session_factory,
    trials=(5, 6),
    input_channels={"observation"},
    output_channels={"motor"},
)
result = Experiment(
    subject=subject,
    protocol=protocol,
    tools=[RecordInputsOutputs()],
    output_dir="grid-recording",  # Use a new directory for each experiment.
).run()
```

`play_game` and `play_gym_episode` use the same session driver. These convenience
functions leave subject reset to the caller. Use `SessionProtocol` when combining
independent trials and interventions so resets occur before tools attach.

## Read the record

- **Proposed:** the policy returned this command.
- **Applied:** `environment.step(action)` returned successfully. This confirms the
  environment accepted the call; it does not prove physical hardware reached a pose.
- **Rejected:** validation failed before calling the environment.
- **Execution unknown:** the environment call failed; it may have changed state.

These labels are in motor-event `meta['action_status']`. Final observations are
retained, without asking the policy for another action. `is_terminal` means a
natural ending; `is_last` without `is_terminal` means a truncated episode.

`replay_sessions(result.directory)` supplies saved observations to another run.
It does not restart the simulator. Its outputs are proposals, not applied actions.
Inference durations can differ on replay; compare action values when checking
policy agreement, rather than expecting identical timing metadata.

## Time, reset, and action chunks

The environment owns trajectory time. Set `context['t_ms']`, `time_source`, and
`time_inferred` when a physical or simulated time scale exists. Without one,
`t_ms` is `None` and `step_num` preserves order. `latency_ms` measures the policy
call in wall time. Native sessions wait for the policy; they enforce no real-time
deadline.

`seed` and `options` are forwarded to reset only when supplied. Unsupported reset
arguments fail before starting. Record initial-state and policy RNG controls as
well: a seed alone does not guarantee reproduction of a remote policy.

A policy can predict a chunk of future actions. The environment receives one
scheduled command per motor event. Record `prediction_horizon`,
`execution_horizon`, `scheduler`, and, where available, `chunk_id`, `chunk_index`,
and `control_period_ms` in response metadata.

`DroidPolicy` schedules its own queue. `LiberoChunkPolicy` returns the full chunk
and leaves scheduling to the reference evaluator. Pass the evaluator's actual
`execution_horizon` to the adapter; `None` means it has not been declared.
The adapter does not verify what the external evaluator actually executes.
A chunk must not be submitted directly to a native environment session.

## Qualification and scientific targets

The [DROID guide](droid_integration.md) describes recorded-policy evaluation.
The release gate requires trained-policy agreement between direct and UMI calls,
plus recording, replay, cleanup, action semantics, and time provenance. These
session tests do not satisfy that trained-policy gate.

Task success, demonstration agreement, movement similarity, and neural similarity
are different measurements. Each benchmark declares its target and metric.
Generic action specifications do not establish support for a new robot.

For externally managed evaluation, see the [LIBERO integration](robotics_benchmark_integration.md).
