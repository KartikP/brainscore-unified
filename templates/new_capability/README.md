# Template: a new capability

There are two ways to give a model a new ability. Neither edits Brain-Score's core.

## 1. Fill an existing slot

The built-in abilities are `BrainScoreModel` constructor slots. Pass a function:

| Slot | Runs on | Signature |
|------|---------|-----------|
| `extractor` | stimuli, for recording | `{modality: wrapper}`; each wrapper takes `(stimuli, layers=...)` |
| `answer` | a `StimulusSet` + `TaskContext.instruction` | `(stimulus_row, instruction, label_set) -> str` |
| `policy` | an `EnvironmentStep` / `Message` | `(env_step) -> EnvironmentResponse` |
| `intervention` | a `StateChange` | `(state_change) -> (PerturbationApplied, cleanup)` |

```python
from brainscore_core.model_interface import BrainScoreModel, EnvironmentResponse


def build_policy():
    def policy(env_step):
        # TODO: look at env_step.observation and choose an action.
        return EnvironmentResponse(action=0)
    return policy


model = BrainScoreModel(
    identifier='your-agent',
    policy=build_policy(),
)
```

Ready-made builders return the function you pass to a slot:
`build_api_generation_fn(...)` returns an `answer`, `build_api_action_fn(...)`
a `policy` (both in `brainscore/model_helpers/api_behavioral.py`), and
`build_pytorch_ablation_fn(model)` an `intervention`
(`brainscore/perturbation.py`). `brainscore/harnesses/grid_game.py` runs
policies in a closed loop.

## 2. Register a new capability

For a new kind of request, with its own input and output channels, copy
`capability.py`. It is a small runnable, clearly synthetic example: typed
dataclass payloads, two catalog entries whose validators reject bad payloads
before any model sees them, and a `Capability` subclass. Its `process` answers
one request; its `participate` sends each answer back with `session.respond`;
its `setup` checks the model's settings before the first request.

1. Rename the identifier, channels and dataclasses; replace the counting in
   `process` and the field checks in the two validators.
2. Call `register()` once from your package. A second call raises an error
   rather than replacing a registered name.
3. Opt a model in through `capability_config={IDENTIFIER: settings}`. Other
   models do not gain the channels.
4. Adapt `test_capability.py` and run it. It runs a `BufferedSession` through
   `subject.participate(session)`.

See the [tool-authoring guide](../../docs/tool_authoring.md) for channel
registration details.
