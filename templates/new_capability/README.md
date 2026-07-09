# Template: a new capability

This template configures an **existing UMI capability** for one model. Write a
callable and pass it to a `BrainScoreModel` constructor slot; `process()`
reaches it through UMI's existing framework capability registry. You do not
register the callable itself.

This is distinct from adding a new framework capability. That core-level task
subclasses `brainscore_core.capabilities.Capability` and calls
`register_capability`; it changes Python source and is not what this template
does. See `EXTENDING.md` (Seam 4).

| Slot | Fires on | Returns |
|------|----------|---------|
| `generation_fn` | `StimulusSet` + `TaskContext.instruction` | a label `str` |
| `action_fn` | `EnvironmentStep` / `Message` | `EnvironmentResponse` |
| `state_change_fn` | `StateChange` | `(PerturbationApplied, cleanup)` |

## Use it

1. Copy `capability.py`, keep the builder for the slot you need, delete the rest.
2. Implement the closure to satisfy that slot's contract.
3. Pass it into a `BrainScoreModel(...)` in your model registration (see `templates/new_model/`).
4. Adapt `test_capability.py` and run it — assert it dispatches through `process()`.

## The boundary

Filling an existing slot needs **no core change**. A genuinely new input/output
event or dispatch path is a core framework contribution and belongs in the
class-based capability registry instead of this callable template.

## Worked references (read these)

- `brainscore/model_helpers/api_behavioral.py` — `build_api_generation_fn` + `build_api_action_fn`.
- `brainscore/perturbation.py` — `build_pytorch_ablation_fn` (a real `state_change_fn`).
- `brainscore/harnesses/grid_game.py` — drives `action_fn` through a closed loop.
