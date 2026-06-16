# Template: a new capability

A capability is a callable you pass to a `BrainScoreModel` constructor slot; `process()`
dispatches to it by input-event type. **There is no registry for capabilities** — you write the
closure and wire it into a model registration. See `EXTENDING.md` (Seam 4).

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

Filling an existing slot needs **no core change** — that's the point. A genuinely new input or
output *type* (a new `InputEvent`/`OutputEvent` member) is the only thing that touches the core
dataclasses + `process()` dispatch; that's a core contribution, not a plugin.

## Worked references (read these)

- `brainscore/model_helpers/api_behavioral.py` — `build_api_generation_fn` + `build_api_action_fn`.
- `brainscore/perturbation.py` — `build_pytorch_ablation_fn` (a real `state_change_fn`).
- `brainscore/harnesses/grid_game.py` — drives `action_fn` through a closed loop.
