"""TODO: which capability are you adding behavior for? Capabilities are callables you
pass to a BrainScoreModel slot; process() dispatches to them by input-event type. You do
NOT register a capability — you write the closure and wire it into a model registration.

Pick the slot that matches the input event you want the model to handle:
  - generation_fn : StimulusSet + TaskContext.instruction  -> a text label
  - action_fn     : EnvironmentStep / Message              -> an EnvironmentResponse
  - state_change_fn: StateChange                            -> (PerturbationApplied, cleanup)

A genuinely NEW input/output event type (not one of the above) is the only case that touches
the core dataclasses + process() dispatch in core/brainscore_core/model_interface.py.
"""
from brainscore_core.model_interface import EnvironmentResponse, PerturbationApplied


def build_generation_fn(**cfg):
    """Behavioral generation: answer a forced-choice / instruction task as text."""
    def generate(stimulus_row, instruction, label_set) -> str:
        # TODO: read the stimulus (image path / text) from stimulus_row, ask the model,
        # parse a label_set member. Return the raw string on a parse miss (the caller warns
        # + falls back to label_set[0]). See model_helpers/api_behavioral.py for a full one.
        ...
    return generate


def build_action_fn(**cfg):
    """Embodied closed loop: see an observation, return one action."""
    def act(env_step) -> 'EnvironmentResponse':
        # obs = env_step.observation  (frame / ascii / legal_actions / instruction)
        # TODO: choose an action index. MUST return EnvironmentResponse(action=...).
        return EnvironmentResponse(action=...)
    return act


def build_state_change_fn(model, **cfg):
    """Lesion/perturbation: apply a StateChange, return how to undo it."""
    def state_change_fn(state_change):
        # TODO: resolve state_change.target (which units) + apply state_change.perturbation
        # (zero/scale/replace). Return (PerturbationApplied(handle_id=...), cleanup_callable).
        # See unified/brainscore/perturbation.py::build_pytorch_ablation_fn for a real one.
        applied = PerturbationApplied(handle_id=...)
        def cleanup():
            ...
        return applied, cleanup
    return state_change_fn


# --- wire it into a model (capabilities have no registry; they live on the model) ---
# from brainscore_core.model_interface import BrainScoreModel
# model = BrainScoreModel(identifier='your-model', model=backbone,
#                         region_layer_map=..., preprocessors=...,
#                         action_fn=build_action_fn(...))   # <- the slot you filled
