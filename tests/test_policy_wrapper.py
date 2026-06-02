"""Tests for the v1.5 closed-loop PolicyWrapper (stateful embodied dispatch)."""
from brainscore.model_helpers.policy_wrapper import PolicyWrapper
from brainscore_core.model_interface import (
    BrainScoreModel,
    EnvironmentResponse,
    EnvironmentStep,
)


class TestPolicyWrapper:
    def test_stateless_when_history_disabled(self):
        pw = PolicyWrapper(lambda obs, hist: obs["v"] * 2, max_history=0)
        r = pw(EnvironmentStep(observation={"v": 3}))
        assert isinstance(r, EnvironmentResponse) and r.action == 6
        pw(EnvironmentStep(observation={"v": 5}))
        assert pw.history == ()  # history disabled

    def test_history_accumulates_and_caps(self):
        seen = []
        def policy(obs, hist):
            seen.append(len(hist))
            return obs["v"]
        pw = PolicyWrapper(policy, max_history=4)
        for v in range(5):
            pw(EnvironmentStep(observation={"v": v}, is_first=(v == 0)))
        assert seen == [0, 1, 2, 3, 4]   # grows, last call sees 4 (capped at max_history)
        assert len(pw.history) == 4

    def test_reset_on_is_first(self):
        pw = PolicyWrapper(lambda obs, hist: len(hist), max_history=8)
        pw(EnvironmentStep(observation={}, is_first=True))  # cleared -> sees 0
        pw(EnvironmentStep(observation={}))                 # sees 1
        r = pw(EnvironmentStep(observation={}, is_first=True))  # cleared -> sees 0
        assert r.action == 0

    def test_explicit_reset(self):
        pw = PolicyWrapper(lambda obs, hist: len(hist), max_history=8)
        pw(EnvironmentStep(observation={}))
        pw(EnvironmentStep(observation={}))
        pw.reset()
        assert pw(EnvironmentStep(observation={})).action == 0

    def test_wraps_raw_action_in_response(self):
        pw = PolicyWrapper(lambda obs, hist: "LEFT", max_history=0)
        r = pw(EnvironmentStep(observation={}))
        assert isinstance(r, EnvironmentResponse) and r.action == "LEFT"

    def test_plugs_into_brainscoremodel_action_fn(self):
        pw = PolicyWrapper(lambda obs, hist: obs["n"] + len(hist), max_history=8)
        model = BrainScoreModel(
            identifier="stateful-policy",
            model=None,
            region_layer_map={},
            preprocessors={"vision": lambda x: x},
            action_fn=pw,
        )
        r0 = model.process(EnvironmentStep(observation={"n": 10}, is_first=True))
        assert r0.action == 10  # 10 + 0 prior steps
        r1 = model.process(EnvironmentStep(observation={"n": 10}))
        assert r1.action == 11  # 10 + 1 prior step (closed-loop state carried)
