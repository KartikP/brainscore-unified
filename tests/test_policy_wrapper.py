"""Tests for the closed-loop PolicyWrapper (stateful embodied dispatch)."""
from brainscore.model_helpers.policy_wrapper import PolicyWrapper
from brainscore import BrainScoreModel
from brainscore_core.streaming import (
    Session, StreamEvent, begin_event, end_event, respond_event,
)


def _row(**observation):
    return {"observation": observation}


class TestPolicyWrapper:
    def test_stateless_when_history_disabled(self):
        pw = PolicyWrapper(lambda obs, hist: obs["v"] * 2, max_history=0)
        r = pw(_row(v=3))
        assert r == {"motor": 6}
        pw(_row(v=5))
        assert pw.history == ()  # history disabled

    def test_history_accumulates_and_caps(self):
        seen = []
        def policy(obs, hist):
            seen.append(len(hist))
            return obs["v"]
        pw = PolicyWrapper(policy, max_history=4)
        for v in range(5):
            pw(_row(v=v))
        assert seen == [0, 1, 2, 3, 4]   # grows, last call sees 4 (capped at max_history)
        assert len(pw.history) == 4

    def test_explicit_reset(self):
        pw = PolicyWrapper(lambda obs, hist: len(hist), max_history=8)
        pw(_row())
        pw(_row())
        pw.reset()
        assert pw(_row())["motor"] == 0

    def test_returns_the_action_on_the_chosen_channel(self):
        pw = PolicyWrapper(lambda obs, hist: "LEFT", max_history=0, channel="text")
        assert pw(_row()) == {"text": "LEFT"}

    def test_plugs_into_brainscoremodel_policy_and_clears_when_an_episode_begins(self):
        pw = PolicyWrapper(lambda obs, hist: obs["n"] + len(hist), max_history=8)
        model = BrainScoreModel(
            identifier="stateful-policy",
            model=None,
            region_layer_map={},
            preprocessors={"vision": lambda x: x},
            policy=pw,
        )
        obs = StreamEvent("observation", {"n": 10}, None)
        episode = [begin_event(), obs, respond_event(), obs, respond_event(), end_event("truncated")]
        session = Session(episode + episode, ["motor"])
        model.participate(session)
        # 10 + 0 prior steps, 10 + 1 prior step (state carried), then begin clears it.
        assert [event.payload for event in session.emitted] == [10, 11, 10, 11]
