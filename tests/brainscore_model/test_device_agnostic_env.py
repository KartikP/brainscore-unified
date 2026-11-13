"""Core is device-agnostic: a closed-loop row carries harness-defined payloads.

A row is a mapping of channel -> payload. The policy reads it and returns a
mapping of output channel -> payload; core never inspects either.
"""
import numpy as np

from brainscore import BrainScoreModel


class TestDeviceAgnosticRow:
    def test_generic_observation_row(self):
        row = {"observation": {"frame": np.zeros((84, 84, 3)), "score": 0},
               "instruction": "play breakout"}
        model = BrainScoreModel(identifier="generic", policy=lambda r: {"motor": r["observation"]["score"]})
        assert model.process(row) == {"motor": 0}

    def test_action_payload_is_harness_defined(self):
        # A non-DROID action payload (e.g. a discrete game action) is fine.
        model = BrainScoreModel(identifier="discrete", policy=lambda r: {"motor": "LEFT"})
        assert model.process({"observation": {}}) == {"motor": "LEFT"}
        model = BrainScoreModel(identifier="vector", policy=lambda r: {"motor": np.zeros(7)})
        assert model.process({"observation": {}})["motor"].shape == (7,)

    def test_policy_dispatch_reads_observation(self):
        def policy(row):
            return {"motor": row["observation"]["n"] * 2}

        model = BrainScoreModel(
            identifier="generic-policy",
            model=None,
            region_layer_map={},
            supported_modalities={"vision"},
            policy=policy,
        )
        assert model.process({"observation": {"n": 21}}) == {"motor": 42}
