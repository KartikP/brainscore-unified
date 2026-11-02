"""Null subjects declare what they produce (Contract Rules v3, rule 2)."""
from brainscore_core.compatibility import check_compatibility

from brainscore.models.chance_baseline.model import ChanceBaseline


class _BehavioralBenchmark:
    identifier = 'behavioral'
    required_modalities = {'vision'}
    requested_output_channels = {'behavior'}


def test_chance_baseline_declares_behavior_and_passes_behavioral_preflight():
    model = ChanceBaseline()
    assert model.out_channels == {'behavior'}
    assert check_compatibility(model, _BehavioralBenchmark()).kind == 'full'
