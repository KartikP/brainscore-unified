"""Check ROAR registrations without loading benchmark data or model weights."""
import pytest


@pytest.mark.parametrize('identifier', [
    'Yeatman2021-lexical_decision-image',
    'Yeatman2021-lexical_decision-text',
    'Yeatman2021-lexical_decision',
])
def test_roar_variant_registered(identifier):
    import brainscore

    assert identifier in brainscore.benchmark_registry
