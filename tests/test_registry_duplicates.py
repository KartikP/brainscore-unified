"""A second registration under an existing key raises and names both sources."""
import pytest

import brainscore


@pytest.mark.parametrize('registry', ['model_registry', 'benchmark_registry', 'metric_registry',
                                      'data_registry', 'stimulus_set_registry'])
def test_a_second_registration_raises_and_names_both_sources(registry):
    registry = getattr(brainscore, registry)
    registry['duplicate-check'] = lambda: 'first'
    try:
        with pytest.raises(ValueError, match=r"already registered \(by .*test_registry_duplicates"
                                             r"\.py:\d+\).*second registration \(by .*"
                                             r"test_registry_duplicates\.py:\d+\)"):
            registry['duplicate-check'] = lambda: 'second'
        assert registry['duplicate-check']() == 'first'
    finally:
        del registry['duplicate-check']
    registry['duplicate-check'] = lambda: 'after removal'      # removal frees the key
    del registry['duplicate-check']
