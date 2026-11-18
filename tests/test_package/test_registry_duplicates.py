"""A second registration under an existing key raises and names both sources."""
import pytest

import brainscore


@pytest.mark.parametrize('registry', ['model_registry', 'benchmark_registry', 'metric_registry',
                                      'data_registry', 'stimulus_set_registry',
                                      'data_builder_registry'])
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


def _factory_from(module_name, qualname='get_model'):
    def factory():
        return module_name
    factory.__module__ = module_name
    factory.__qualname__ = qualname
    return factory


def test_the_same_source_registered_again_replaces_quietly():
    registry = brainscore.model_registry
    registry['reload-check'] = _factory_from('plugin_a')
    try:
        second = _factory_from('plugin_a')
        registry['reload-check'] = second              # a module imported a second time
        registry.update({'reload-check': _factory_from('plugin_a')})
        registry.setdefault('reload-check', _factory_from('plugin_b'))   # key held: no-op
        assert registry['reload-check']() == 'plugin_a'
    finally:
        del registry['reload-check']


def test_a_plugin_module_that_is_reloaded_does_not_raise():
    import importlib
    import sys
    import types
    module = types.ModuleType('reload_plugin_probe')
    exec("import brainscore\n"
         "def get_model():\n    return 1\n"
         "brainscore.model_registry['reload-probe'] = get_model\n", module.__dict__)
    registry = brainscore.model_registry
    try:
        exec("import brainscore\n"
             "def get_model():\n    return 1\n"
             "brainscore.model_registry['reload-probe'] = get_model\n", module.__dict__)
        assert registry['reload-probe']() == 1
    finally:
        del registry['reload-probe']


def test_update_with_an_existing_key_from_another_source_raises():
    registry = brainscore.benchmark_registry
    registry['update-check'] = _factory_from('plugin_a')
    try:
        with pytest.raises(ValueError, match=r"already registered \(by plugin_a\.get_model\).*plugin_b"):
            registry.update({'update-check': _factory_from('plugin_b')})
        with pytest.raises(ValueError, match='already registered'):
            registry |= {'update-check': _factory_from('plugin_b')}
        assert registry['update-check']() == 'plugin_a'
    finally:
        del registry['update-check']
