"""Cache contract tests: only test-created temporary entries are ever enabled."""
import inspect
import logging

import numpy as np
import pytest
import torch
import result_caching

from brainscore_core.extraction_cache import extraction_fingerprint, fingerprint
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
from brainscore.model_helpers.video_wrapper import VideoWrapper
from brainscore.model_helpers.audio_wrapper import AudioWrapper
from tests.test_umi_review_regressions import language_pair, _stimuli


class Settings:
    def __init__(self, value=1):
        self.value = value

    def cache_config(self):
        return {'value': self.value}


def _wrapper(cls):
    wrapper = cls.__new__(cls)
    wrapper._model = torch.nn.Linear(2, 2).eval()
    wrapper._identifier = wrapper._backbone_id = 'cache-fixture'
    for name in cls.CACHE_FIELDS:
        setattr(wrapper, name, 1)
    for name in ('_processor', '_tokenizer'):
        if name in cls.CACHE_FIELDS:
            setattr(wrapper, name, Settings())
    return wrapper


def _storage(method):
    return next(cell.cell_contents for cell in method.__func__.__closure__
                if isinstance(cell.cell_contents, result_caching._XarrayStorage))


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    # Keep RESULTCACHING_DISABLE=1. Only our sentinel model's temporary cache
    # is enabled, to exercise the real result_caching read/merge/write code.
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    def enable(identifier):
        return 'identifier=cache-fixture' in identifier
    monkeypatch.setattr(result_caching, 'is_enabled', enable)
    monkeypatch.setattr('brainscore_core.extraction_cache.is_enabled', enable)
    def install(method):
        storage = _storage(method)
        monkeypatch.setattr(storage, '_storage_directory', str(tmp_path))
        return storage
    return install


def _assembly(inputs, layers, value):
    return NeuroidAssembly(np.full((len(inputs), len(layers)), value, dtype=float),
        dims=['presentation', 'neuroid'], coords={
            'stimulus_path': ('presentation', inputs),
            'layer': ('neuroid', layers), 'neuroid_id': ('neuroid', layers)})


@pytest.mark.parametrize('cls', [TextWrapper, VLMVisionWrapper, VideoWrapper, AudioWrapper])
def test_same_configuration_hits_changed_configuration_misses(cls, monkeypatch, isolated_cache):
    wrapper = _wrapper(cls)
    inp = 'texts' if cls is TextWrapper else 'paths'
    method = getattr(wrapper, f'_from_{inp}_cached')
    isolated_cache(getattr(wrapper, f'_from_{inp}_stored'))
    calls = []
    def extract(inputs, layers, stimuli_identifier=None):
        calls.append(list(layers))
        return _assembly(inputs, layers, wrapper._batch_size)
    monkeypatch.setattr(wrapper, f'_from_{inp}', extract)
    expected = method(['a', 'b'], ['L'], 'stimuli')
    np.testing.assert_array_equal(method(['a', 'b'], ['L'], 'stimuli'), expected)
    assert calls == [['L']]
    # Additional layers are merged without recomputing L.
    combined = method(['a', 'b'], ['L', 'M'], 'stimuli')
    assert calls == [['L'], ['M']]
    np.testing.assert_array_equal(combined.sel(layer='L'), expected.sel(layer='L'))
    wrapper._batch_size = 2
    np.testing.assert_array_equal(method(['a', 'b'], ['L'], 'stimuli'), 2)
    assert len(calls) == 3
    wrapper._batch_size = 1
    np.testing.assert_array_equal(method(['a', 'b'], ['L'], 'stimuli'), expected)
    assert len(calls) == 3
    # Separate instance with identical config and the same registered weights.
    other = _wrapper(cls)
    monkeypatch.setattr(other, f'_from_{inp}', lambda *a, **k: pytest.fail('cache missed'))
    np.testing.assert_array_equal(getattr(other, f'_from_{inp}_cached')(['a', 'b'], ['L'], 'stimuli'), expected)


@pytest.mark.parametrize('cls', [TextWrapper, VLMVisionWrapper, VideoWrapper, AudioWrapper])
def test_every_declared_extraction_setting_participates(cls):
    wrapper = _wrapper(cls)
    original = fingerprint(wrapper.cache_config())
    ignored = {'self', 'model', 'identifier', 'backbone_id'}
    # A newly added constructor option cannot silently escape this contract.
    assert set(inspect.signature(cls).parameters) - ignored == {key[1:] for key in cls.CACHE_FIELDS}
    for name in cls.CACHE_FIELDS:
        old = getattr(wrapper, name)
        setattr(wrapper, name, Settings(2) if isinstance(old, Settings) else 2)
        assert fingerprint(wrapper.cache_config()) != original, name
        setattr(wrapper, name, old)
        assert fingerprint(wrapper.cache_config()) == original
    wrapper._model.double()
    assert fingerprint(wrapper.cache_config()) != original


def test_unversioned_entry_is_logged_and_left_untouched(monkeypatch, isolated_cache, caplog):
    wrapper = _wrapper(TextWrapper)
    method = wrapper._from_texts_stored
    storage = isolated_cache(method)
    function = next(cell.cell_contents for cell in method.__func__.__closure__ if inspect.isfunction(cell.cell_contents))
    args = storage.getcallargs(function, wrapper, 'cache-fixture', ['L'], 'stimuli', ['a'], 'placeholder')
    args.pop('extraction_fingerprint')
    legacy_key = result_caching._XarrayStorage.get_function_identifier(storage, function, args)
    storage.save(_assembly(['a'], ['L'], 99), legacy_key)
    from pathlib import Path
    old_path = Path(storage.storage_path(legacy_key))
    old_bytes = old_path.read_bytes()
    monkeypatch.setattr(wrapper, '_from_texts', lambda texts, layers, identifier: _assembly(texts, layers, 1))
    with caplog.at_level(logging.INFO):
        actual = wrapper._from_texts_cached(['a'], ['L'], 'stimuli')
    np.testing.assert_array_equal(actual, 1)
    assert 'Skipping unversioned activation cache' in caplog.text
    assert old_path.read_bytes() == old_bytes
    assert len(list(old_path.parent.glob('*.pkl'))) == 2


def test_real_text_cache_preserves_values_and_pereira_context(language_pair, isolated_cache, monkeypatch):
    native, _ = language_pair
    wrapper = native._preprocessors['text']
    wrapper._backbone_id = 'cache-fixture'
    isolated_cache(wrapper._from_texts_stored)
    calls = []
    extract = wrapper._from_texts
    def counted(*args, **kwargs):
        calls.append(1)
        return extract(*args, **kwargs)
    monkeypatch.setattr(wrapper, '_from_texts', counted)
    stimuli = _stimuli()
    stimuli['context_id'] = 'passage'
    native.start_recording('language_system')
    first = native.process(stimuli)
    np.testing.assert_array_equal(native.process(stimuli), first)
    assert len(calls) == 1  # tokenizer backend runtime padding cannot alter key
    # Compare against direct cold extraction, not another cache read.
    cold = extract(wrapper._extract_texts(stimuli), ['transformer.h.0'])
    np.testing.assert_array_equal(first, cold)
    wrapper._max_length = 3
    shortened = native.process(stimuli)
    assert len(calls) == 2 and not np.allclose(shortened, first)
    wrapper._tokenizer.truncation_side = 'right'
    right = native.process(stimuli)
    assert len(calls) == 3 and not np.allclose(right, shortened)
    stimuli['context_id'] = ['a', 'a', 'b', 'b']
    native.process(stimuli)
    assert len(calls) == 4


def test_legacy_extractor_hooks_preprocessing_and_dtype(monkeypatch, isolated_cache):
    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    wrapper = PytorchWrapper(torch.nn.Linear(2, 2), lambda x: x, identifier='cache-fixture')
    helper = wrapper._extractor
    isolated_cache(helper._from_paths_stored)
    calls = []
    def extract(layers, stimuli_paths, require_variance=False):
        calls.append(1)
        value = 2 if helper._batch_activations_hooks else 1
        return _assembly(stimuli_paths, layers, value)
    monkeypatch.setattr(helper, '_from_paths', extract)
    run = lambda: helper.from_paths(['a'], ['L'], 'stimuli')
    np.testing.assert_array_equal(run(), 1)
    np.testing.assert_array_equal(run(), 1)
    assert len(calls) == 1
    hook = helper.register_batch_activations_hook(lambda x: x * 2)
    np.testing.assert_array_equal(run(), 2)
    np.testing.assert_array_equal(run(), 2)
    assert len(calls) == 2
    hook.remove()
    np.testing.assert_array_equal(run(), 1)
    assert len(calls) == 2
    wrapper._model.double()
    run()
    assert len(calls) == 3
    helper.preprocess = lambda x: x + 1
    run()
    assert len(calls) == 4


def test_canonical_order_closures_and_opaque_state():
    assert fingerprint({'b': 2, 'a': 1}) == fingerprint({'a': 1, 'b': 2})
    def factory(scale):
        return lambda x: x * scale
    assert fingerprint(factory(2)) == fingerprint(factory(2))
    assert fingerprint(factory(2)) != fingerprint(factory(3))
    assert extraction_fingerprint({'opaque': object()}) is None


def test_layer_pca_configuration_and_lazy_state(monkeypatch):
    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
    from brainscore_vision.model_helpers.activations.pca import LayerPCA
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    wrapper = PytorchWrapper(torch.nn.Linear(2, 2), lambda x: x, identifier='cache-fixture')
    helper = wrapper._extractor
    before = fingerprint(helper.cache_config())
    handle = LayerPCA.hook(wrapper, n_components=1)
    pca = next(iter(helper._batch_activations_hooks.values()))
    installed = fingerprint(helper.cache_config())
    assert before != installed
    calls = []
    def fit(**kwargs):
        calls.append(kwargs['extraction_fingerprint'])
        return {'L': None}
    monkeypatch.setattr(pca, '_pcas', fit)
    pca._ensure_initialized(['L'])
    assert fingerprint(helper.cache_config()) == installed
    pca._ensure_initialized(['L'])
    assert len(calls) == 1
    pca._n_components = 2
    assert fingerprint(helper.cache_config()) != installed
    pca._ensure_initialized(['L'])
    assert len(calls) == 2 and calls[0] != calls[1]
    handle.disable()
    assert fingerprint(helper.cache_config()) == before
    handle.enable()
    assert fingerprint(helper.cache_config()) != before
    handle.remove()
    assert fingerprint(helper.cache_config()) == before


def test_pytorch_perturbation_hooks_participate_and_reset_restores_key():
    from brainscore.perturbation import build_pytorch_ablation_fn
    from brainscore_core.model_interface import StateChange, Selection, Perturbation
    from brainscore_core.extraction_cache import model_config
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    original = fingerprint(model_config(model))
    apply = build_pytorch_ablation_fn(model)
    _, cleanup = apply(StateChange('ablation', target=Selection('0', [0]), perturbation=Perturbation('zero')))
    try:
        changed = fingerprint(model_config(model))
        assert changed != original
        assert fingerprint(model_config(model)) == changed
        assert model(torch.ones(1, 2))[0, 0].item() == 0
    finally:
        cleanup()
    assert fingerprint(model_config(model)) == original


def test_unsupported_provider_bypasses_storage(monkeypatch):
    wrapper = _wrapper(TextWrapper)
    wrapper.cache_config = lambda: {'opaque': object()}
    monkeypatch.setattr(wrapper, '_from_texts_stored', lambda **kwargs: pytest.fail('unsafe cache lookup'))
    monkeypatch.setattr(wrapper, '_from_texts', lambda texts, layers, identifier: _assembly(texts, layers, 5))
    np.testing.assert_array_equal(wrapper._from_texts_cached(['a'], ['L'], 'stimuli'), 5)
