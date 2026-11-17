"""A2: activation extraction must bypass result_caching while a perturbation is
active, so a lesioned run never returns cached UNperturbed activations."""
import os

import pytest

from brainscore._model.dispatch import extract_for_modality


@pytest.fixture(autouse=True)
def preserve_cache_environment():
    """Keep cache-control tests from changing later suites' environment."""
    previous = os.environ.get('RESULTCACHING_DISABLE')
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('RESULTCACHING_DISABLE', None)
        else:
            os.environ['RESULTCACHING_DISABLE'] = previous


class _FakeActivationsModel:
    """Records whether the result cache was disabled during the call."""
    def __init__(self):
        self.seen_disable = 'unset'

    def __call__(self, stimuli, layers):
        self.seen_disable = os.environ.get('RESULTCACHING_DISABLE')
        return 'features'


class _FakeOwner:
    def __init__(self, perturbed):
        self.vision_extractor = _FakeActivationsModel()
        self._extractors = {'vision': self.vision_extractor}
        self._shared_calls = None
        from types import SimpleNamespace
        self._interventions = SimpleNamespace(
            active_interventions={'h1': (lambda: None)} if perturbed else {})


def _clean_env():
    os.environ.pop('RESULTCACHING_DISABLE', None)


def test_cache_bypassed_while_perturbed():
    _clean_env()
    owner = _FakeOwner(perturbed=True)
    out = extract_for_modality(owner, 'stim', 'vision', ['layer'])
    assert out == 'features'
    # cache was disabled DURING the perturbed extraction
    assert owner.vision_extractor.seen_disable == '1'
    # and restored (removed) afterwards
    assert os.environ.get('RESULTCACHING_DISABLE') is None


def test_cache_untouched_when_not_perturbed():
    _clean_env()
    owner = _FakeOwner(perturbed=False)
    extract_for_modality(owner, 'stim', 'vision', ['layer'])
    # normal path: cache flag never set
    assert owner.vision_extractor.seen_disable is None
    assert os.environ.get('RESULTCACHING_DISABLE') is None


def test_prior_disable_flag_is_restored():
    os.environ['RESULTCACHING_DISABLE'] = 'some.module'
    try:
        owner = _FakeOwner(perturbed=True)
        extract_for_modality(owner, 'stim', 'vision', ['layer'])
        assert owner.vision_extractor.seen_disable == '1'
        assert os.environ.get('RESULTCACHING_DISABLE') == 'some.module'  # restored
    finally:
        _clean_env()
