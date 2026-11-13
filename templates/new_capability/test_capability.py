"""Tests are the spec: the capability registers once, only opted-in models gain
its channels, wrong payloads are rejected, and a session runs through
subject.participate. Adapt them as you replace the synthetic word counter.
"""
import importlib.util
from pathlib import Path

import pytest

from brainscore_core import io_catalog
from brainscore.capabilities import capability_registry
from brainscore import BrainScoreModel
from brainscore_core.streaming import Session, StreamEvent


@pytest.fixture
def capability(monkeypatch):
    """The template module with its registrations isolated to this test."""
    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    monkeypatch.setattr(io_catalog, '_STIMULUS_COLUMNS', dict(io_catalog._STIMULUS_COLUMNS))
    previous = dict(capability_registry)
    path = Path(__file__).with_name('capability.py')
    spec = importlib.util.spec_from_file_location('word_count_capability', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)   # importing alone registers nothing
    assert module.IDENTIFIER not in capability_registry
    module.register()
    yield module
    capability_registry.clear()
    capability_registry.update(previous)


def _requests(module, *texts):
    return [StreamEvent(channel=module.REQUEST, payload=module.WordRequest(text=text),
                        t_ms=float(index), meta={'stimulus_id': f'request{index}'})
            for index, text in enumerate(texts)]


def test_session_runs_through_participate(capability):
    subject = capability.build_model(min_length=2)
    session = Session(_requests(capability, 'a big red ball', 'go on'), [capability.RESULT])
    subject.participate(session)
    answers = session.collect(capability.RESULT)
    assert [event.payload for event in answers] == [
        capability.WordCount(words=3, words_so_far=3),
        capability.WordCount(words=2, words_so_far=5),
    ]
    assert [event.meta['stimulus_id'] for event in answers] == ['request0', 'request1']


def test_reset_clears_state_and_the_model_is_reusable(capability):
    subject = capability.build_model()
    for _ in range(2):   # the second pass starts from fresh state after reset
        session = Session(_requests(capability, 'one two'), [capability.RESULT])
        subject.participate(session)
        assert session.collect(capability.RESULT)[0].payload.words_so_far == 2
        subject.reset()


def test_only_opted_in_models_gain_the_channels(capability):
    assert capability.RESULT in capability.build_model().out_channels
    assert capability.REQUEST in capability.build_model().in_channels
    other = BrainScoreModel(identifier='other-model')
    assert capability.RESULT not in other.out_channels
    assert capability.REQUEST not in other.in_channels


def test_wrong_payloads_are_rejected(capability):
    # Wrong type, or the right dataclass holding a wrong field: rejected up front,
    # before any model sees the request.
    for payload in ('not a request', capability.WordRequest(text=42)):
        with pytest.raises(ValueError):
            Session([StreamEvent(channel=capability.REQUEST, payload=payload, t_ms=0.0)], [capability.RESULT])


def test_bad_settings_fail_before_counting(capability):
    with pytest.raises(ValueError, match='min_length'):
        capability.build_model(min_length=0)
    # A model built directly, bypassing build_model, is checked in setup.
    subject = BrainScoreModel(identifier='direct',
                              capability_config={capability.IDENTIFIER: {'min_length': True}})
    session = Session(_requests(capability, 'one two'), [capability.RESULT])
    with pytest.raises(ValueError, match='min_length'):
        subject.participate(session)
    assert session.next_input() is not None   # no request was consumed


def test_registering_twice_is_an_error(capability):
    with pytest.raises(ValueError, match='already registered'):
        capability.register()
