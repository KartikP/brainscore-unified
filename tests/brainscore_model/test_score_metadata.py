import numpy as np
import pytest

from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import behavioral_response

from brainscore_core.contract import TaskContext
from brainscore_core.legacy_facade import LegacyBenchmarkFacade
from brainscore import BrainScoreModel
from brainscore_core.response_formats import response_record
from brainscore_core.score_metadata import record_measures
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from .test_behavioral_readout import (
    _fake_preprocessor_returning,
    _make_image_stimulus_set,
)


# --- the response records of the answers a run's sessions accepted --------


def _methods(measures):
    return {record['method'] for record in measures.responses}


def _stimuli():
    return _make_image_stimulus_set(['cat'] * 3 + ['dog'] * 3, identifier='measure-test')


def _generation_model():
    return BrainScoreModel(
        identifier='test-vlm', model=None, region_layer_map={},
        extractor={'vision': lambda *a, **kw: None},
        answer=lambda stimulus_row, instruction, label_set: 'cat')


def _readout_model():
    features = np.random.default_rng(0).normal(size=(6, 8))
    return BrainScoreModel(
        identifier='test-readout', model=None, region_layer_map={},
        extractor={'vision': _fake_preprocessor_returning(features)},
        behavioral_readout_layer='some_layer',
        answer_owners={'behavior:choice': ('readout', 'argmax')})


_GENERATION_TASK = dict(response=Choice(['cat', 'dog']), instruction='What animal is this?')


@pytest.mark.parametrize('model_factory, task_kwargs, expected', [
    (_generation_model, _GENERATION_TASK, {'generation'}),
    (_readout_model, dict(response=Choice(['cat', 'dog'])), {'readout'}),
])
@pytest.mark.parametrize('route', ['session', 'facade'])
def test_brainscore_model_records_name_the_answering_method(model_factory, task_kwargs, expected, route):
    model = model_factory()
    task = TaskContext(fitting_stimuli=_stimuli(), **task_kwargs)
    with record_measures() as measures:
        if route == 'session':
            behavioral_response(model, _stimuli(), task)
        else:
            facade = LegacyBenchmarkFacade(model, visual_degrees=8)
            facade.start_task(task)
            facade.look_at(_stimuli())

    assert measures.responses, 'the run recorded no response record'
    assert _methods(measures) == expected


# --- direct sessions, nesting, mixed collection --------------------------

def test_direct_session_with_a_generation_model_is_recorded():
    """participate + collect, without behavioral_response or the facade."""

    model = _generation_model()
    task = TaskContext(fitting_stimuli=_stimuli(), **_GENERATION_TASK)
    with record_measures() as measures:
        session = stimulus_session(_stimuli(), ['behavior'], task)
        model.participate(session)
        session.collect('behavior')

    assert _methods(measures) == {'generation'}


def test_a_decoded_saved_generation_record_does_not_change_a_readout_stamp():
    """Events built but never accepted by a session (a decoded run record, a
    fixture, a copy) are not responses of this run."""
    model = _readout_model()
    task = TaskContext(fitting_stimuli=_stimuli(), response=Choice(['cat', 'dog']))
    saved = BehavioralAssembly(
        [[1.0, 0.0]], coords={'stimulus_id': ('presentation', ['s0']),
                              'choice': ['cat', 'dog']},
        dims=['presentation', 'choice'],
        attrs={'response': response_record('behavior', 'generation', parsed_by='model')})
    with record_measures() as measures:
        StreamEvent('behavior', saved, 0.0, meta={'decoded': True})   # read, not sent
        behavioral_response(model, _stimuli(), task)

    assert _methods(measures) == {'readout'}


# --- direct sessions (moved from core) ------------------------------------


@pytest.mark.parametrize('kind, expected', [('reply', {'generation'}),
                                            ('likelihood', {'likelihood'})])
def test_direct_response_task_sessions_are_recorded(kind, expected):
    from .test_task_responses import _ask, _model
    from brainscore_core.responses import Choice

    if kind == 'reply':
        model, _ = _model(replies={s: 'cat' for s in 'abcd'})
        task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
        channels = ['behavior:choice']
    else:
        model, _ = _model(scores={s: {'cat': -1.0, 'dog': -2.0} for s in 'abcd'})
        task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
        channels = ['behavior:choice', 'behavior:probabilities']
    with record_measures() as measures:
        session = _ask(model, task, channels)
        for channel in channels:
            session.collect(channel)

    assert _methods(measures) == expected
