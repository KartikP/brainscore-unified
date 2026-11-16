"""A session that requests only probabilities runs only the part that gives them:
no part writes an answer that no output requests (ADR-0063 decision 1)."""
import numpy as np
import pytest

from brainscore import BrainScoreModel
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

STIMULI = StimulusSet({'stimulus_id': ['a', 'b'], 'sentence': ['real x', 'fake y']})
STIMULI.identifier = 'passes-follow-formats'
FITTING = StimulusSet({'stimulus_id': ['f0', 'f1', 'f2', 'f3'],
                       'sentence': ['real 0', 'fake 1', 'real 2', 'fake 3'],
                       'label': ['cat', 'dog', 'cat', 'dog']})
FITTING.identifier = 'passes-follow-formats-fit'


def features(stimuli, layers, **kwargs):
    values = np.array([[2.0, 0.0] if 'real' in text else [0.0, 2.0] for text in stimuli['sentence']])
    return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
        'neuroid_id': ('neuroid', ['n0', 'n1']), 'layer': ('neuroid', [layers[0]] * 2)})


def subject(calls, statement, likelihood=False):
    def answer(stimulus_row, instruction, label_set):
        calls.append('answer')
        return 'cat'

    def score(prompt, candidates):
        calls.append('likelihood')
        return {'cat': -1.0, 'dog': -2.0}
    parts = {'answer': answer, **({'likelihood': score} if likelihood else {})}
    return BrainScoreModel('passes', extractor={'text': features}, behavioral_readout_layer='late',
                           capability_config={'behavior-sessions': statement}, **parts)


@pytest.mark.parametrize('statement, likelihood, owner', [
    ({'behavior': 'generation'}, False, None),                                   # readout gives them
    ({'behavior': 'generation', 'behavior:probabilities': ('likelihood', 'softmax')}, True, 'likelihood'),
])
def test_only_the_owner_of_the_probabilities_runs_when_only_they_are_requested(statement, likelihood, owner):
    calls = []
    session = stimulus_session(STIMULI, ['behavior:probabilities'],
                               TaskContext(response=Choice(['cat', 'dog']), fitting_stimuli=FITTING))
    subject(calls, statement, likelihood).participate(session)
    assert 'answer' not in calls
    assert calls == ([owner] * len(STIMULI) if owner else [])
    assert set(session.collect('behavior:probabilities')['response_state'].values) == {'valid'}


def test_a_requested_choice_still_runs_the_answer_part():
    calls = []
    session = stimulus_session(STIMULI, ['behavior:choice', 'behavior:probabilities'],
                               TaskContext(response=Choice(['cat', 'dog']), fitting_stimuli=FITTING))
    subject(calls, {'behavior': 'generation'}).participate(session)
    assert calls.count('answer') == len(STIMULI)
