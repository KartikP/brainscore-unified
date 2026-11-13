"""Contract Rule 2.4: declarations come from what is wired up, and a declared channel works.

A subject that declares a channel runs a session that uses it, or the session
fails before any compute. These tests build a BrainScoreModel with every part,
then run a session for each output it declares.
"""
import numpy as np
import pandas as pd
import pytest

from brainscore import BrainScoreModel
from brainscore.harnesses.grid_game import GridGameEnv, GridGameEnvironment
from brainscore_core.compatibility import check_compatibility
from brainscore_core.contract import TaskContext
from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import apply_state_change, run_environment, stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


def _table(ids, labels=None):
    data = {'stimulus_id': ids, 'image_path': [f'/tmp/{i}.png' for i in ids]}
    if labels is not None:
        data['image_label'] = labels
    stimuli = StimulusSet(pd.DataFrame(data))
    stimuli.identifier = '-'.join(ids)
    return stimuli


STIMULI = _table(['a', 'b'])
FITTING = _table(['f0', 'f1', 'f2', 'f3'], ['cat', 'dog', 'cat', 'dog'])


def _model(calls):
    """Every part works and notes each call in ``calls``."""
    def extractor(stimuli, layers=None, **kwargs):
        calls.append('extractor')
        ids = list(stimuli['stimulus_id'])
        values = np.array([[float(i % 2), 1.0 - i % 2] for i in range(len(ids))])
        return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', [0, 1]),
            'layer': ('neuroid', [layers[0]] * 2)})

    def policy(row):
        calls.append('policy')
        return {'motor': 3}

    def intervention(change):
        calls.append('intervention')
        return PerturbationApplied('h', change.target, change.perturbation), (lambda: None)

    return BrainScoreModel('all-parts', region_layer_map={'IT': 'layer'},
                           extractor={'vision': extractor}, behavioral_readout_layer='layer',
                           policy=policy, intervention=intervention)


class _Benchmark:
    def __init__(self, output):
        self.identifier = output
        self.requested_output_channels = {output}
        self.required_input_channels = {'vision:image'}


def test_every_part_declares_registered_channels():
    model = _model([])
    assert model.out_channels == {'neural:IT', 'behavior', 'behavior:choice',
                                  'behavior:probabilities', 'behavior:rating', 'behavior:rt',
                                  'motor', 'perturbation'}
    assert {'vision:image', 'observation', 'lesion', 'stimulation'} <= model.in_channels
    for output in sorted(model.out_channels):
        check_compatibility(model, _Benchmark(output))  # registered and produced


def test_a_part_that_is_not_wired_declares_nothing():
    bare = BrainScoreModel('bare', region_layer_map={'IT': 'layer'})
    assert not bare.out_channels & {'neural:IT', 'behavior', 'motor', 'perturbation'}


def test_every_declared_output_answers_or_fails_before_any_compute():
    """A session that brings no task or change: the subject answers, or refuses before work."""
    calls = []
    model = _model(calls)
    for output in sorted(model.out_channels):
        calls.clear()
        session = stimulus_session(STIMULI, [output])
        try:
            model.participate(session)
        except Exception:                                   # noqa: BLE001
            assert calls == [], f'{output}: failed after {calls}'
            assert session.emitted == [], output
        else:
            assert {event.channel for event in session.emitted} <= {output}, output


@pytest.mark.parametrize('output, task', [
    ('neural:IT', None),
    ('behavior', TaskContext(task_type='probabilities', fitting_stimuli=FITTING)),
    ('behavior:choice', TaskContext(response=Choice(['cat', 'dog']), fitting_stimuli=FITTING)),
    ('behavior:probabilities', TaskContext(response=Choice(['cat', 'dog'], probabilities=True),
                                           fitting_stimuli=FITTING)),
])
def test_a_declared_stimulus_output_runs_with_the_session_it_needs(output, task):
    calls = []
    session = stimulus_session(STIMULI, [output], task)
    _model(calls).participate(session)
    assert 'extractor' in calls
    answer = session.collect(output)
    assert answer.sizes['presentation'] == len(STIMULI)


def test_a_declared_perturbation_runs_through_its_session():
    calls = []
    applied = apply_state_change(_model(calls), StateChange(
        'ablation', target=Selection('layer'), perturbation=Perturbation('zero')))
    assert calls == ['intervention'] and applied.handle_id == 'h'


def test_a_declared_motor_output_runs_an_environment():
    calls = []
    trajectory = run_environment(_model(calls), GridGameEnvironment(GridGameEnv()))
    assert calls and set(calls) == {'policy'}
    assert len(trajectory) == len(calls)
