"""ExperimentResult.trials() shows the state core stored for each trial and its reason,
from the finalized answers the runner saves after each session (step 9, chunk 9.5)."""
import pytest

from brainscore import BrainScoreModel
from brainscore_core import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.experiments import (Experiment, ExperimentResult, RecordBehavior,
                                    RecordInputsOutputs, TaskProtocol)
from brainscore.run_record import RunRecord

pytestmark = pytest.mark.unit

STIMULI = StimulusSet({'stimulus_id': ['cat', 'boat', 'unknown'],
                       'sentence': ['cat', 'boat', 'ambiguous']})
STIMULI.identifier = 'trial-states'


def _run(response, replies, directory, conditions=('baseline',), tools=None):
    def reply(prompt, _seen=[]):
        _seen.append(prompt)
        return replies[(len(_seen) - 1) % len(replies)]
    subject = BrainScoreModel(identifier='scripted', in_channels={'text'}, reply=reply)
    task = TaskContext(instruction='Respond to the word.', response=response)
    return Experiment(subject=subject,
                      protocol=TaskProtocol(STIMULI, task, conditions=list(conditions), repetitions=[0]),
                      tools=tools or [RecordBehavior(), RecordInputsOutputs()],
                      output_dir=directory).run()


def test_valid_invalid_and_missing_trials_show_their_states_and_reasons(tmp_path):
    result = _run(Choice(['animal', 'vehicle']), ['animal', 'nonsense', None], tmp_path / 'run')
    trials = result.trials()
    assert list(trials['choice_state']) == ['valid', 'invalid', 'missing']
    assert list(trials['choice_reason']) == ['', 'no label in the reply', 'no answer']
    assert len(trials) == 3                                   # no row from the finalized answers
    # A saved run read back gives the same table.
    reopened = ExperimentResult.open(tmp_path / 'run').trials()
    assert reopened.equals(trials)


@pytest.mark.parametrize('response, replies, column, states', [
    (Scale(1, 7, 1), ['6', '9', None], 'rating', ['valid', 'invalid', 'missing']),
    (Text(), ['A cat.', 'A boat.', None], 'text', ['valid', 'valid', 'missing'])])
def test_every_behavior_format_has_the_two_columns(tmp_path, response, replies, column, states):
    trials = _run(response, replies, tmp_path / 'run').trials()
    assert list(trials[f'{column}_state']) == states


def test_each_condition_reads_its_own_states(tmp_path):
    trials = _run(Choice(['animal', 'vehicle']), ['animal', 'vehicle', 'nonsense'], tmp_path / 'run',
                  conditions=('baseline', 'again')).trials()
    assert list(trials['condition']) == ['baseline'] * 3 + ['again'] * 3
    assert list(trials['choice_state']) == ['valid', 'valid', 'invalid'] * 2


def test_the_input_output_record_holds_no_finalized_answer(tmp_path):
    """Replay rebuilds the exchange from the input/output record; the finalized answers
    are kept only in the behavior record."""
    result = _run(Choice(['animal', 'vehicle']), ['animal', 'nonsense', None], tmp_path / 'run')
    kinds = {row['payload']['kind'] for row in RunRecord(tmp_path / 'run' / 'inputs_outputs').events()}
    assert 'answer' not in kinds
    behavior = [row['payload']['kind'] for row in RunRecord(tmp_path / 'run' / 'behavior').events()]
    assert behavior.count('answer') == 1


def test_a_session_that_ends_with_an_error_saves_no_finalized_answer(tmp_path):
    def reply(prompt):
        raise RuntimeError('the model failed')
    subject = BrainScoreModel(identifier='failing', in_channels={'text'}, reply=reply)
    task = TaskContext(instruction='Respond to the word.', response=Choice(['animal', 'vehicle']))
    with pytest.raises(RuntimeError, match='the model failed'):
        Experiment(subject=subject, protocol=TaskProtocol(STIMULI, task, repetitions=[0]),
                   tools=[RecordBehavior()], output_dir=tmp_path / 'run').run()
    kinds = [row['payload']['kind'] for row in RunRecord(tmp_path / 'run' / 'behavior').events()]
    assert 'answer' not in kinds


def test_a_session_whose_history_limit_dropped_responses_publishes_none():
    """collect() refuses such a session; the runner then saves no finalized answer."""
    from types import SimpleNamespace
    from brainscore.experiments.runner import _publish_finalized_answers
    published = []
    session = SimpleNamespace(dropped=2, outputs=('behavior:choice',), emitted=[],
                              collect=lambda *a: pytest.fail('collect must not be called'))
    _publish_finalized_answers(session, SimpleNamespace(publish=lambda *a, **k: published.append(a)))
    assert published == []
