"""A format the session does not request needs no owner (stage 3, S3-01 and S3-02).

The task's response names the formats it can be answered in; the session's requested
outputs decide what is measured. A reaction-time link names the rt format; it is no
setting of the choice. The API part arrangement (answer and reply, no owner of the
probabilities, as ``brainscore/models/api_closed/model.py`` wires it) answers a linked
Choice when the session requests only the choice, and refuses a request for the rt.
"""
import pytest

from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore.experiments import Experiment, RecordBehavior, TaskProtocol
from tests.brainscore_model.test_answer_ownership import API, BLIP2, REGISTERED_STATEMENTS, _model, _run
from tests.brainscore_model.test_behavior_paths import STIMULI

pytestmark = pytest.mark.unit

LINKS = Choice.REACTION_TIMES


def _api(calls):
    return _model(*API, owners=REGISTERED_STATEMENTS[API], calls=calls)


def _linked(link):
    return TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], reaction_time=link))


@pytest.mark.parametrize('link', LINKS)
def test_a_linked_choice_session_is_answered_without_a_probability_owner(link):
    calls = []
    choice = _run(_api(calls), ['behavior:choice'], _linked(link)).collect('behavior:choice')
    assert calls == ['reply'] * 3
    assert choice.values.tolist() == [[1.0, 0.0]] * 3


@pytest.mark.parametrize('link', LINKS)
def test_bare_behavior_under_a_linked_choice_is_the_reply_choice(link):
    calls = []
    behavior = _run(_api(calls), ['behavior'], _linked(link)).collect('behavior')
    assert calls == ['reply'] * 3
    assert behavior.values.tolist() == [[1.0, 0.0]] * 3
    assert behavior.attrs['response']['method'] == 'generation'


@pytest.mark.parametrize('outputs', [['behavior:rt'], ['behavior:choice', 'behavior:rt']])
@pytest.mark.parametrize('link', LINKS)
def test_a_requested_reaction_time_without_an_owner_is_refused_before_any_call(link, outputs):
    calls = []
    with pytest.raises(ExecutionNotFeasibleError, match=r"does not produce \['behavior:rt'\]"):
        _run(_api(calls), outputs, _linked(link))
    assert calls == []


@pytest.mark.parametrize('link', LINKS)
def test_the_recorder_records_the_choice_and_refuses_a_reaction_time_it_cannot_get(link, tmp_path):
    calls = []
    subject = _api(calls)
    result = Experiment(subject=subject, protocol=TaskProtocol(STIMULI, _linked(link)),
                        tools=[RecordBehavior()], output_dir=tmp_path / 'choice').run()
    assert result.behavior(measure='choice').values.tolist() == [[1.0, 0.0]] * 3
    calls.clear()
    experiment = Experiment(subject=subject, protocol=TaskProtocol(STIMULI, _linked(link)),
                            tools=[RecordBehavior(['choice', 'rt'])], output_dir=tmp_path / 'rt')
    with pytest.raises(ValueError, match=r"requires outputs \['behavior:rt'\]"):
        experiment.validate()
    assert calls == []


@pytest.mark.parametrize('parts', [API, BLIP2], ids=['answer+reply', 'answer+readout without fitting'])
def test_a_repeated_setup_of_an_unchanged_choice_changes_nothing(parts):
    calls = []
    model = _model(*parts, owners=REGISTERED_STATEMENTS[parts], calls=calls)
    fits = []
    model._behavioral.fit_behavioral_readout = lambda fitting: fits.append(fitting)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    model.start_task(task)
    state = (model._behavioral.response_path, model._behavioral.probability_source,
             model._behavioral.readout_classifier)
    model.start_task(task)
    assert (model._behavioral.response_path, model._behavioral.probability_source,
            model._behavioral.readout_classifier) == state
    assert fits == [] and calls == []
