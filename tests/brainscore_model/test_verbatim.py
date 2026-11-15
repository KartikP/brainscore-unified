"""``Choice(verbatim=True)``: the trial text exactly as given (ADR-0063, point o).

A reply part keeps its own chat template, and its answer's record names that use;
``answer=`` cannot show a text verbatim and is refused before any compute."""
import pytest

from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from tests.brainscore_model.test_answer_ownership import API, BLIP2, REGISTERED_STATEMENTS, _model, _run
from tests.brainscore_model.test_behavior_paths import FITTING

pytestmark = pytest.mark.unit


def test_a_verbatim_choice_from_a_reply_names_the_chat_template():
    calls = []
    model = _model(*API, owners=REGISTERED_STATEMENTS[API], calls=calls)
    choice = _run(model, ['behavior:choice'], TaskContext(response=Choice(['cat', 'dog'], verbatim=True)))
    choice = choice.collect('behavior:choice')
    assert calls == ['reply'] * 3
    assert choice.attrs['response']['parameters'] == {'template': "the reply part's chat template"}
    plain = _run(_model(*API, owners=REGISTERED_STATEMENTS[API]), ['behavior:choice'],
                 TaskContext(instruction='Which?', response=Choice(['cat', 'dog'])))
    assert plain.collect('behavior:choice').attrs['response']['parameters'] == {}


def test_an_answer_and_readout_set_refuses_a_verbatim_choice_before_any_call():
    calls = []
    model = _model(*BLIP2, owners=REGISTERED_STATEMENTS[BLIP2], calls=calls)
    task = TaskContext(response=Choice(['cat', 'dog'], verbatim=True), fitting_stimuli=FITTING)
    with pytest.raises(ExecutionNotFeasibleError, match='verbatim'):
        _run(model, ['behavior:choice'], task)
    assert calls == []
