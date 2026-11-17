"""The readout is fitted by the first session that requests a format it gives (rule 4.5)."""
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from tests.brainscore_model.test_answer_ownership import BLIP2, READOUT, REGISTERED_STATEMENTS, _model, _run
from tests.brainscore_model.test_behavior_paths import FITTING

pytestmark = pytest.mark.unit

PLAIN = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING)


def _counting_fits(model):
    fits, fit = [], model._behavioral.fit_behavioral_readout
    model._behavioral.fit_behavioral_readout = lambda fitting: fits.append(len(fitting)) or fit(fitting)
    return fits


def test_the_readout_is_fitted_by_the_first_session_that_requests_its_format():
    """The requested formats decide the fit, as they decide the passes: one fit per
    configured task, at the start of the first session that needs the readout."""
    calls = []
    model = _model(*BLIP2, owners=REGISTERED_STATEMENTS[BLIP2], calls=calls)
    fits = _counting_fits(model)
    choice = _run(model, ['behavior:choice'], PLAIN).collect('behavior:choice')
    assert fits == [] and calls == ['answer'] * 3                  # no fit, no extraction
    assert choice.values.tolist() == [[1.0, 0.0]] * 3
    probabilities = _run(model, ['behavior:probabilities'], PLAIN).collect('behavior:probabilities')
    assert fits == [len(FITTING)]
    assert probabilities.attrs['response']['method'] == 'readout'
    _run(model, ['behavior:choice', 'behavior:probabilities'], PLAIN)
    assert fits == [len(FITTING)]                                   # still one fit


def test_a_readout_that_derives_the_choice_is_fitted_for_a_choice_session():
    model = _model(*READOUT, owners=REGISTERED_STATEMENTS[READOUT])
    fits = _counting_fits(model)
    choice = _run(model, ['behavior:choice'], PLAIN).collect('behavior:choice')
    assert fits == [len(FITTING)] and choice.attrs['response']['method'] == 'readout'
