"""Each Qwen2.5-VL registration states the part that gives its probabilities (rule 4.11):
the readout, or the word likelihood for the variants with a lesionable word-form region."""
import pytest

from brainscore import model_registry
from brainscore.models.qwen25_vl_3b import model as registration
from tests.brainscore_model.test_answer_ownership import QWEN, _model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('identifier, part', [
    ('qwen2.5-vl-3b', 'readout'), ('qwen2.5-vl-7b', 'readout'), ('qwen2.5-vl-3b-bf16', 'readout'),
    ('qwen2.5-vl-3b-vwfa', 'likelihood'), ('qwen2.5-vl-7b-vwfa', 'likelihood')])
def test_each_registration_states_its_probabilities_part(monkeypatch, identifier, part):
    built = {}
    monkeypatch.setattr(registration, 'get_model', lambda name, **kwargs: built.update(name=name, **kwargs))
    model_registry[identifier]()
    assert built == {'name': identifier, 'probabilities': part}


@pytest.mark.parametrize('part', sorted(registration.PROBABILITY_PARTS))
def test_each_probabilities_statement_of_the_builder_is_accepted(part):
    owners = {'behavior': 'generation', 'behavior:rating': 'generation',
              'behavior:probabilities': registration.PROBABILITY_PARTS[part]}
    assert 'behavior:probabilities' in _model(*QWEN, owners=owners).out_channels
