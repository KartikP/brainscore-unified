"""Null subjects declare what they produce (Contract Rules v3, rule 2) and stay applicable
on the benchmarks they are floors for."""
from brainscore_core.compatibility import check_compatibility

from brainscore.models.chance_baseline.model import ChanceBaseline
from tests.test_induced_dyslexia import synthetic_roar  # noqa: F401  (fixture)


class _BehavioralBenchmark:
    identifier = 'behavioral'
    required_input_channels = {'vision:image'}
    requested_output_channels = {'behavior'}


class _ChoiceBenchmark(_BehavioralBenchmark):
    requested_output_channels = {'behavior:choice'}


def test_chance_baseline_declares_bare_behavior_and_the_choice():
    model = ChanceBaseline()
    assert model.out_channels == {'behavior', 'behavior:choice'}
    assert check_compatibility(model, _BehavioralBenchmark()).kind == 'full'
    assert check_compatibility(model, _ChoiceBenchmark()).kind == 'full'


def test_chance_baseline_scores_the_chance_floor_on_roar(synthetic_roar):
    """Uniform rows, the first label by core's argmax: 0.5 on the 50/50 test set, as before."""
    from brainscore.benchmarks.roar_yeatman2021.benchmark import Yeatman2021LexicalDecision
    model = ChanceBaseline()
    sessions = []
    participate = model.participate
    model.participate = lambda session: sessions.append(session) or participate(session)
    score = Yeatman2021LexicalDecision(channels=('text',))(model)
    assert float(score.attrs['raw']) == 0.5
    assert (score.attrs['accuracy_real'], score.attrs['accuracy_pseudo']) == (1.0, 0.0)
    record = sessions[0].collect('behavior:choice').attrs['response']
    assert (record['origin'], record['method'], record['conversion'], record['source_format']) == (
        'derived', 'uniform', 'argmax', 'behavior:probabilities')


def test_random_vit_keeps_its_readout_choice_and_is_applicable_on_roar_image(monkeypatch, synthetic_roar):
    from brainscore.benchmarks.roar_yeatman2021.benchmark import Yeatman2021LexicalDecision
    from tests.test_readout_registrations import BUILDERS
    subject = BUILDERS['random-vit-b-32'](monkeypatch).get_model('random-vit-b-32')
    assert 'behavior:choice' in subject.out_channels
    assert check_compatibility(subject, Yeatman2021LexicalDecision(channels=('vision:image',)))
