"""Null subjects declare what they produce (Contract Rules v3, rule 2) and stay applicable
on the benchmarks they are floors for."""
from brainscore_core.compatibility import check_compatibility

from brainscore.models.chance_baseline.model import ChanceBaseline
from brainscore.benchmarks.induced_dyslexia.test import synthetic_roar  # noqa: F401  (fixture)


class _BehavioralBenchmark:
    identifier = 'behavioral'
    required_input_channels = {'vision:image'}
    requested_output_channels = {'behavior'}


class _ChoiceBenchmark(_BehavioralBenchmark):
    requested_output_channels = {'behavior:choice'}


def test_chance_baseline_declares_bare_behavior_the_choice_and_the_probabilities():
    model = ChanceBaseline()
    assert {'behavior', 'behavior:choice', 'behavior:probabilities'} <= model.out_channels
    assert check_compatibility(model, _BehavioralBenchmark()).kind == 'full'
    assert check_compatibility(model, _ChoiceBenchmark()).kind == 'full'


def test_chance_baseline_scores_the_chance_floor_on_roar(synthetic_roar):
    """Uniform rows, the first label by core's argmax: 0.5 on the 50/50 test set, as before."""
    from brainscore.benchmarks.roar_yeatman2021.benchmark import Yeatman2021LexicalDecision
    model = ChanceBaseline()
    sessions = []
    participate = model.participate
    model.participate = lambda session: sessions.append(session) or participate(session)
    score = Yeatman2021LexicalDecision(channels=('text',), identifier='Yeatman2021-lexical_decision-text')(model)
    assert float(score.attrs['raw']) == 0.5
    assert (score.attrs['accuracy_real'], score.attrs['accuracy_pseudo']) == (1.0, 0.0)
    record = sessions[0].collect('behavior:choice').attrs['response']
    assert (record['origin'], record['method'], record['conversion'], record['source_format']) == (
        'derived', 'uniform', 'argmax', 'behavior:probabilities')


def test_random_vit_keeps_its_readout_choice_and_is_applicable_on_roar_image(monkeypatch, synthetic_roar):
    from brainscore.benchmarks.roar_yeatman2021.benchmark import Yeatman2021LexicalDecision
    from tests.test_models.test_readout_registrations import BUILDERS
    subject = BUILDERS['random-vit-b-32'](monkeypatch).get_model('random-vit-b-32')
    assert 'behavior:choice' in subject.out_channels
    assert check_compatibility(subject, Yeatman2021LexicalDecision(channels=('vision:image',), identifier='Yeatman2021-lexical_decision-image'))


def _session(rows, labels):
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    stimuli = StimulusSet(rows)
    stimuli.identifier = 'chance-trials'
    return stimulus_session(stimuli, ['behavior:choice'], TaskContext(response=Choice(labels)))


def test_chance_baseline_answers_every_trial_occurrence():
    session = _session([{'stimulus_id': 's', 'sentence': 'a'}, {'stimulus_id': 's', 'sentence': 'a'},
                        {'stimulus_id': 't', 'sentence': 'b'}], ['real', 'pseudo'])
    ChanceBaseline().participate(session)
    choice = session.collect('behavior:choice')
    assert list(choice['stimulus_id'].values) == ['s', 's', 't']
    assert list(choice['response_state'].values) == ['valid'] * 3
    assert choice.values.tolist() == [[1.0, 0.0]] * 3


def test_chance_baseline_refuses_options_per_row_at_the_start_of_the_session():
    import pytest
    session = _session([{'stimulus_id': 's', 'sentence': 'a', 'options': ('x', 'y')}], 'options')
    with pytest.raises(ValueError, match=r"options per row \(Choice\('options'\)\) are not supported"):
        ChanceBaseline().participate(session)
    assert session.emitted == []
