"""ROAR lexical decision declares what it sends and asks for, and reads a
participate-only subject through a session. Synthetic words; no ROAR data."""
import pytest

from brainscore.benchmarks.roar_yeatman2021.benchmark import Yeatman2021LexicalDecision
from tests.benchmark_declarations import assert_declares, assert_preflight
from tests.test_induced_dyslexia import _ReaderSubject, synthetic_roar  # noqa: F401  (fixture)


@pytest.mark.parametrize('word', ['vision:image', 'text'])
def test_declares_the_word_and_the_question_in_and_the_choice_out(synthetic_roar, word):
    benchmark = Yeatman2021LexicalDecision(channels=(word,), identifier='Yeatman2021-test')
    assert_declares(benchmark, {word, 'instruction'}, {'behavior:choice'})
    assert_preflight(benchmark)


def test_a_participate_only_subject_gets_a_score(synthetic_roar):
    subject = _ReaderSubject()
    score = Yeatman2021LexicalDecision(channels=('text',), identifier='Yeatman2021-lexical_decision-text')(subject)
    assert float(score.attrs['raw']) == 1.0
    assert score.attrs['n_trials'] == 100


def _noisy_readout_model():
    """A readout subject whose features carry the label only weakly, so its accuracy is
    neither 0 nor 1 (deterministic: seeded by each word's number)."""
    import numpy as np
    from brainscore import BrainScoreModel
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    def extract(stimuli, layers, **kwargs):
        ids = list(stimuli['stimulus_id'])
        rows = []
        for word in stimuli['sentence']:
            kind, number = word.split('-')
            rng = np.random.default_rng(int(number))
            rows.append(rng.normal(size=4) + (0.6 if kind == 'real' else -0.6))
        return NeuroidAssembly(np.array(rows), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'sentence': ('presentation', list(stimuli['sentence'])),
            'neuroid_id': ('neuroid', [f'n{i}' for i in range(4)]), 'layer': ('neuroid', [layers[0]] * 4)})
    return BrainScoreModel('noisy-readout', extractor={'text': extract}, in_channels={'text'},
                           behavioral_readout_layer='late',
                           capability_config={'behavior-sessions': {'behavior:choice': ('readout', 'argmax')}})


def test_the_choice_request_scores_what_the_legacy_request_scored(synthetic_roar):
    """ROAR asks for behavior:choice. A readout's derived choice is the argmax of its own
    probability rows, as the benchmark's argmax over the legacy rows was."""
    import numpy as np
    from brainscore_core.contract import TaskContext
    from brainscore_core.streaming_helpers import behavioral_response
    benchmark = Yeatman2021LexicalDecision(channels=('text',), identifier='Yeatman2021-lexical_decision-text')
    new = float(benchmark(_noisy_readout_model()).attrs['raw'])
    legacy = TaskContext(task_type='probabilities', fitting_stimuli=benchmark._train_stimuli,
                         label_set=['real', 'pseudo'], instruction='Is this a real word or a pseudo word?')
    rows = behavioral_response(_noisy_readout_model(), benchmark.stimulus_set, legacy)
    labels = list(rows['choice'].values)
    truth = dict(zip(benchmark.stimulus_set['stimulus_id'], benchmark.stimulus_set['image_label']))
    old = float(np.mean([labels[i] == truth[sid] for sid, i in
                         zip(rows['stimulus_id'].values, np.asarray(rows.values).argmax(axis=1))]))
    assert 0.5 < new < 1.0
    assert new == old


def test_a_readout_without_its_choice_statement_is_not_applicable(synthetic_roar):
    from brainscore import BrainScoreModel
    from brainscore_core.compatibility import CompatibilityError, check_channel_compatibility
    model = BrainScoreModel('readout-no-statement', extractor={'text': lambda *a, **k: None},
                            in_channels={'text'}, behavioral_readout_layer='late')
    import pytest as _pytest
    with _pytest.raises(CompatibilityError, match='behavior:choice'):
        check_channel_compatibility(model, Yeatman2021LexicalDecision(channels=('text',), identifier='Yeatman2021-lexical_decision-text'))
