"""ROAR lexical decision declares what it sends and asks for, and reads a
participate-only subject through a session. Synthetic words; no ROAR data."""
import pytest

from brainscore.benchmarks.roar_yeatman2021.benchmark import Yeatman2021LexicalDecision
from tests.benchmark_declarations import assert_declares, assert_preflight
from tests.test_induced_dyslexia import _ReaderSubject, synthetic_roar  # noqa: F401  (fixture)


@pytest.mark.parametrize('modality', ['vision', 'text'])
def test_declares_the_word_and_the_question_in_and_behavior_out(synthetic_roar, modality):
    benchmark = Yeatman2021LexicalDecision(modality=modality)
    assert_declares(benchmark, {modality, 'instruction'}, {'behavior'})
    assert_preflight(benchmark)


def test_a_participate_only_subject_gets_a_score(synthetic_roar):
    subject = _ReaderSubject()
    score = Yeatman2021LexicalDecision(modality='text')(subject)
    assert float(score.attrs['raw']) == 1.0
    assert score.attrs['n_test_stimuli'] == 100
