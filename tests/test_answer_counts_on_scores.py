"""Every behavior score carries the counts of the trial states, its answer policy and
the validity of the result (Contract Rule 6.2)."""
import numpy as np
import pytest

from brainscore.benchmarks import benchmark_score, not_estimated
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

pytestmark = pytest.mark.unit


class _Benchmark:
    version = 1
    ceiling = not_estimated()


def _answer(states):
    n = len(states)
    return BehavioralAssembly(np.zeros((n, 2), dtype=np.float32), dims=['presentation', 'choice'], coords={
        'stimulus_id': ('presentation', [f's{i}' for i in range(n)]), 'choice': ['a', 'b'],
        'response_state': ('presentation', list(states))})


POLICY = 'invalid: wrong, counted; missing: wrong, counted'


def test_the_counts_of_the_states_and_the_policy():
    score = benchmark_score(_Benchmark(), raw=0.5, metric='accuracy', answer_policy=POLICY,
                            answers=_answer(['valid', 'invalid', 'missing', 'valid']))
    attrs = score.attrs
    assert (attrs['n_trials'], attrs['n_valid'], attrs['n_invalid'], attrs['n_missing']) == (4, 2, 1, 1)
    assert attrs['answer_policy'] == POLICY
    assert attrs['result_validity'] == 'no limit stated' and 'result_validity_reason' not in attrs


def test_without_a_limit_no_check_is_claimed_even_when_every_answer_is_missing():
    score = benchmark_score(_Benchmark(), raw=0.0, metric='accuracy', answer_policy=POLICY,
                            answers=_answer(['missing', 'missing', 'missing']))
    assert score.attrs['n_missing'] == 3
    assert score.attrs['result_validity'] == 'no limit stated'
    assert 'result_validity_reason' not in score.attrs


def test_at_or_below_the_limit_the_result_is_valid():
    score = benchmark_score(_Benchmark(), raw=0.5, metric='accuracy', answer_policy=POLICY,
                            answers=_answer(['valid', 'invalid', 'missing', 'valid']), invalid_limit=0.5)
    assert score.attrs['result_validity'] == 'valid' and 'result_validity_reason' not in score.attrs


def test_above_the_limit_the_result_is_returned_marked_invalid():
    score = benchmark_score(_Benchmark(), raw=0.5, metric='accuracy', answer_policy=POLICY,
                            answers=_answer(['valid', 'invalid', 'missing', 'valid']), invalid_limit=0.25)
    assert float(score) == 0.5
    assert score.attrs['result_validity'] == 'invalid'
    assert score.attrs['result_validity_reason'] == (
        '0.5 of the trials are invalid or missing, above the limit 0.25')


def test_answers_by_condition_give_sums_and_counts_by_condition():
    score = benchmark_score(_Benchmark(), raw=0.5, metric='accuracy', answer_policy=POLICY,
                            answers={'intact': _answer(['valid', 'invalid']),
                                     'lesion': _answer(['missing', 'valid', 'valid'])})
    assert (score.attrs['n_trials'], score.attrs['n_invalid'], score.attrs['n_missing']) == (5, 1, 1)
    assert score.attrs['counts_by_condition'] == {
        'intact': {'n_trials': 2, 'n_valid': 1, 'n_invalid': 1, 'n_missing': 0},
        'lesion': {'n_trials': 3, 'n_valid': 2, 'n_invalid': 0, 'n_missing': 1}}


def test_a_policy_is_required_and_bare_behavior_has_no_states():
    with pytest.raises(TypeError, match='answer_policy'):
        benchmark_score(_Benchmark(), raw=0.5, metric='accuracy', answers=_answer(['valid']))
    with pytest.raises(TypeError, match='pass answers='):
        benchmark_score(_Benchmark(), raw=0.5, metric='accuracy', answer_policy=POLICY)
    bare = BehavioralAssembly(np.zeros((1, 2)), dims=['presentation', 'choice'],
                              coords={'stimulus_id': ('presentation', ['s0']), 'choice': ['a', 'b']})
    with pytest.raises(ValueError, match='response_state'):
        benchmark_score(_Benchmark(), raw=0.5, metric='accuracy', answer_policy=POLICY, answers=bare)
    with pytest.raises(TypeError, match='writes'):
        benchmark_score(_Benchmark(), raw=0.5, metric='accuracy', n_trials=3)
