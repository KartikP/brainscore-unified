"""Give each answer a state; reject invented slider ratings without changing the reference parser."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from brainscore.harnesses.coggym.validation import validate_responses


def query(tag='rating', kind='single-slider', options=()):
    return SimpleNamespace(tag=tag, type=kind, option=list(options), slider_config={'min': 0, 'max': 100})


def validate(raw, parsed, queries=None):
    trial = SimpleNamespace(id='trial-1', queries=queries or [query()])
    results = [{'trial_id': trial.id, 'response': raw, 'parsed': parsed}]
    before = deepcopy(results)
    report = validate_responses([trial], results)
    assert results == before
    return report


@pytest.mark.parametrize('raw', [
    '{"answer": <number between 0 and 100>}',
    'I cannot answer on a scale of 0 to 100.',
    '{"answer": [9, "0"]}',
    '{"answer": {"question_id": "100"}}',
    '{"answer": true}', '{"answer": null}', '{"answer": NaN}',
    '{"answer": 101}', '{"answer": -1}', '{"answer": 1e999}',
    '{"answer": 10, "answer": 20}',
    '{"answer": ' + '9' * 400 + '}',
    'Maybe 20, but perhaps 40.',
    'Here is my answer: {"answer": 20}',
])
def test_permissive_native_parse_does_not_make_invalid_text_valid(raw):
    report = validate(raw, {'rating': {'value': 0}})
    assert report['checked_queries'] == len(report['issues']) == 1
    assert report['issues'][0]['trial_id'] == 'trial-1'
    assert report['issues'][0]['query'] == 'rating'


@pytest.mark.parametrize('raw', ['42', 'Answer: 42', '{"answer": 42}', '{"answer": "42"}',
                                  '```json\n{"answer": 42}\n```'])
def test_unambiguous_single_rating_matches_native_value(raw):
    report = validate(raw, {'rating': {'value': 42.0}})
    assert report['issues'] == []


@pytest.mark.parametrize('raw', [
    '{"left": 42, "right": 61}',
    '{"left": {"answer": 42}, "right": {"answer": 61}}',
    '{"answer": 42}\n{"answer": 61}',
    '[42, 61]', '{"answers": [42, 61]}',
    '{"2": 61, "1": 42}',
    '{"Question 1": {"answer": 42}, "Question 2": {"answer": 61}}',
    '{"Question 2": 61, "Question 1": 42}',
    'Question 1: 42\nQuestion 2: 61',
])
def test_multiple_answers_are_assigned_explicitly(raw):
    report = validate(raw, {'left': {'value': 42}, 'right': {'value': 61}}, [query('left'), query('right')])
    assert report['issues'] == []
    assert report['checked_queries'] == 2


@pytest.mark.parametrize('raw', [
    '{"answer": 42}', '{"left": [9, "0"], "right": [100, "100"]}',
    '{"first_random_key": 42, "second_random_key": 61}',
    'Question 1: 42\nQuestion 1: 61',
    '42 61',
    '{"Question 1": 42}',
    '{"Question 1": 42, "Question 3": 61}',
    '{"Question 1": 42, "2": 61}',
    '{"Question 1": 42, "Question 2": 61, "Question 3": 80}',
    '{"Question 1": 42, "Question 1": 61}',
])
def test_ambiguous_multi_question_answers_are_rejected(raw):
    report = validate(raw, {'left': {'value': 42}, 'right': {'value': 61}}, [query('left'), query('right')])
    assert report['issues']


def test_disagreement_with_native_parser_is_not_silently_corrected():
    report = validate('{"answer": 42}', {'rating': {'value': 0}})
    assert report['issues'][0]['reason'] == 'reference_parse_mismatch'


def test_multi_slider_requires_every_named_option_and_native_agreement():
    queries = [query('ratings', 'multi-slider', ['A', 'B'])]
    parsed = {'ratings': {'values': {'A': 10, 'B': 90}}}
    assert validate('{"A": 10, "B": 90}', parsed, queries)['issues'] == []
    assert validate('{"answer": {"A": 10, "B": 90}}', parsed, queries)['issues'] == []
    assert validate('{"A": 10}', parsed, queries)['issues']
    assert validate('{"A": [10], "B": 90}', parsed, queries)['issues']


def test_every_requested_answer_is_counted_and_a_choice_takes_coggyms_parse():
    queries = [query('choice', 'multi-choice', ['A', 'B']), query('rating')]
    parsed = {'choice': {'selected': 'A', 'selected_index': 0}, 'rating': {'value': 42}}
    report = validate('{"choice": "A", "rating": 42}', parsed, queries)
    assert report['checked_queries'] == 2
    assert report['issues'] == []


def test_a_choice_coggym_could_not_read_is_missing():
    queries = [query('choice', 'multi-choice', ['A', 'B'])]
    report = validate('any native response', {'choice': {'selected': 'x', 'selected_index': -1}}, queries)
    assert report['checked_queries'] == 1
    assert report['issues'][0]['response_state'] == 'missing'


def test_missing_response_is_not_validated_as_success():
    trial = SimpleNamespace(id='trial-1', queries=[query()])
    report = validate_responses([trial], [])
    assert report['issues'][0]['response_state'] == 'missing'


@pytest.mark.parametrize('kind', ['single-slider', 'multi-choice'])
def test_coggyms_entry_of_a_call_with_no_reply_is_missing_for_every_question_type(kind):
    trial = SimpleNamespace(id='trial-1', queries=[query('answer', kind, ['A', 'B'])])
    report = validate_responses([trial], [{'trial_id': 'trial-1', 'error': 'no answer', 'model': 'm'}])
    assert [(issue['response_state'], issue['reason']) for issue in report['issues']] == [('missing', 'no_answer')]


def test_duplicate_question_tags_are_a_fault_of_the_run():
    with pytest.raises(ValueError, match='one tag'):
        validate('{"rating": 42}', {'rating': {'value': 42}}, [query(), query()])


def test_question_labels_preserve_mixed_query_order():
    queries = [query('choice', 'multi-choice', ['A', 'B']), query('ratings', 'multi-slider', ['A', 'B'])]
    report = validate(
        '{"Question 2": {"answer": {"A": 10, "B": 90}}, "Question 1": "A"}',
        {'choice': {'selected': 'A', 'selected_index': 0}, 'ratings': {'values': {'A': 10, 'B': 90}}},
        queries,
    )
    assert report['issues'] == []
    assert report['checked_queries'] == 2


def test_question_labels_still_require_native_parser_agreement():
    report = validate(
        '{"Question 1": 42, "Question 2": 61}',
        {'left': {'value': 61}, 'right': {'value': 42}},
        [query('left'), query('right')],
    )
    assert len(report['issues']) == 2
    assert all(issue['reason'] == 'reference_parse_mismatch' for issue in report['issues'])


def test_each_rejected_answer_has_a_state():
    # CogGym read a value the check rejects: invalid; CogGym read none: missing.
    assert validate('Maybe 20.', {'rating': {'value': 20}})['issues'][0]['response_state'] == 'invalid'
    assert validate('no idea', {'rating': {'value': None}})['issues'][0]['response_state'] == 'missing'
    report = validate_responses([SimpleNamespace(id='trial-1', queries=[query()])], [])
    assert report['issues'][0]['response_state'] == 'missing'


def test_repeated_numbering_in_one_answer_is_invalid_and_does_not_stop_the_check():
    # One numbered answer holds a value, not a second numbered list: no depth to run out of.
    report = validate('1. ' * 5000 + '50', {'rating': {'value': 50}})
    assert report['issues'][0]['response_state'] == 'invalid'
    assert validate('1. 50', {'rating': {'value': 50}})['issues'] == []
