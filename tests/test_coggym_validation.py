"""Reject invented slider ratings without changing the reference parser."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from brainscore.harnesses.coggym_validation import validate_slider_responses


def query(tag='rating', kind='single-slider', options=()):
    return SimpleNamespace(tag=tag, type=kind, option=list(options), slider_config={'min': 0, 'max': 100})


def validate(raw, parsed, queries=None):
    trial = SimpleNamespace(id='trial-1', queries=queries or [query()])
    results = [{'trial_id': trial.id, 'response': raw, 'parsed': parsed}]
    before = deepcopy(results)
    report = validate_slider_responses([trial], results)
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
    assert report['status'] == 'invalid'
    assert report['checked_queries'] == report['invalid_queries'] == 1
    assert report['issues'][0]['trial_id'] == 'trial-1'
    assert report['issues'][0]['query'] == 'rating'


@pytest.mark.parametrize('raw', ['42', 'Answer: 42', '{"answer": 42}', '{"answer": "42"}',
                                  '```json\n{"answer": 42}\n```'])
def test_unambiguous_single_rating_matches_native_value(raw):
    report = validate(raw, {'rating': {'value': 42.0}})
    assert report['status'] == 'valid'
    assert report['invalid_queries'] == 0


@pytest.mark.parametrize('raw', [
    '{"left": 42, "right": 61}',
    '{"left": {"answer": 42}, "right": {"answer": 61}}',
    '{"answer": 42}\n{"answer": 61}',
    '[42, 61]', '{"answers": [42, 61]}',
    '{"2": 61, "1": 42}',
    'Question 1: 42\nQuestion 2: 61',
])
def test_multiple_answers_are_assigned_explicitly(raw):
    report = validate(raw, {'left': {'value': 42}, 'right': {'value': 61}}, [query('left'), query('right')])
    assert report['status'] == 'valid'
    assert report['checked_queries'] == 2


@pytest.mark.parametrize('raw', [
    '{"answer": 42}', '{"left": [9, "0"], "right": [100, "100"]}',
    '{"first_random_key": 42, "second_random_key": 61}',
    'Question 1: 42\nQuestion 1: 61',
    '42 61',
])
def test_ambiguous_multi_question_answers_are_rejected(raw):
    report = validate(raw, {'left': {'value': 42}, 'right': {'value': 61}}, [query('left'), query('right')])
    assert report['status'] == 'invalid'


def test_disagreement_with_native_parser_is_not_silently_corrected():
    report = validate('{"answer": 42}', {'rating': {'value': 0}})
    assert report['issues'][0]['reason'] == 'reference_parse_mismatch'


def test_multi_slider_requires_every_named_option_and_native_agreement():
    queries = [query('ratings', 'multi-slider', ['A', 'B'])]
    parsed = {'ratings': {'values': {'A': 10, 'B': 90}}}
    assert validate('{"A": 10, "B": 90}', parsed, queries)['status'] == 'valid'
    assert validate('{"answer": {"A": 10, "B": 90}}', parsed, queries)['status'] == 'valid'
    assert validate('{"A": 10}', parsed, queries)['status'] == 'invalid'
    assert validate('{"A": [10], "B": 90}', parsed, queries)['status'] == 'invalid'


def test_mixed_questions_only_validate_slider_answers():
    queries = [query('choice', 'multi-choice'), query('rating')]
    report = validate('{"choice": "A", "rating": 42}', {'rating': {'value': 42}}, queries)
    assert report['checked_queries'] == 1
    assert report['status'] == 'valid'


def test_trials_without_sliders_are_not_reinterpreted():
    report = validate('any native response', {}, [query('choice', 'multi-choice')])
    assert report['status'] == 'not_applicable'
    assert report['checked_queries'] == 0


def test_missing_response_is_not_validated_as_success():
    trial = SimpleNamespace(id='trial-1', queries=[query()])
    report = validate_slider_responses([trial], [])
    assert report['status'] == 'invalid'


def test_duplicate_question_tags_cannot_broadcast_one_rating():
    report = validate('{"rating": 42}', {'rating': {'value': 42}}, [query(), query()])
    assert report['invalid_queries'] == 2
