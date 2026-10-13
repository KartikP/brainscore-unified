"""Check slider answers without replacing CogGym's parser or scientific scores."""
import json
import math
import re
from typing import Any, Sequence

SLIDER_VALIDATION_POLICY = 'coggym-slider-v1'
_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate answer key')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('Nonfinite rating')


_DECODER = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def _documents(text):
    """Read whole JSON values; never search prose for a number or object."""
    values = []
    while text.strip():
        text = text.lstrip()
        value, end = _DECODER.raw_decode(text)
        values.append(value)
        text = text[end:]
    return values


def _segments(raw, queries):
    if not isinstance(raw, str):
        raise ValueError('Missing response')
    tags = [query.tag for query in queries]
    if len(set(tags)) != len(tags):
        raise ValueError('Ambiguous question tags')
    text = raw.strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\s*```', text, flags=re.DOTALL)
    if fence:
        text = fence[1]
    try:
        documents = _documents(text)
    except ValueError:
        # A bare rating or explicitly numbered answers are also unambiguous.
        if len(queries) == 1:
            match = re.fullmatch(rf'(?:Answer:\s*)?({_NUMBER})', text, flags=re.IGNORECASE)
            if match:
                return [match[1]]
        parts = re.split(r'(?im)^\s*(?:Question\s+)?([1-9]\d*)[.):]\s*', text)
        if parts[0].strip() or len(parts) != 1 + 2 * len(queries):
            raise ValueError('Ambiguous answer format') from None
        if parts[1::2] != [str(i) for i in range(1, len(queries) + 1)]:
            raise ValueError('Ambiguous question numbering')
        return [_segments(part, [query])[0] for part, query in zip(parts[2::2], queries)]
    if len(documents) == len(queries) and len(documents) > 1 and all(
        isinstance(value, dict) for value in documents
    ):
        return documents
    if len(documents) != 1:
        raise ValueError('Missing answers')
    value = documents[0]
    if isinstance(value, dict) and set(value) == set(tags):
        return [value[tag] for tag in tags]
    if len(queries) == 1:
        return [value]
    if isinstance(value, dict):
        numbers = [str(i) for i in range(1, len(queries) + 1)]
        if set(value) == set(numbers):
            return [value[key] for key in numbers]
        if set(value) == {'answers'}:
            value = value['answers']
    if isinstance(value, list) and len(value) == len(queries):
        return value
    raise ValueError('Missing or ambiguous question assignments')


def _rating(value, query):
    if isinstance(value, str) and re.fullmatch(_NUMBER, value.strip()):
        value = float(value)
    if type(value) not in (int, float):
        raise ValueError('Expected one finite number')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError('Expected one finite number')
    config = query.slider_config or {}
    if not config.get('min', 0) <= value <= config.get('max', 100):
        raise ValueError('Rating outside the declared scale')
    return value


def _agrees(value, query, parsed):
    if isinstance(value, dict) and set(value) == {'answer'}:
        value = value['answer']
    if query.type == 'single-slider':
        expected = _rating(value, query)
        actual = parsed.get('value')
        return type(actual) in (int, float) and actual == expected
    if not isinstance(value, dict) or set(value) != set(query.option):
        raise ValueError('Expected a rating for every named option')
    expected = {key: _rating(value[key], query) for key in query.option}
    actual = parsed.get('values')
    return isinstance(actual, dict) and set(actual) == set(expected) and all(
        type(actual[key]) in (int, float) and actual[key] == rating
        for key, rating in expected.items()
    )


def validate_slider_responses(trials: Sequence[Any], results: Sequence[dict]) -> dict:
    """Report invalid slider answers; leave native responses and scores untouched.

    Require explicit question assignments, finite ratings within the scale, and
    agreement with the native parser. Do not repair, clamp, or drop answers.
    Other question types keep their native evaluation behavior.
    """
    by_id = {result.get('trial_id'): result for result in results}
    checked, issues = 0, []
    for trial in trials:
        queries = [query for query in trial.queries if query.type != 'text-instruction']
        sliders = [(index, query) for index, query in enumerate(queries)
                   if query.type in ('single-slider', 'multi-slider')]
        if not sliders:
            continue
        result = by_id.get(trial.id, {})
        try:
            segments = _segments(result.get('response'), queries)
        except ValueError:
            segments = None
        for index, query in sliders:
            checked += 1
            reason = None
            if segments is None:
                reason = 'ambiguous_or_invalid_format'
            else:
                parsed_answers = result.get('parsed')
                parsed = parsed_answers.get(query.tag, {}) if isinstance(parsed_answers, dict) else {}
                try:
                    if not isinstance(parsed, dict) or not _agrees(segments[index], query, parsed):
                        reason = 'reference_parse_mismatch'
                except ValueError:
                    reason = 'invalid_rating'
            if reason:
                issues.append({'trial_id': trial.id, 'query': query.tag, 'reason': reason})
    return {
        'policy': SLIDER_VALIDATION_POLICY,
        'status': 'invalid' if issues else 'valid' if checked else 'not_applicable',
        'checked_queries': checked,
        'invalid_queries': len(issues),
        'issues': issues,
    }
