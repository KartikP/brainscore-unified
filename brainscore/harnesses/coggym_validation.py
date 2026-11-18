"""Give each CogGym answer a state without replacing CogGym's parser or scientific scores."""
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
        # CogGym labels prompts and accepts JSON keys as "Question N".
        labels = [f'Question {number}' for number in numbers]
        if set(value) == set(labels):
            return [value[key] for key in labels]
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


def _native_value(parsed, query):
    """Whether CogGym's parser gave this answer a value (CogGym fits only such answers)."""
    if not isinstance(parsed, dict):
        return False
    if query.type == 'single-slider':
        return parsed.get('value') is not None
    if query.type == 'multi-slider':
        values = parsed.get('values')
        return isinstance(values, dict) and bool(values) and all(v is not None for v in values.values())
    if query.type == 'multi-choice':
        index = parsed.get('selected_index')
        return type(index) is int and 0 <= index < len(query.option)
    if query.type == 'multi-select':
        return len(parsed.get('binary') or []) == len(query.option)
    if query.type == 'ranking':
        return len(parsed.get('ranking') or []) >= len(query.option)
    return bool(str(parsed.get('text') or '').strip())      # textbox


def validate_responses(trials: Sequence[Any], results: Sequence[dict]) -> dict:
    """Give every requested answer a state; leave native responses and scores untouched.

    Each question of a trial that is not an instruction is one requested answer, counted
    once. A slider answer goes through the strict check ``coggym-slider-v1``: explicit
    question assignments, finite ratings within the scale, agreement with the native
    parser. It repairs, clamps and drops nothing. Any other answer takes its state from
    CogGym's own parse.

    Each issue has a ``response_state`` (Contract Rule 6.2) and a ``reason``: ``missing``
    when CogGym has no value for the answer (no reply: CogGym's entry of a failed call,
    reason ``no_answer``; or its parser read none), so CogGym leaves it out of its fit;
    ``invalid`` when CogGym read a slider value that the check rejects (CogGym keeps it in
    its fit). An answer with no issue is ``valid``. The issues are the detector's
    findings; whether the result is valid is the benchmark's decision.

    A trial with two questions of one tag is refused (``ValueError``): one parsed value
    would stand for both answers.
    """
    by_id = {result.get('trial_id'): result for result in results}
    checked, issues = 0, []
    for trial in trials:
        queries = [query for query in trial.queries if query.type != 'text-instruction']
        tags = [query.tag for query in queries]
        if len(set(tags)) != len(tags):
            raise ValueError(f'CogGym trial {trial.id} has two questions with one tag {tags}: '
                             'one parsed value would stand for both answers')
        result = by_id.get(trial.id)
        replied = isinstance(result, dict) and 'error' not in result
        parsed_answers = result.get('parsed') if replied else None
        parsed_answers = parsed_answers if isinstance(parsed_answers, dict) else {}
        segments = None
        if replied and any(query.type in ('single-slider', 'multi-slider') for query in queries):
            try:
                segments = _segments(result.get('response'), queries)
            except ValueError:
                pass
        for index, query in enumerate(queries):
            checked += 1
            parsed = parsed_answers.get(query.tag)
            if not replied:
                reason = 'no_answer'
            elif query.type not in ('single-slider', 'multi-slider'):
                reason = None if _native_value(parsed, query) else 'not_read_by_coggym'
            elif segments is None:
                reason = 'ambiguous_or_invalid_format'
            else:
                reason = None
                try:
                    if not isinstance(parsed, dict) or not _agrees(segments[index], query, parsed):
                        reason = 'reference_parse_mismatch'
                except ValueError:
                    reason = 'invalid_rating'
            if reason:
                state = 'invalid' if replied and _native_value(parsed, query) else 'missing'
                issues.append({'trial_id': trial.id, 'query': query.tag, 'reason': reason,
                               'response_state': state})
    return {'policy': SLIDER_VALIDATION_POLICY, 'checked_queries': checked, 'issues': issues}
