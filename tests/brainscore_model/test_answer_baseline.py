"""Stage 2 numeric baseline: the registered part sets give the recorded answers, byte for byte.

``tests/fixtures/stage2_answers_baseline.json`` is written by
``stage2_answers_baseline.py`` from the code of the commit before stage 2 changed
any answer path. The comparison is on the rendered text, so every float64 is
compared by its exact bytes (NaN and signed zero included).
"""
import json
from pathlib import Path

import pytest

from tests.brainscore_model.stage2_answers_baseline import FIXTURE, capture, render

pytestmark = pytest.mark.unit


def test_the_registered_part_sets_give_the_recorded_answers():
    expected = Path(FIXTURE).read_text(encoding='utf-8')
    found = render(capture())
    if found != expected:
        recorded, now = json.loads(expected), json.loads(found)
        differences = [(part, session, key) for part in sorted(set(recorded) | set(now))
                       for session in sorted(set(recorded.get(part, {}).get('sessions', {}))
                                             | set(now.get(part, {}).get('sessions', {})))
                       for key in ('outputs', 'error')
                       if recorded.get(part, {}).get('sessions', {}).get(session, {}).get(key)
                       != now.get(part, {}).get('sessions', {}).get(session, {}).get(key)]
        differences += [(part, 'out_channels', '') for part in recorded
                        if recorded[part]['out_channels'] != now.get(part, {}).get('out_channels')]
        pytest.fail(f'{len(differences)} differences from the baseline: {differences[:5]}')


def test_the_baseline_covers_every_part_set_and_session():
    recorded = json.loads(Path(FIXTURE).read_text(encoding='utf-8'))
    assert len(recorded) == 5
    assert all(len(entry['sessions']) == 11 for entry in recorded.values())
