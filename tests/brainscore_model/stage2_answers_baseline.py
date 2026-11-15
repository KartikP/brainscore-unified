"""Capture the full answers of the registered part sets, from the code at hand.

Run from the unified root to regenerate ``tests/fixtures/stage2_answers_baseline.json``::

    python -m tests.brainscore_model.stage2_answers_baseline

For each registered part set and each session (the five of ``SESSIONS``, a Choice
with each reaction-time link, and one session that requests only the
probabilities) it records, per requested channel, the values as the exact bytes
of each float64 (NaN and signed zero kept), the dtype, the trial states, the raw
answers, the response record and the model's ``out_channels``. A refused session
records the error type and message. The stand-in parts are fixed functions: no
weights, no downloads.
"""
import json
import struct
import warnings
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'stage2_answers_baseline.json'


def _hex(value):
    return struct.pack('<d', float(value)).hex()


def _cell(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return 'f64:' + _hex(value)
    return str(value)


def _channel(answer):
    values = np.asarray(answer.values)
    if values.dtype.kind in 'fiub':
        flat = ['f64:' + _hex(v) for v in values.astype('<f8').ravel()]
    else:
        flat = [str(v) for v in values.ravel()]
    index = answer.indexes['presentation']
    return {'dtype': str(values.dtype), 'dims': list(answer.dims), 'shape': list(values.shape), 'values': flat,
            'presentation': {name: [_cell(v) for v in index.get_level_values(name)] for name in index.names},
            'response': json.loads(json.dumps(answer.attrs.get('response'), sort_keys=True, default=str))}


def _cases():
    """(name, outputs, task builder) for every session beyond the ones named in SESSIONS."""
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Choice
    from tests.brainscore_model.test_behavior_paths import FITTING

    def choice(**settings):
        return TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], **settings),
                           fitting_stimuli=FITTING)
    cases = {f'choice with reaction time {link}': (
        ['behavior:choice', 'behavior:probabilities', 'behavior:rt'],
        choice(probabilities=True, reaction_time=link)) for link in Choice.REACTION_TIMES}
    cases['probabilities only'] = (['behavior:probabilities'], choice(probabilities=True))
    return cases


def capture():
    from brainscore_core.streaming_helpers import stimulus_session
    from tests.brainscore_model.test_answer_ownership import (
        REGISTERED_PART_SETS, RATINGS, SESSIONS, _model)
    from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI
    from tests.brainscore_model.fixed_order_baseline import task_from
    from tests.fixed_order import fixed_order_owners

    sessions = {}
    for name, (description, outputs) in SESSIONS.items():
        sessions[name] = (outputs, task_from(description, RATINGS if description['kind'] == 'scale' else FITTING))
    sessions.update(_cases())
    found = {}
    for parts in sorted(REGISTERED_PART_SETS):
        model = _model(*parts, owners=fixed_order_owners(*parts))
        entry = {'out_channels': sorted(model.out_channels), 'sessions': {}}
        for name in sorted(sessions):
            outputs, task = sessions[name]
            session = stimulus_session(STIMULI, outputs, task)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model.participate(session)
                    entry['sessions'][name] = {'outputs': {c: _channel(session.collect(c)) for c in outputs}}
            except Exception as error:   # a refusal is part of the baseline
                entry['sessions'][name] = {'error': [type(error).__name__, str(error)]}
        found['+'.join(parts)] = entry
    return found


def render(found):
    return json.dumps(found, sort_keys=True, indent=1) + '\n'


if __name__ == '__main__':
    found = capture()
    FIXTURE.write_text(render(found), encoding='utf-8')
    print(f'{FIXTURE}: {len(found)} part sets')
