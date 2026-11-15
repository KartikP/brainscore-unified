"""Capture the full answers of the registered part sets, from the code at hand.

Run from the unified root to regenerate ``tests/fixtures/stage2_answers_baseline.json``::

    python -m tests.brainscore_model.stage2_answers_baseline

For each registered part set and each session (the five of ``SESSIONS``, a Choice
with each reaction-time link, and one session that requests only the
probabilities, and the three task forms of ``stage3_cases``) it records, per requested channel, the values as the exact bytes
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

from tests.brainscore_model.test_behavior_paths import FITTING

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'stage2_answers_baseline.json'

RATINGS = FITTING.copy()
RATINGS['label'] = [6.0, 2.0, 6.0, 2.0]
RATINGS.identifier = 'behavior-paths-ratings'
CHOICE_FORM = {'kind': 'choice', 'labels': 'shared', 'reaction_time': None,
               'raw_scores': False, 'verbatim': False, 'confidence': False, 'fitting': True}
SESSIONS = {   # task descriptions and the outputs the session requests
    'legacy answer': ({'kind': 'legacy', 'instruction': True, 'label_set': ['cat', 'dog'], 'fitting': True},
                      ['behavior']),
    'legacy readout': ({'kind': 'legacy', 'instruction': False, 'label_set': ['cat', 'dog'], 'fitting': True},
                       ['behavior']),
    'choice': (dict(CHOICE_FORM), ['behavior:choice', 'behavior:probabilities']),
    'raw scores': (dict(CHOICE_FORM, raw_scores=True, fitting=False),
                   ['behavior:choice', 'behavior:probabilities']),
    'rating': ({'kind': 'scale', 'fitting': True}, ['behavior:rating']),
}


def task_from(description, fitting_stimuli):
    """The TaskContext of a task description of ``SESSIONS``."""
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Choice, Scale
    fitting = fitting_stimuli if description['fitting'] else None
    kind = description['kind']
    if kind == 'legacy':
        return TaskContext(task_type='probabilities', label_set=description['label_set'],
                           instruction='Which?' if description['instruction'] else None,
                           fitting_stimuli=fitting)
    if kind == 'scale':
        return TaskContext(instruction='How much?', response=Scale(1, 7), fitting_stimuli=fitting)
    choice = Choice(['cat', 'dog'] if description['labels'] == 'shared' else 'options',
                    reaction_time=description['reaction_time'], raw_scores=description['raw_scores'],
                    verbatim=description['verbatim'],
                    confidence=Scale(1, 5) if description['confidence'] else None)
    return TaskContext(instruction=None if description['verbatim'] else 'Which?', response=choice,
                       fitting_stimuli=fitting)


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

    def choice(**settings):
        return TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], **settings),
                           fitting_stimuli=FITTING)
    cases = {f'choice with reaction time {link}': (
        ['behavior:choice', 'behavior:probabilities', 'behavior:rt'],
        choice(reaction_time=link)) for link in Choice.REACTION_TIMES}
    cases['probabilities only'] = (['behavior:probabilities'], choice())
    return cases


def stage3_cases():
    """(outputs, task, stimuli) of the task forms stage 3 changes: ROAR (a plain Choice with
    an instruction and fitting stimuli), EWoK log scores (options per row, raw scores and
    raw text, no fitting stimuli) and EWoK choice (raw text, no fitting stimuli)."""
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Choice
    from tests.brainscore_model.test_answer_ownership import OPTIONS
    from tests.brainscore_model.test_behavior_paths import STIMULI
    return {
        'roar plain choice': (['behavior:choice'], TaskContext(
            instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING), STIMULI),
        'ewok log scores': (['behavior:probabilities'], TaskContext(
            response=Choice('options', raw_scores=True, verbatim=True)), OPTIONS),
        'ewok choice': (['behavior:choice'], TaskContext(response=Choice(['1', '2'], verbatim=True)), STIMULI),
    }


def capture():
    from brainscore_core.streaming_helpers import stimulus_session
    from tests.brainscore_model.test_answer_ownership import REGISTERED_PART_SETS, REGISTERED_STATEMENTS, _model
    from tests.brainscore_model.test_behavior_paths import STIMULI

    sessions = {}
    for name, (description, outputs) in SESSIONS.items():
        sessions[name] = (outputs, task_from(description, RATINGS if description['kind'] == 'scale' else FITTING),
                          STIMULI)
    sessions.update({name: (*case, STIMULI) for name, case in _cases().items()})
    sessions.update(stage3_cases())
    found = {}
    for parts in sorted(REGISTERED_PART_SETS):
        model = _model(*parts, owners=REGISTERED_STATEMENTS[parts])
        entry = {'out_channels': sorted(model.out_channels), 'sessions': {}}
        for name in sorted(sessions):
            outputs, task, stimuli = sessions[name]
            session = stimulus_session(stimuli, outputs, task)
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
