"""The calls each session makes to the parts and the extractor, from the code at hand.

Run from the unified root to regenerate ``tests/fixtures/session_call_log.json``::

    python -m tests.brainscore_model.session_call_log

The fixture was written from the code of the commit before the session plan
(unified ``1808a13``). For each part set of ``stage2_answers_baseline`` and each
session there (as a table, as a stream, with a region recorded during the task,
and for a subject that keeps its trial history), and for sessions that record two
regions under both pass plans, it records the calls in order: ``answer``,
``reply`` and ``likelihood`` with the stimulus, and ``extract`` with the layers
and the stimuli. A refused session records ``refused``. The stand-in parts are
fixed functions: no weights, no downloads.
"""
import json
import os
import warnings
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'session_call_log.json'
REGIONS = {'IT': 'late', 'V4': 'early'}


def _model(parts, owners, log, memory=None):
    from brainscore import BrainScoreModel
    from tests.brainscore_model.test_behavior_paths import _features

    def answer(stimulus_row, instruction, label_set):
        log.append(['answer', str(stimulus_row['stimulus_id'])])
        return label_set[0]

    def reply(prompt):
        log.append(['reply', str(prompt.stimulus_id)])
        return 'cat'

    def likelihood(prompt, labels):
        log.append(['likelihood', str(prompt.stimulus_id)])
        return {label: -float(i) for i, label in enumerate(labels)}

    def extract(stimuli, layers, **kwargs):
        log.append(['extract', list(layers), [str(s) for s in stimuli['stimulus_id']]])
        return _features(stimuli, layers, **kwargs)
    available = {'answer': answer, 'reply': reply, 'likelihood': likelihood,
                 'behavioral_readout_layer': 'late'}
    config = {'capability_config': {'behavior-sessions': owners}} if owners else {}
    return BrainScoreModel('logged', extractor={'text': extract}, in_channels={'text'},
                           region_layer_map=dict(REGIONS), memory=memory,
                           **{part: available[part] for part in parts}, **config)


def _sessions():
    """name -> (outputs, task, stimuli) of the answer baseline."""
    from tests.brainscore_model.stage2_answers_baseline import (
        FITTING, RATINGS, SESSIONS, _cases, stage3_cases, task_from)
    from tests.brainscore_model.test_behavior_paths import STIMULI
    sessions = {name: (outputs, task_from(description, RATINGS if description['kind'] == 'scale' else FITTING),
                       STIMULI) for name, (description, outputs) in SESSIONS.items()}
    sessions.update({name: (*case, STIMULI) for name, case in _cases().items()})
    sessions.update(stage3_cases())
    return sessions


def _run(model, log, outputs, task, stimuli, streaming=False, plan='all'):
    from brainscore_core.streaming_helpers import stimulus_session
    previous = os.environ.get('BRAINSCORE_REGIONS_PER_PASS')
    os.environ['BRAINSCORE_REGIONS_PER_PASS'] = plan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.participate(stimulus_session(stimuli, outputs, task, streaming=streaming))
        return {'calls': log, 'refused': False}
    except Exception:            # a refusal is part of the log
        return {'calls': log, 'refused': True}
    finally:
        if previous is None:
            os.environ.pop('BRAINSCORE_REGIONS_PER_PASS', None)
        else:
            os.environ['BRAINSCORE_REGIONS_PER_PASS'] = previous


def capture():
    from tests.brainscore_model.test_answer_ownership import REGISTERED_PART_SETS, REGISTERED_STATEMENTS
    from tests.brainscore_model.test_behavior_paths import STIMULI
    found = {}
    sessions = _sessions()
    for parts in sorted(REGISTERED_PART_SETS):
        owners = REGISTERED_STATEMENTS[parts]
        for name in sorted(sessions):
            outputs, task, stimuli = sessions[name]
            variants = {'table': ({}, outputs), 'stream': ({'streaming': True}, outputs),
                        'with region': ({}, [*outputs, 'neural:IT']), 'history': ({}, outputs)}
            for variant, (settings, requested) in variants.items():
                log = []
                memory = 'trial_history' if variant == 'history' else None
                found[f"{'+'.join(parts)} | {name} | {variant}"] = _run(
                    _model(parts, owners, log, memory), log, requested, task, stimuli, **settings)
    for plan in ('all', 'one'):
        for streaming in (False, True):
            log = []
            found[f'neural | IT and V4 | {plan} | {"stream" if streaming else "table"}'] = _run(
                _model((), {}, log), log, ['neural:IT', 'neural:V4'], None, STIMULI,
                streaming=streaming, plan=plan)
    return found


def render(found):
    return json.dumps(found, sort_keys=True, indent=1) + '\n'


if __name__ == '__main__':
    found = capture()
    FIXTURE.write_text(render(found), encoding='utf-8')
    print(f'{FIXTURE}: {len(found)} sessions')
