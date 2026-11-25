"""The NodeKit harness with a stand-in at the child-process boundary: a small worker that speaks the
worker's protocol without nkl (one question; an arrow key answers it). No clone, no nkl, no browser."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import brainscore.harnesses.nodekit as nodekit
from brainscore import BrainScoreModel

pytestmark = pytest.mark.unit

STAND_IN = '''
import base64, json, os, sys
read_fd, write_fd = (int(fd) for fd in os.environ['BRAINSCORE_CHILD_FDS'].split(','))
commands, messages = os.fdopen(read_fd, 'rb'), os.fdopen(write_fd, 'wb')
width, height = int(sys.argv[3]), int(sys.argv[4])
answer, t, left, settings, log = None, 0.0, False, None, []
def send(message):
    messages.write((json.dumps(message) + '\\n').encode()); messages.flush()
def screen(system=None, start=None):
    ended = answer is not None or left
    reply = {'t_ms': t, 'ended': ended, 'body': {'pointer': [0.0, 0.0], 'pointer_down': False,
             'pointer_connected': True, 'keys_down': []}, 'extent': [width, height],
             'acts_per_step': 1 if settings['pacing'][0] == 'Dense' else 20}
    if not ended:
        reply['picture'] = {'data': base64.b64encode(bytes([128]) * (width * height * 3)).decode(),
                            'shape': [height, width, 3], 'at_ms': t}
    times = [k * 1000 / 60 for k in range(1, 100000) if start is not None and start < k * 1000 / 60 <= t]
    if times:
        reply['frames'] = {'data': base64.b64encode(bytes([64]) * (len(times) * 512 * 384 * 3)).decode(),
                           'shape': [len(times), 384, 512, 3], 'at_ms': times}
    if system is not None:
        reply['system'] = system
    return reply
while line := commands.readline():
    command = json.loads(line)
    if 'reset' in command:
        answer, t, left, settings = None, 0.0, False, command['reset']
        send({'done': screen(f"pacing {settings['pacing'][0]}; seconds {settings['seconds']}; " + settings['instructions'])})
    elif 'step' in command:
        step, start = command['step'], t
        log.append(step)
        t += step['seconds'] * 1000 if settings['pacing'][0] == 'Realtime' else 0.0
        for act in step['acts']:
            t += act.get('ms', 1.0)
            if act['act'] == 'press' and act['key'] in ('ArrowLeft', 'ArrowRight'):
                answer = act['key']
        send({'done': screen(start=start)})
    elif 'leave' in command:
        left = True
        send({'done': True})
    elif 'trace' in command:
        open(command['trace'], 'w').write(json.dumps([[0, 0, 'TraceStart', {}]] + [[1, 0, 'Step', s] for s in log]))
        send({'done': command['trace']})
    else:
        send({'done': [{'response': answer, 'program': sys.argv[1]}] if answer else None})
'''


def _program(folder: Path, imports=("keyboard.inputs",)) -> Path:
    folder.mkdir(parents=True)
    (folder / 'program.json').write_text(json.dumps({'program': {'format': 'nkl-ir/8', 'imports': imports},
                                                     'arguments': [], 'sidecar': {}}))
    return folder


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    """nkl's folder with this Python as its environment's Python, and the stand-in worker."""
    folder = tmp_path / 'environments' / 'nodekit' / 'ready'
    (folder / 'env' / 'bin').mkdir(parents=True)
    (folder / 'env' / 'bin' / 'python').symlink_to(sys.executable)
    worker = tmp_path / 'stand_in_worker.py'
    worker.write_text(STAND_IN)
    monkeypatch.setattr(nodekit, 'library', lambda: None)
    monkeypatch.setattr(nodekit, 'ready_folder', lambda library, identifier, given=None: folder)
    monkeypatch.setattr(nodekit, '_WORKER', worker)
    return tmp_path


def _keys(answer):
    """A policy that waits on the first screen and then presses ``answer``; it records what it saw."""
    seen = []

    def policy(row):
        seen.append(row)
        if len(seen) == 1:
            return {'motor:key': [{'act': 'wait', 'ms': 250}]}
        return {'motor:key': [{'act': 'press', 'key': answer}, {'act': 'release', 'key': answer}]}
    return policy, seen


def _subject(identifier, policy, form='motor:key'):
    return BrainScoreModel(identifier, policy=policy, policy_in_channels=('vision:image', 'vision:video', 'instruction', 'observation'),
                           policy_out_channels=(form,))


def test_a_subject_takes_part_through_participate_and_the_program_result_comes_back(stand_in):
    program = _program(stand_in / 'keys')
    policy, seen = _keys('ArrowLeft')
    result, environment, session = nodekit.play(_subject('keys', policy), program, form='motor:key',
                                                instructions='Answer with an arrow key.')
    assert result == [{'response': 'ArrowLeft', 'program': str(program)}]
    assert session.source.end_reason == 'terminal'
    picture = seen[0]['vision:image']
    assert picture.shape == (768, 1024, 3) and picture.dtype == np.uint8
    assert seen[0]['instruction'].endswith('Answer with an arrow key.')        # nkl's briefing, then the benchmark's
    assert seen[1]['observation']['t_ms'] == 250.0 and seen[1]['observation']['act_form'] == 'motor:key'
    # The frames since the last turn: one at each frame time the clock passed, at half size, with their times.
    assert seen[1]['vision:video'].shape == (15, 384, 512, 3) and len(seen[1]['observation']['frame_t_ms']) == 15
    assert 'vision:video' not in seen[0]
    assert seen[0]['observation']['turns_left'] == 20000 and seen[1]['observation']['turns_left'] == 19999


def test_the_pacing_is_frozen_by_default_and_a_setting(stand_in):
    program = _program(stand_in / 'keys')
    policy, seen = _keys('ArrowLeft')
    nodekit.play(_subject('keys', policy), program, form='motor:key')
    assert seen[0]['instruction'].startswith('pacing Frozen')
    policy, seen = _keys('ArrowLeft')
    nodekit.play(_subject('keys', policy), program, form='motor:key', pacing='realtime')
    assert seen[0]['instruction'].startswith('pacing Realtime') and seen[1]['observation']['t_ms'] > 250.0
    with pytest.raises(ValueError, match='pacing'):
        nodekit.play(_subject('keys', policy), program, form='motor:key', pacing='fast')


def test_dense_pacing_takes_one_act_a_step(stand_in):
    program = _program(stand_in / 'keys')
    policy, seen = _keys('ArrowLeft')
    with pytest.raises(ValueError, match='one act a step'):
        nodekit.play(_subject('keys', policy), program, form='motor:key', pacing='dense')
    assert seen[0]['observation']['acts_per_step'] == 1


def test_an_act_of_the_other_form_is_refused_before_the_program_gets_it(stand_in):
    program = _program(stand_in / 'keys-only')
    with pytest.raises(ValueError, match='down'):
        nodekit.play(_subject('pointer', lambda row: {'motor:key': [{'act': 'down'}]}), program, form='motor:key')


def test_a_run_that_does_not_end_is_truncated_and_has_no_result(stand_in):
    program = _program(stand_in / 'waits')
    subject = _subject('waits', lambda row: {'motor:key': [{'act': 'wait', 'ms': 10}]})
    result, environment, session = nodekit.play(subject, program, form='motor:key', max_steps=3,
                                                record=stand_in / 'record', name='waits-run')
    assert result is None and session.source.end_reason == 'truncated' and environment.steps == 3
    assert environment.trace == stand_in / 'record' / 'waits' / 'submissions' / 'waits-run.json'
    assert json.loads(environment.trace.read_text())[0][2] == 'TraceStart'


def test_an_empty_answer_is_a_valid_no_act_and_none_is_invalid(stand_in):
    program = _program(stand_in / 'idle')
    answers = iter([[], None, [], None])
    subject = _subject('idle', lambda row: {'motor:key': next(answers)})
    result, environment, session = nodekit.play(subject, program, form='motor:key', max_steps=4)
    assert session.source.end_reason == 'truncated' and environment.invalid_steps == 2
    assert environment.steps == 4 and environment.t_ms == 0.0         # frozen: no act, no time


def test_a_subject_that_takes_only_pictures_gets_the_last_screen_only(stand_in):
    policy, seen = _keys('ArrowRight')
    subject = BrainScoreModel('pictures', policy=policy, policy_in_channels=('vision:image',),
                              policy_out_channels=('motor:key',))
    nodekit.play(subject, _program(stand_in / 'keys'), form='motor:key')
    assert all(set(row) == {'vision:image'} and row['vision:image'].shape == (768, 1024, 3) for row in seen)
