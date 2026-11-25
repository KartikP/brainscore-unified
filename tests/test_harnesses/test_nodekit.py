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
import base64, json, math, os, sys
read_fd, write_fd = (int(fd) for fd in os.environ['BRAINSCORE_CHILD_FDS'].split(','))
commands, messages = os.fdopen(read_fd, 'rb'), os.fdopen(write_fd, 'wb')
width, height, frame_ms, video = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), (int(sys.argv[6]), int(sys.argv[7]))
answer, t, left = None, 0.0, False
def send(message):
    messages.write((json.dumps(message) + '\\n').encode()); messages.flush()
def pictures(n, w, h):
    return {'data': base64.b64encode(bytes([128]) * (n * w * h * 3)).decode(), 'shape': [n, h, w, 3]}
def screen(frames):
    ended = answer is not None or left
    return {'frames': pictures(frames, *video) if frames else None,
            'picture': None if ended else pictures(1, width, height), 't_ms': t, 'ended': ended}
while line := commands.readline():
    command = json.loads(line)
    if 'reset' in command:
        answer, t, left = None, 0.0, False
        send({'done': screen(0)})
    elif 'step' in command:
        start = t
        for act in command['step']:
            t += act.get('ms', 1.0)
            if act['act'] == 'KeyPress' and act['key'] in ('ArrowLeft', 'ArrowRight'):
                answer = act['key']
        frames = math.floor(t / frame_ms + 1e-9) - math.floor(start / frame_ms + 1e-9)
        if frames == 0:
            t, frames = (math.floor(t / frame_ms + 1e-9) + 1) * frame_ms, 1
        send({'done': screen(frames)})
    elif 'leave' in command:
        left = True
        send({'done': True})
    elif 'trace' in command:
        open(command['trace'], 'w').write(json.dumps([[0, 0, 'TraceStart', {}], [1, int(t * 1000), 'TraceEnd', {}]]))
        send({'done': command['trace']})
    else:
        send({'done': [{'response': answer, 'program': sys.argv[1]}] if answer else None})
'''


def _program(folder: Path, imports) -> Path:
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


def test_a_subject_takes_part_through_participate_and_the_program_result_comes_back(stand_in):
    program = _program(stand_in / 'keys', ['keyboard.inputs', 'time.elapsed'])
    policy, seen = _keys('ArrowLeft')
    subject = BrainScoreModel('keys', policy=policy, policy_in_channels=('vision:image', 'vision:video', 'instruction'),
                              policy_out_channels=('motor:key',))
    result, environment, session = nodekit.play(subject, program, form='motor:key')
    assert result == [{'response': 'ArrowLeft', 'program': str(program)}]
    assert session.source.end_reason == 'terminal'
    picture = seen[0]['vision:image']
    assert picture.shape == (768, 1024, 3) and picture.dtype == np.uint8
    assert 'press' in seen[0]['instruction'] and 'stroke' not in seen[0]['instruction']
    # The clock moved by the acts only, and each step to at least the next frame time.
    assert seen[1]['vision:video'].shape == (15, 384, 512, 3)         # 250 ms at 60 frames a second
    assert 'vision:video' not in seen[0]                              # the first screen has no frames before it
    assert environment.t_ms >= 252.0


def test_an_act_of_the_other_form_is_refused_before_the_program_gets_it(stand_in):
    program = _program(stand_in / 'keys-only', ['keyboard.inputs'])
    subject = BrainScoreModel('pointer', policy=lambda row: {'motor:key': [{'act': 'down'}]},
                              policy_in_channels=('vision:image',), policy_out_channels=('motor:key',))
    with pytest.raises(ValueError, match='down'):
        nodekit.play(subject, program, form='motor:key')


def test_a_run_that_does_not_end_is_truncated_and_has_no_result(stand_in):
    program = _program(stand_in / 'waits', ['keyboard.inputs'])
    subject = BrainScoreModel('waits', policy=lambda row: {'motor:key': [{'act': 'wait', 'ms': 10}]},
                              policy_in_channels=('vision:image',), policy_out_channels=('motor:key',))
    result, environment, session = nodekit.play(subject, program, form='motor:key', max_steps=3, record=stand_in / 'record',
                                                name='waits-run')
    assert result is None and session.source.end_reason == 'truncated' and environment.steps == 3
    assert environment.trace == stand_in / 'record' / 'waits' / 'submissions' / 'waits-run.json'
    assert json.loads(environment.trace.read_text())[0][2] == 'TraceStart'


def test_an_empty_answer_is_a_valid_no_act_and_none_is_invalid_and_both_move_one_frame(stand_in):
    program = _program(stand_in / 'idle', ['keyboard.inputs'])
    answers = iter([[], None, [], None])
    subject = BrainScoreModel('idle', policy=lambda row: {'motor:key': next(answers)}, policy_in_channels=('vision:image',), policy_out_channels=('motor:key',))
    result, environment, session = nodekit.play(subject, program, form='motor:key', max_ms=60)
    assert session.source.end_reason == 'truncated' and environment.invalid_steps == 2
    assert environment.steps == 4 and environment.t_ms == pytest.approx(4 * 1000 / 60)


def test_a_subject_that_takes_only_pictures_gets_the_last_screen_only(stand_in):
    program = _program(stand_in / 'keys', ['keyboard.inputs'])
    policy, seen = _keys('ArrowRight')
    nodekit.play(BrainScoreModel('pictures', policy=policy, policy_in_channels=('vision:image',), policy_out_channels=('motor:key',)), program, form='motor:key')
    assert all('vision:video' not in row and row['vision:image'].shape == (768, 1024, 3) for row in seen)


def test_pixels_of_the_picture_become_nkl_scene_units_in_one_place():
    acts = nodekit.nkl_acts([{'act': 'move', 'x': 952, 'y': 734}, {'act': 'down'},
                             {'act': 'stroke', 'points': [[512, 384]]}, {'act': 'press', 'key': 'Space'}],
                            (1024, 768))
    assert acts == [{'act': 'PointerMove', 'x': 440, 'y': -350}, {'act': 'PointerDown'},
                    {'act': 'PointerStroke', 'points': [[0, 0]]}, {'act': 'KeyPress', 'key': 'Space'}]
