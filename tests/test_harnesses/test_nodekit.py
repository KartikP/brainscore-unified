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
answer, t = None, 0.0
def send(message):
    messages.write((json.dumps(message) + '\\n').encode()); messages.flush()
def screen():
    picture = base64.b64encode(bytes([128]) * (width * height * 3)).decode()
    return {'picture': None if answer else picture, 'shape': None if answer else [height, width, 3],
            't_ms': t, 'ended': answer is not None}
while line := commands.readline():
    command = json.loads(line)
    if 'reset' in command:
        answer, t = None, 0.0
        send({'done': screen()})
    elif 'step' in command:
        for act in command['step']:
            t += act.get('ms', 1.0)
            if act['act'] == 'KeyPress' and act['key'] in ('ArrowLeft', 'ArrowRight'):
                answer = act['key']
        send({'done': screen()})
    else:
        send({'done': [{'response': answer, 'program': sys.argv[1]}]})
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
            return {'motor': [{'act': 'wait', 'ms': 250}]}
        return {'motor': [{'act': 'press', 'key': answer}, {'act': 'release', 'key': answer}]}
    return policy, seen


def test_a_subject_takes_part_through_participate_and_the_program_result_comes_back(stand_in):
    program = _program(stand_in / 'keys', ['keyboard.inputs', 'time.elapsed'])
    policy, seen = _keys('ArrowLeft')
    subject = BrainScoreModel('keys', policy=policy, policy_in_channels=('vision:image', 'instruction'))
    result, environment, session = nodekit.play(subject, program)
    assert result == [{'response': 'ArrowLeft', 'program': str(program)}]
    assert session.source.end_reason == 'terminal'
    picture = seen[0]['vision:image']
    assert picture.shape == (768, 1024, 3) and picture.dtype == np.uint8
    assert 'press' in seen[0]['instruction'] and 'stroke' not in seen[0]['instruction']
    assert environment.t_ms == 252.0         # the clock moved by the acts only


def test_an_act_the_program_does_not_read_fails_at_the_first_action_with_the_forms_it_takes(stand_in):
    program = _program(stand_in / 'keys-only', ['keyboard.inputs'])
    subject = BrainScoreModel('pointer', policy=lambda row: {'motor': [{'act': 'down'}]},
                              policy_in_channels=('vision:image',))
    with pytest.raises(Exception, match='keys-only takes acts') as error:
        nodekit.play(subject, program)
    assert "'press'" in str(error.value)


def test_a_run_that_does_not_end_is_truncated_and_has_no_result(stand_in):
    program = _program(stand_in / 'waits', ['keyboard.inputs'])
    subject = BrainScoreModel('waits', policy=lambda row: {'motor': [{'act': 'wait', 'ms': 10}]},
                              policy_in_channels=('vision:image',))
    result, environment, session = nodekit.play(subject, program, max_steps=3)
    assert result is None and session.source.end_reason == 'truncated' and environment.steps == 3


def test_what_a_program_needs_is_read_from_its_imports(tmp_path):
    assert nodekit.program_needs(_program(tmp_path / 'a', ['keyboard.inputs'])) == {'key', 'time'}
    assert nodekit.program_needs(_program(tmp_path / 'b', ['pointer.state'])) == {'pointer', 'time'}
    both = nodekit.program_needs(_program(tmp_path / 'c', ['pointer.inputs', 'keyboard.inputs']))
    assert {'key', 'pointer'} <= both


def test_pixels_of_the_picture_become_nkl_scene_units_in_one_place():
    acts = nodekit.nkl_acts([{'act': 'move', 'x': 952, 'y': 734}, {'act': 'down'},
                             {'act': 'stroke', 'points': [[512, 384]]}, {'act': 'press', 'key': 'Space'}],
                            (1024, 768))
    assert acts == [{'act': 'PointerMove', 'x': 440, 'y': -350}, {'act': 'PointerDown'},
                    {'act': 'PointerStroke', 'points': [[0, 0]]}, {'act': 'KeyPress', 'key': 'Space'}]
