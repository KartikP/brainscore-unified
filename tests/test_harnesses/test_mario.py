"""The Super Mario Bros. environment without the emulator: its actions, its clear rule, the ROM check, and a
play through environment_session against a stand-in worker that speaks the worker's protocol."""
import hashlib
import sys

import numpy as np
import pytest

from brainscore.harnesses import mario

pytestmark = pytest.mark.unit

# A stand-in worker: the protocol of _worker.py, frames of zeros, Mario walking 30 scroll units a step;
# the scenario ends the play after 6 steps at the flag.
STAND_IN = r'''
import base64, json, os, sys
read_fd, write_fd = (int(fd) for fd in os.environ['BRAINSCORE_CHILD_FDS'].split(','))
commands, messages = os.fdopen(read_fd, 'rb'), os.fdopen(write_fd, 'wb')
assert sys.argv[1] == 'SuperMarioBros-Nes' and sys.argv[2] == 'Level1-1' and len(sys.argv) == 6
step = 0
while line := commands.readline():
    command = json.loads(line)
    step = 0 if 'reset' in command else step + 1
    n = 1 if 'reset' in command else command['frames']
    scroll = 40 + 30 * step
    variables = {'xscrollHi': scroll // 256, 'xscrollLo': scroll % 256, 'lives': 2, 'player_y_screen': 1,
                 'player_state': 8, 'held': len(command.get('step', []))}
    message = {'frames': base64.b64encode(bytes(n * 224 * 240 * 3)).decode(), 'shape': [n, 224, 240, 3],
               'sound': base64.b64encode(bytes(2 * 267 * n)).decode(), 'sound_rate': 32000.0,
               'variables': variables, 'done': step == 6}
    messages.write((json.dumps({'done': message}) + '\n').encode()); messages.flush()
'''


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    """A ROM-like file whose SHA1 is the expected one here, a folder whose Python is this one, a state file."""
    rom = tmp_path / 'game.nes'
    rom.write_bytes(b'made-up cartridge')
    monkeypatch.setattr(mario, 'ROM_SHA1', hashlib.sha1(rom.read_bytes()).hexdigest())
    folder = tmp_path / 'folder'
    (folder / 'env' / 'bin').mkdir(parents=True)
    (folder / 'env' / 'bin' / 'python').symlink_to(sys.executable)
    monkeypatch.setattr(mario, 'ready_folder', lambda library, identifier, folder_=None: folder)
    monkeypatch.setattr(mario, 'level_state', lambda level: tmp_path / f'{level}.state')
    worker = tmp_path / 'stand_in_worker.py'
    worker.write_text(STAND_IN)
    monkeypatch.setattr(mario, '_WORKER', worker)
    return rom


def test_each_action_holds_its_buttons_in_the_order_of_the_emulator():
    assert mario.buttons(0) == [0] * 9
    assert mario.buttons(4) == [1, 0, 0, 0, 0, 0, 0, 1, 1]          # RIGHT + A + B: B first, A last
    assert len(mario.ACTIONS) == 12


def test_a_play_is_cleared_as_the_participants_replays_count_it():
    assert mario.cleared(y_screen=1, lives=2, player_state=8)
    assert not mario.cleared(y_screen=2, lives=2, player_state=8)    # fell below the screen
    assert not mario.cleared(y_screen=1, lives=-1, player_state=8)   # out of lives
    assert not mario.cleared(y_screen=1, lives=2, player_state=11)   # dying


@pytest.mark.parametrize('content, match', [(None, 'no ROM'), (b'another cartridge', 'SHA1')])
def test_a_missing_or_other_rom_is_refused_before_any_work(tmp_path, monkeypatch, content, match):
    made = []
    monkeypatch.setattr(mario, 'ready_folder', lambda *args, **kwargs: made.append(args))
    monkeypatch.setattr(mario, 'level_state', lambda level: made.append(level))
    rom = tmp_path / 'game.nes'
    if content is not None:
        rom.write_bytes(content)
    with pytest.raises((FileNotFoundError, ValueError), match=match):
        mario.MarioEnvironment(rom)
    assert not made


def test_the_library_pins_the_integration_and_never_its_rom():
    assert mario.MARIO.commit == 'e3ecccf76d7776733ac1a98cf997c97e97f65f66'
    assert not any('rom.nes' in path for path in mario.MARIO.required_files)
    assert mario.ROM_SHA1 == 'facee9c577a5262dbe33ac4930bb0b58c8c037f7'


def test_an_agent_plays_through_environment_session_and_its_layers_are_recorded(stand_in):
    from brainscore import load_model
    progress, session = mario.play_level(load_model('mario-ppo-untrained'), stand_in, 'Level1-1',
                                         outputs=('motor', 'neural:cortex'))
    assert progress['steps'] == 6 and progress['ended_by'] == 'scenario'
    assert progress['x_traveled'] == 6 * 30 and progress['cleared']
    assert session.collect('neural:cortex').sizes['presentation'] == 6     # one recording per action
