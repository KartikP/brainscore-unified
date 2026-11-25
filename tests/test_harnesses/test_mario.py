"""The Super Mario Bros. environment without the emulator: its actions, its clear rule and the ROM it needs."""
import pytest

from brainscore.harnesses import mario

pytestmark = pytest.mark.unit


def test_each_action_holds_its_buttons_in_the_order_of_the_emulator():
    assert mario.buttons(0) == [0] * 9
    assert mario.buttons(4) == [1, 0, 0, 0, 0, 0, 0, 1, 1]          # RIGHT + A + B: B first, A last
    assert len(mario.ACTIONS) == 12 and mario.MarioEnvironment.action_spec(None).count == 12


def test_a_play_is_cleared_as_the_participants_replays_count_it():
    assert mario.cleared(y_screen=1, lives=2, player_state=8)
    assert not mario.cleared(y_screen=2, lives=2, player_state=8)    # fell below the screen
    assert not mario.cleared(y_screen=1, lives=-1, player_state=8)   # out of lives
    assert not mario.cleared(y_screen=1, lives=2, player_state=11)   # dying


def test_a_missing_rom_is_refused_before_any_folder_is_made(tmp_path, monkeypatch):
    made = []
    monkeypatch.setattr(mario, 'ready_folder', lambda *args, **kwargs: made.append(args))
    with pytest.raises(FileNotFoundError, match='ROM'):
        mario.MarioEnvironment(tmp_path / 'absent.nes')
    assert not made


def test_the_library_pins_the_integration_and_never_its_rom():
    assert mario.MARIO.commit == 'e3ecccf76d7776733ac1a98cf997c97e97f65f66'
    assert not any('rom.nes' in path for path in mario.MARIO.required_files)
    assert 'stable-retro==0.9.9' in mario.MARIO.requirements.read_text().split()
