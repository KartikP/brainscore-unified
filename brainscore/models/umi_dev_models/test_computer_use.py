"""Qwen3-VL's computer-use format: the reader and its translation to motor acts (offline)."""
import json

import pytest

from brainscore.models.umi_dev_models.computer_use import MAX_ACTS, SYSTEM, read

pytestmark = pytest.mark.unit
SIZE = (1024, 768)


def call(**arguments):
    return '<tool_call>\n' + json.dumps({'name': 'computer_use', 'arguments': arguments}) + '\n</tool_call>'


def test_a_click_on_the_grid_lands_on_the_face_in_pixels():
    acts, problems = read(call(action='left_click', coordinate=[680, 170]), 'motor:pointer', (0, 0), SIZE)
    assert problems == [] and acts == [{'act': 'move', 'x': 696.32, 'y': 130.56}, {'act': 'down'}, {'act': 'up'}]


def test_a_drag_starts_where_the_pointer_is_and_calls_run_in_order():
    reply = call(action='mouse_move', coordinate=[789, 633]) + '\n' + call(action='left_click_drag', coordinate=[500, 500])
    acts, _ = read(reply, 'motor:pointer', (0, 0), SIZE)
    assert [a['act'] for a in acts] == ['move', 'down', 'stroke', 'up'] and acts[2]['points'] == [[512.0, 384.0]]


def test_double_and_triple_clicks_are_clicks_in_place():
    acts, _ = read(call(action='double_click', coordinate=[500, 500]), 'motor:pointer', (0, 0), SIZE)
    assert [a['act'] for a in acts] == ['move', 'down', 'up', 'down', 'up']


def test_keys_map_to_w3c_codes_and_release_in_reverse():
    acts, _ = read(call(action='key', keys=['Return']), 'motor:key', (0, 0), SIZE)
    assert acts == [{'act': 'press', 'key': 'Enter'}, {'act': 'release', 'key': 'Enter'}]
    acts, _ = read(call(action='key', keys=['left']), 'motor:key', (0, 0), SIZE)
    assert acts[0] == {'act': 'press', 'key': 'ArrowLeft'}


@pytest.mark.parametrize('reply, named', [
    (call(action='right_click', coordinate=[1, 1]), 'right_click'),
    (call(action='key', keys=['F13']), 'F13'),
    (call(action='left_click', coordinate=[1, 1]), 'left_click'),          # a pointer action on the key form
    ('I will click the face.', 'tool_call'),
    (call(action='wait', time=0.01) * (MAX_ACTS + 1), 'at most'),
])
def test_what_the_task_cannot_take_is_an_invalid_answer_with_its_name(reply, named):
    acts, problems = read(reply, 'motor:key', (0, 0), SIZE)
    assert acts is None and named in problems[0]


def test_the_system_turn_is_the_authors_tool_specification():
    assert SYSTEM.startswith('You are a helpful assistant.\n\n# Tools') and "The screen's resolution is 1000x1000." in SYSTEM


def test_a_subject_takes_part_in_its_own_format_through_participate(tmp_path, monkeypatch):
    import sys
    import brainscore.harnesses.nodekit as nodekit
    from brainscore import BrainScoreModel
    from brainscore.models.umi_dev_models.computer_use import computer_use_policy
    from tests.test_harnesses.test_nodekit import STAND_IN, _program
    folder = tmp_path / 'environments' / 'nodekit' / 'ready'
    (folder / 'env' / 'bin').mkdir(parents=True)
    (folder / 'env' / 'bin' / 'python').symlink_to(sys.executable)
    (tmp_path / 'worker.py').write_text(STAND_IN)
    monkeypatch.setattr(nodekit, 'library', lambda: None)
    monkeypatch.setattr(nodekit, 'ready_folder', lambda library, identifier, given=None: folder)
    monkeypatch.setattr(nodekit, '_WORKER', tmp_path / 'worker.py')
    prompts = []
    policy = computer_use_policy(lambda prompt: prompts.append(prompt) or call(action='key', keys=['left']))
    subject = BrainScoreModel('format', policy=policy, policy_in_channels=('vision:image', 'observation'),
                              policy_out_channels=('motor:key',))
    result, _, _ = nodekit.play(subject, _program(tmp_path / 'keys'), form='motor:key', pacing='dense',
                                dense_step_ms=250)
    assert result[0]['response'] == 'ArrowLeft'
    assert prompts[0].parts[1][1].startswith('Do the task on the screen. This task takes keyboard actions only')


def test_frames_as_video_gives_every_frame_of_the_turn_as_one_lossless_clip_through_participate(tmp_path, monkeypatch):
    import sys
    import numpy as np
    import brainscore.harnesses.nodekit as nodekit
    from brainscore import BrainScoreModel
    from brainscore.model_helpers.vlm_video_wrapper import read_clip
    from brainscore.models.umi_dev_models.computer_use import computer_use_policy
    from tests.test_harnesses.test_nodekit import STAND_IN, _program
    folder = tmp_path / 'environments' / 'nodekit' / 'ready'
    (folder / 'env' / 'bin').mkdir(parents=True)
    (folder / 'env' / 'bin' / 'python').symlink_to(sys.executable)
    (tmp_path / 'worker.py').write_text(STAND_IN)
    monkeypatch.setattr(nodekit, 'library', lambda: None)
    monkeypatch.setattr(nodekit, 'ready_folder', lambda library, identifier, given=None: folder)
    monkeypatch.setattr(nodekit, '_WORKER', tmp_path / 'worker.py')
    shown, seen = [], []

    def reply(prompt):
        kinds = [kind for kind, _ in prompt.parts]
        if 'video' in kinds:      # read while the turn's clip exists
            shown.append(read_clip(prompt.parts[kinds.index('video')][1]))
        seen.append(kinds)
        return call(action='wait', time=0.25) if len(seen) == 1 else call(action='key', keys=['left'])
    policy = computer_use_policy(reply, frames_as='video')
    subject = BrainScoreModel('format', policy=policy, policy_out_channels=('motor:key',),
                              policy_in_channels=('vision:image', 'vision:video', 'observation'))
    result, _, _ = nodekit.play(subject, _program(tmp_path / 'keys'), form='motor:key', pacing='dense',
                                dense_step_ms=250)
    assert result[0]['response'] == 'ArrowLeft'
    # The 250 ms wait passed the stand-in's 14 frame times: one clip of all of them at 60 a second, then the screen.
    (frames, rate), = shown
    assert seen[1][-3:] == ['video', 'text', 'image'] and seen[1].count('image') == 1
    assert rate == 60.0 and frames.shape == (14, 384, 512, 3) and len(policy.trace[1]['frames_shown']) == 14


def test_frames_as_video_keeps_the_frames_exactly_and_states_the_form():
    import numpy as np
    from brainscore.model_helpers.vlm_video_wrapper import read_clip
    from brainscore.models.umi_dev_models.computer_use import computer_use_policy
    video = (np.random.default_rng(0).random((15, 384, 512, 3)) * 255).astype(np.uint8)
    row = {'vision:image': np.zeros((768, 1024, 3), np.uint8), 'vision:video': video,
           'observation': {'act_form': 'motor:key', 'task': 'Answer.', 'frame_t_ms': [16.667 * k for k in range(1, 16)]}}
    clips = []

    def reply(prompt):
        clips.extend(read_clip(value) for kind, value in prompt.parts if kind == 'video')
        return call(action='key', keys=['left'])
    policy = computer_use_policy(reply, frames_as='video')
    policy(row)
    (frames, rate), = clips
    assert np.array_equal(frames, video) and abs(rate - 60.0) < 0.01
    stills = []
    computer_use_policy(lambda prompt: stills.append([k for k, _ in prompt.parts]) or call(action='key', keys=['left']))(row)
    assert 'video' not in stills[0] and stills[0].count('image') == 1 + 4      # default: the changed frames as pictures
    with pytest.raises(ValueError, match='frames_as'):
        computer_use_policy(reply, frames_as='film')
