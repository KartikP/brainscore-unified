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
def screen(system=None, start=None, file=None):
    ended = answer is not None or left
    reply = {'t_ms': t, 'ended': ended, 'body': {'pointer': [0.0, 0.0], 'pointer_down': False,
             'pointer_connected': True, 'keys_down': []}, 'extent': [width, height],
             'acts_per_step': 20}
    if not ended:
        reply['picture'] = {'data': base64.b64encode(bytes([128]) * (width * height * 3)).decode(),
                            'shape': [height, width, 3], 'at_ms': t}
    if file:
        times = [k * 1000 / 60 for k in range(1, 100000) if start < k * 1000 / 60 < t]
        open(file, 'wb').write(bytes([64]) * (len(times) * 512 * 384 * 3))
        reply['frames'] = {'file': file, 'shape': [len(times), 384, 512, 3], 'at_ms': times, 'not_sent': 0}
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
        send({'done': screen(start=start, file=step.get('frames'))})
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
    assert seen[1]['vision:video'].shape == (14, 384, 512, 3) and len(seen[1]['observation']['frame_t_ms']) == 14
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


def test_dense_pacing_keeps_the_normal_action_limit(stand_in):
    policy, seen = _keys('ArrowLeft')
    result, environment, _ = nodekit.play(_subject('keys', policy), _program(stand_in / 'keys'), form='motor:key',
                                          pacing='dense', dense_step_ms=250)
    assert result is not None and environment.acts_per_step == 20       # a press and its release in one turn
    assert seen[0]['observation']['acts_per_step'] == 20


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


# --- the act policy, with a stand-in for nkl's reader at its child-process boundary ---------------------------
READER = '''
import json, os
read_fd, write_fd = (int(fd) for fd in os.environ['BRAINSCORE_CHILD_FDS'].split(','))
commands, messages = os.fdopen(read_fd, 'rb'), os.fdopen(write_fd, 'wb')
def send(message):
    messages.write((json.dumps(message) + '\\n').encode()); messages.flush()
while line := commands.readline():
    command = json.loads(line)
    if 'system' in command:
        width, height = command['system']['grid']
        send({'done': f'Positions are on a grid laid over it: {width} x {height}.'})
    elif 'brief' in command:
        turn = command['brief']
        send({'done': [['problems: ' + '; '.join(turn['problems']), None], ['The screen now:', 'screen']]})
    else:
        turn = command['read']
        if turn['text'] == 'look':
            send({'done': {'acts': None, 'memories': [], 'problems': ['command 1 (look): look is not offered here']}})
        else:
            send({'done': {'acts': json.loads(turn['text']), 'memories': [], 'problems': []}})
'''


def test_the_act_policy_reads_each_reply_and_shows_its_problems_and_changed_frames(stand_in, monkeypatch):
    import brainscore.harnesses._environment_folder as environment_folder
    import brainscore.harnesses.nodekit.policy as act
    reader = stand_in / 'stand_in_reader.py'
    reader.write_text(READER)
    monkeypatch.setattr(act, '_READER', reader)
    monkeypatch.setattr(environment_folder, 'ready_folder', nodekit.ready_folder)
    replies = iter(['look', '[{"act": "wait", "ms": 250}]',
                    '[{"act": "press", "key": "ArrowLeft"}, {"act": "release", "key": "ArrowLeft"}]'])
    prompts = []

    def reply(prompt):
        prompts.append(prompt)
        return next(replies)
    policy = act.act_policy(reply)
    subject = BrainScoreModel('text-subject', policy=policy,
                              policy_in_channels=('vision:image', 'vision:video', 'instruction', 'observation'),
                              policy_out_channels=('motor:key', 'motor:pointer'))
    result, environment, session = nodekit.play(subject, _program(stand_in / 'keys'), form='motor:key')
    assert result[0]['response'] == 'ArrowLeft' and environment.invalid_steps == 1
    assert prompts[0].names[0] == 'instruction' and prompts[0].parts[0][1] == environment.system
    assert 'look is not offered' in prompts[1].parts[1][1]                     # the problem, at the next turn
    # The 250 ms wait passed 15 frame times; the frames are equal, so one changed frame is shown, then the screen.
    assert prompts[2].names.count('frame') == 2 and prompts[2].names[-1] == 'screen'
    assert [turn['problems'] for turn in policy.trace][:1] == [['command 1 (look): look is not offered here']]
    assert prompts[0].response.temperature is None and prompts[0].response.seed is None
    # The earlier turns come before, as text: the briefing and the reply, no picture.
    assert prompts[0].history == ()
    assert [answered for _, answered in prompts[2].history] == ['look', '[{"act": "wait", "ms": 250}]']
    assert all(kind == 'text' for earlier, _ in prompts[2].history for kind, _ in earlier.parts)
    policy.close()


def test_changed_frames_keeps_frames_that_differ_up_to_the_cap():
    from brainscore.harnesses.nodekit.policy import changed_frames
    video = np.zeros((10, 2, 2, 3), np.uint8)
    video[3:] = 1
    video[7:] = 2
    assert [at for at, _ in changed_frames(video, list(range(10)), None, 4)] == [0, 3, 7]
    video = np.arange(10, dtype=np.uint8)[:, None, None, None] * np.ones((10, 2, 2, 3), np.uint8)
    assert [at for at, _ in changed_frames(video, list(range(10)), video[0], 4)] == [1, 4, 6, 9]


def test_the_outputs_hold_the_act_form(stand_in):
    policy, _ = _keys('ArrowLeft')
    with pytest.raises(ValueError, match='act form'):
        nodekit.play(_subject('keys', policy), _program(stand_in / 'keys'), form='motor:key', outputs=['neural:IT'])


def test_the_act_policy_gives_each_turn_the_sampling_settings():
    from brainscore.harnesses.nodekit.policy import act_policy
    policy = act_policy(lambda prompt: '', temperature=0.7, seed=3)
    assert callable(policy.reset) and policy.trace == []


def test_chrome_on_linux_is_wrapped_once_with_the_gpu_flags(tmp_path):
    chrome = tmp_path / 'chrome'
    chrome.write_text('the program')
    nodekit.gpu_chrome(chrome)
    nodekit.gpu_chrome(chrome)
    assert (tmp_path / 'chrome.real').read_text() == 'the program'
    assert '--use-angle=vulkan' in chrome.read_text() and 'chrome.real' in chrome.read_text()


def test_a_run_stops_after_answers_in_a_row_that_give_no_act(stand_in):
    subject = _subject('mute', lambda row: {'motor:key': None})
    result, environment, session = nodekit.play(subject, _program(stand_in / 'mute'), form='motor:key')
    assert session.source.end_reason == 'truncated' and environment.stop_cause == 'invalid answers in a row'
    assert environment.steps == nodekit.MAX_INVALID_IN_A_ROW
    environment.reset()                 # a reused environment starts its next run with no count and no cause
    try:
        assert environment.stop_cause is None and environment.invalid_in_a_row == environment.same_in_a_row == 0
    finally:
        environment.close()


def test_the_act_policy_compares_frames_with_the_last_frame_and_cleans_its_pictures(stand_in, monkeypatch):
    import brainscore.harnesses._environment_folder as environment_folder
    import brainscore.harnesses.nodekit.policy as act
    reader = stand_in / 'stand_in_reader.py'
    reader.write_text(READER)
    monkeypatch.setattr(act, '_READER', reader)
    monkeypatch.setattr(environment_folder, 'ready_folder', nodekit.ready_folder)
    prompts = []
    policy = act.act_policy(lambda prompt: prompts.append(prompt) or '[{"act": "wait", "ms": 100}]')
    still = np.zeros((3, 384, 512, 3), np.uint8)
    row = {'vision:image': np.zeros((768, 1024, 3), np.uint8), 'vision:video': still, 'instruction': 'brief',
           'observation': {'t_ms': 0.0, 'frame_t_ms': [1.0, 2.0, 3.0], 'act_form': 'motor:key'}}
    policy(row)
    policy(row)                         # the same frames as the last frame before: none changed
    assert prompts[0].names.count('frame') == 2 and prompts[1].names.count('frame') == 0
    pictures = Path(prompts[0].parts[-1][1]).parent
    assert pictures.is_dir()
    policy.reset()
    assert not pictures.exists()
    policy.close()


def test_the_agent_view_names_its_pictures_and_a_look_is_refused_in_the_participant_view(stand_in):
    seen = []

    def policy(row):
        seen.append(row)
        return {'motor:key': [{'act': 'look', 'id': 'patch'}, {'act': 'press', 'key': 'ArrowLeft'}]}
    subject = BrainScoreModel('looks', policy=policy, policy_in_channels=('vision:image', 'vision:video', 'observation'),
                              policy_out_channels=('motor:key',))
    nodekit.play(subject, _program(stand_in / 'keys'), form='motor:key', view='agent')
    assert set(seen[0]['vision:image']) == {'screen'} and 'vision:video' not in seen[0]
    assert seen[0]['observation']['view'] == 'agent'
    with pytest.raises(ValueError, match='agent view'):
        nodekit.play(subject, _program(stand_in / 'keys2'), form='motor:key')


def test_the_act_policy_shows_no_earlier_turn_in_the_agent_view(stand_in, monkeypatch):
    import brainscore.harnesses._environment_folder as environment_folder
    import brainscore.harnesses.nodekit.policy as act
    reader = stand_in / 'stand_in_reader.py'
    reader.write_text(READER)
    monkeypatch.setattr(act, '_READER', reader)
    monkeypatch.setattr(environment_folder, 'ready_folder', nodekit.ready_folder)
    prompts = []
    policy = act.act_policy(lambda prompt: prompts.append(prompt) or '[{"act": "wait", "ms": 100}]')
    row = {'vision:image': {'screen': np.zeros((768, 1024, 3), np.uint8)}, 'instruction': 'brief',
           'observation': {'t_ms': 0.0, 'act_form': 'motor:key', 'view': 'agent'}}
    policy(row)
    policy(row)
    assert prompts[1].history == ()
    policy.close()


def test_positions_on_the_model_s_grid_become_pixels():
    from brainscore.harnesses.nodekit.policy import to_pixels
    # The 30B's case: a target it wrote on the grid stays on face 2 (nkl's grid option shows and reads the grid).
    assert to_pixels([{'act': 'stroke', 'points': [[680, 170]]}], 1000, 1024, 768)[0]['points'] == [[696.32, 130.56]]


def test_each_run_has_nkl_data_of_its_own_and_a_shared_home(stand_in):
    one = nodekit.NodeKitEnvironment(_program(stand_in / 'a'), form='motor:key').child_environment()
    two = nodekit.NodeKitEnvironment(_program(stand_in / 'b'), form='motor:key').child_environment()
    assert one['HOME'] == two['HOME'] and one['NKL_HOME'] != two['NKL_HOME']


def test_a_run_stops_after_the_same_acts_in_a_row(stand_in):
    subject = _subject('stuck', lambda row: {'motor:key': [{'act': 'press', 'key': 'Space'}]})
    result, environment, session = nodekit.play(subject, _program(stand_in / 'stuck'), form='motor:key')
    # The same acts on the same screen are a cycle of one step: the cycle rule stops it first (the 40-in-a-row
    # rule stops the same acts on screens that change).
    assert session.source.end_reason == 'truncated' and environment.stop_cause == 'a cycle of the same acts and screens'
    assert environment.steps == nodekit.CYCLE_REPEATS


def test_under_a_grid_the_system_briefing_is_nkl_s_for_its_grid(stand_in, monkeypatch):
    import brainscore.harnesses._environment_folder as environment_folder
    import brainscore.harnesses.nodekit.policy as act
    reader = stand_in / 'stand_in_reader.py'
    reader.write_text(READER)
    monkeypatch.setattr(act, '_READER', reader)
    monkeypatch.setattr(environment_folder, 'ready_folder', nodekit.ready_folder)
    prompts = []
    policy = act.act_policy(lambda prompt: prompts.append(prompt) or '[]', position_grid=1000)
    row = {'vision:image': np.zeros((768, 1024, 3), np.uint8), 'instruction': 'The screen image is 1024 x 768 pixels.',
           'observation': {'t_ms': 0.0, 'act_form': 'motor:key', 'pointer': [0.0, 0.0],
                           'briefing': {'pacing': ['Frozen', 16667], 'limits': {'turns': 5}, 'seconds': 1,
                                        'instructions': ''}}}
    policy(row)
    system = prompts[0].parts[0][1]
    assert '1000 x 1000' in system and system != row['instruction']
    policy.close()


def test_a_run_stops_after_a_cycle_of_the_same_acts_and_screens(stand_in):
    cycle = [[{'act': 'press', 'key': 'KeyA'}], [{'act': 'press', 'key': 'KeyB'}], [{'act': 'press', 'key': 'KeyC'}]]
    count = [0]

    def policy(row):
        count[0] += 1
        return {'motor:key': cycle[(count[0] - 1) % 3]}
    result, environment, session = nodekit.play(_subject('loop', policy), _program(stand_in / 'loop'), form='motor:key')
    assert session.source.end_reason == 'truncated' and environment.stop_cause == 'a cycle of the same acts and screens'
    assert environment.steps == 3 * nodekit.CYCLE_REPEATS


def test_a_cycle_needs_the_same_screens_too():
    same = [('a', 's1'), ('b', 's2')] * 5
    assert nodekit.repeats_a_cycle(same)
    assert not nodekit.repeats_a_cycle([('a', f's{i}') for i in range(10)])      # the same acts, new screens
    assert not nodekit.repeats_a_cycle([('null', 's')] * 10)                     # no act: the invalid-answer rule


def test_the_looks_nkl_keeps_are_a_stated_setting(stand_in):
    environment = nodekit.NodeKitEnvironment(_program(stand_in / 'a'), form='motor:key', view='agent', looks_kept=2)
    assert environment.looks_kept == 2
    with pytest.raises(ValueError, match='looks_kept'):
        nodekit.NodeKitEnvironment(_program(stand_in / 'b'), form='motor:key', looks_kept=-1)
