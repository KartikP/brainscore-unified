"""Public-tool conformance; no weights, service credentials or hardware."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from brainscore.instrumentation import observe, intervene
from brainscore.run_record import RunRecord, RunRecorder
from brainscore.harnesses.robotics import ActionSpec, evaluate_droid_episode
from brainscore_core.events import EnvironmentResponse, StateChange

pytestmark = pytest.mark.unit


def test_nested_observers_restore_instance_override_on_error():
    seen = [[], []]
    subject = SimpleNamespace(process=lambda value: value * 2)
    original = subject.process
    outer = SimpleNamespace(on_result=lambda call, result: seen[0].append(result))
    inner = SimpleNamespace(on_result=lambda call, result: seen[1].append(result))
    with pytest.raises(RuntimeError), observe(subject, outer):
        subject.process(1)
        with observe(subject, inner):
            subject.process(2)
        subject.process(3)
        raise RuntimeError('experiment failed')
    assert subject.process is original
    assert seen == [[2, 4, 6], [4]]


@pytest.mark.parametrize('method,payload', [
    ('digest_text', ['hello', 'world']), ('look_at', ['image-reference']),
    ('process', None)])
def test_witness_preserves_legacy_and_non_dataframe_calls(method, payload):
    from brainscore.witness import Witness
    from brainscore_core.streaming import StreamEvent
    if payload is None:
        payload = StreamEvent('generation_request', {'prompt': 'hello'}, 0)
    original = lambda value: {'result': 'unchanged'}
    subject = SimpleNamespace(**{method: original})
    with Witness(subject) as witness:
        assert getattr(subject, method)(payload) == {'result': 'unchanged'}
    assert len(witness.events) == 1
    assert getattr(subject, method) is original


def test_intervention_removes_only_owned_handle_on_error():
    calls = []
    def process(event):
        calls.append(event)
        return SimpleNamespace(handle_id='owned')
    with pytest.raises(RuntimeError), intervene(SimpleNamespace(process=process), StateChange('ablation')):
        raise RuntimeError('failed')
    assert calls[-1].kind == 'reset' and calls[-1].handle_id == 'owned'


def test_record_snapshot_replay_and_corruption(tmp_path):
    source = np.arange(6).reshape(2, 3)
    def process(array):
        array[:] = 2
        return array + 1
    subject = SimpleNamespace(process=process)
    with RunRecorder(tmp_path/'run', metadata={'model': 'fixture', 'seed': 0}) as recorder:
        with observe(subject, recorder):
            subject.process(source)
    record = RunRecord(tmp_path/'run')
    rows = list(record.events())
    np.testing.assert_array_equal(rows[0]['args'][0], np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(list(record.outputs())[0], np.full((2, 3), 3))
    array_path = next((tmp_path/'run'/'arrays').iterdir())
    array_path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        list(record.events())


def test_record_copies_asset_and_replays_metric(tmp_path):
    path = tmp_path/'stimulus.txt'
    path.write_text('original observation')
    with RunRecorder(tmp_path/'run', metadata={}) as recorder:
        recorder.record('input', path)
        recorder.record('output', np.array([1., 2.]))
    path.unlink()
    record = RunRecord(tmp_path/'run')
    assert next(record.events())['payload'].read_text() == 'original observation'
    assert record.evaluate(lambda actual, target: float(np.mean((actual-target)**2)), np.zeros(2)) == 2.5


def test_assembly_and_stimulus_roundtrip(tmp_path):
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    stimuli = StimulusSet({'stimulus_id': ['a', 'b'], 'x': [1, 2]})
    stimuli.identifier = 'fixture'
    stimuli.stimulus_paths = {}
    assembly = NeuroidAssembly(np.ones((2, 2)), dims=['presentation', 'neuroid'],
        coords={'stimulus_id': ('presentation', ['a', 'b']),
                'neuroid_id': ('neuroid', [0, 1]), 'region': ('neuroid', ['IT', 'IT'])})
    with RunRecorder(tmp_path/'run', metadata={}) as recorder:
        recorder.record('input', stimuli)
        recorder.record('output', assembly)
    rows = list(RunRecord(tmp_path/'run').events())
    assert rows[0]['payload'].equals(stimuli)
    assert rows[1]['payload'].equals(assembly)


def test_caught_model_error_marks_run_failed(tmp_path):
    def fail(value):
        raise RuntimeError('inference failed')
    subject = SimpleNamespace(process=fail)
    with RunRecorder(tmp_path/'run', metadata={}) as recorder:
        with observe(subject, recorder), pytest.raises(RuntimeError):
            subject.process('x')
    assert RunRecord(tmp_path/'run').manifest['status'] == 'failed'
    with pytest.raises(ValueError, match='completed run'):
        RunRecord(tmp_path/'run').evaluate(lambda actual, target: 0, None)


def droid_fixture():
    obs = {name: np.zeros((4, 6, 3), dtype=np.uint8) for name in (
        'wrist_image_left', 'exterior_image_1_left', 'exterior_image_2_left')}
    obs.update(joint_position=np.zeros(7), cartesian_position=np.zeros(6),
               gripper_position=np.zeros(1))
    return {'steps': [dict(observation=obs, action=np.full(7, .5),
        is_first=i == 0, is_last=i == 1, language_instruction=b'pick up cup') for i in range(2)]}


def action_spec():
    return ActionSpec(tuple(f'a{i}' for i in range(7)), ('normalized',)*7,
                      'fixture-only', 50., (-1.,)*7, (1.,)*7)


def test_droid_no_target_leakage_timing_and_resets(tmp_path):
    seen, resets = [], []
    def process(step):
        assert set(step.observation) == {'cameras', 'proprioception'}
        assert 'action' not in step.context and step.reward is None
        seen.append(step)
        return EnvironmentResponse(action=np.zeros(7))
    subject = SimpleNamespace(process=process, reset=lambda: resets.append(True))
    with RunRecorder(tmp_path/'run', metadata={'fixture': True}) as recorder:
        result = evaluate_droid_episode(subject, droid_fixture(), action_spec=action_spec(),
                                         action_source=lambda row: row['action'], recorder=recorder)
    assert len(resets) == 2 and [s.context['t_ms'] for s in seen] == [0, 50]
    assert result['evaluation'] == 'recorded_trajectory'
    assert not np.array_equal(result['predictions'], result['targets'])
    outputs = list(RunRecord(tmp_path/'run').outputs())
    assert outputs[1].t_ms == 50 and outputs[1].meta['step_num'] == 1


@pytest.mark.parametrize('action', [np.zeros(6), np.full(7, np.nan), np.full(7, 2)])
def test_action_validation(action):
    with pytest.raises(ValueError):
        action_spec().validate(action)


def test_trace_preserves_invalid_response_and_session(tmp_path):
    from brainscore.model_helpers.response_trace import build_trace_subject
    from brainscore_core.streaming import InMemorySession, StreamEvent
    model = build_trace_subject('fixture', provider=lambda req: {'text': 'unparseable', 'tokens': [1]},
                                parse=int, provenance={'provider': 'fixture'})
    session = InMemorySession([StreamEvent('generation_request', {'prompt': 'a'}, 12)])
    session.requested_output_channels = ('response_trace',)
    model.interact(session)
    result = session.emitted[0]
    assert result.payload['answer'] is None and not result.payload['valid']
    assert result.payload['raw']['tokens'] == [1] and result.t_ms == 12
    assert 'response_trace' in model.out_channels


def test_api_episode_reset():
    from brainscore.model_helpers.api_behavioral import build_api_action_fn
    from brainscore_core.events import EnvironmentStep
    prompts = []
    def provider(model, system, prompt, image, max_tokens):
        prompts.append(prompt)
        return 'Action: 0'
    act = build_api_action_fn(provider, 'fixture', obs_mode='ascii', history_window=8)
    for first in (True, False, True):
        act(EnvironmentStep(observation={'ascii': 'P G', 'legal_actions': {0: 'stay'}}, is_first=first))
    assert 'recent moves' in prompts[1].lower()
    assert 'recent moves' not in prompts[2].lower()
    assert len(act.trace) == 1


def test_generation_cache_distinguishes_inference_settings(tmp_path):
    from brainscore.model_helpers.api_behavioral import build_api_generation_fn
    calls = []
    def provider(*args):
        calls.append(args)
        return 'yes'
    for system, tokens in [('one', 16), ('two', 16), ('two', 32), ('two', 32)]:
        generate = build_api_generation_fn(provider, 'fixture', system_prompt=system,
                                            max_tokens=tokens, cache_dir=tmp_path)
        generate({'text': 'stimulus'}, 'choose', ['yes', 'no'])
    assert len(calls) == 3


@pytest.mark.parametrize('exception', [ImportError('missing dependency'), AssertionError('broken weights'), KeyError('weight')])
def test_loader_does_not_hide_construction_errors(monkeypatch, exception):
    import brainscore
    import brainscore_vision
    def fail(identifier):
        raise exception
    monkeypatch.setattr(brainscore_vision, 'load_model', fail)
    with pytest.raises(type(exception), match=str(exception).strip("'")):
        brainscore.load_model('fixture-broken')
