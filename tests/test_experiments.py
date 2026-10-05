"""Lifecycle, compatibility and domain-neutral payload tests; no model downloads."""
from contextlib import contextmanager
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from brainscore.experiments import (Experiment, SessionProtocol, CallableProtocol,
    RecordInputsOutputs, RecordActivity, Ablate, TorchInstrumentation, Tool)
from brainscore.run_record import RunRecord
from brainscore_core.contract import Subject
from brainscore_core.streaming import InMemorySession, StreamEvent
pytestmark = pytest.mark.unit


class FixtureSubject(Subject):
    identifier = 'fixture'
    in_channels = {'stimulus'}
    out_channels = {'behavior', 'activity'}
    def __init__(self):
        self.layer = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.layer.weight.copy_(torch.eye(2))
        self.state = self.resets = 0
    def reset(self):
        self.state = 0
        self.resets += 1
    def interact(self, session):
        while (event := session.next_input()) is not None:
            value = self.layer(torch.tensor(event.payload, dtype=torch.float32))
            self.state += 1
            session.emit(StreamEvent('behavior', value.detach().numpy(), event.t_ms,
                                     {'state': self.state}))


def protocol():
    @contextmanager
    def factory(trial):
        yield InMemorySession([StreamEvent('stimulus', [1., 2.], 12)])
    return SessionProtocol('fixture', factory, trials=('baseline', 'ablation'),
                          input_channels=('stimulus',), output_channels=('behavior',))


def rows(path):
    return [e['payload'] for e in RunRecord(path / 'inputs_outputs').events()]


def test_session_intervention_order_reset_and_cleanup(tmp_path):
    subject = FixtureSubject()
    result = Experiment(subject=subject, protocol=protocol(), tools=[
        RecordInputsOutputs(), RecordActivity(['population'], when='before', name='before'),
        Ablate(['population'], trials=['ablation']), RecordActivity(['population'])],
        instrumentation=TorchInstrumentation({'population': subject.layer}),
        output_dir=tmp_path/'run').run()
    events = rows(result.directory)
    outputs = [e for e in events if e['kind'] == 'output']
    np.testing.assert_equal(outputs[0]['payload'].payload, [1, 2])
    np.testing.assert_equal(outputs[1]['payload'].payload, [0, 0])
    assert [e['payload'].meta['state'] for e in outputs] == [1, 1]
    assert [e['trial_id'] for e in outputs] == ['baseline', 'ablation']
    activity = [e for e in events if e['kind'] == 'activity' and e['trial_id'] == 'ablation']
    np.testing.assert_equal(activity[0]['payload']['value']['array'], [1, 2])
    np.testing.assert_equal(activity[1]['payload']['value']['array'], [0, 0])
    assert subject.resets == 4 and subject.state == 0
    assert len(subject.layer._forward_hooks) == 0
    assert result.manifest['status'] == 'complete'
    np.testing.assert_equal(subject.layer(torch.ones(2)).detach().numpy(), [1, 1])


@pytest.mark.parametrize('domain,payload', [
    ('vision', np.zeros((4, 4, 3), dtype=np.uint8)),
    ('language', {'text': 'hello'}),
    ('vlm', {'image': np.zeros((4, 4, 3)), 'question': 'what?'}),
    ('robotics', {'cameras': np.zeros((2, 4, 4, 3)), 'state': np.zeros(8)}),
    ('brain_behavior', {'stimulus': np.ones(2), 'time_ms': 30}),
])
def test_identical_api_for_domain_payloads(tmp_path, domain, payload):
    class Echo(Subject):
        identifier = domain
        in_channels = {'custom_input'}
        out_channels = {'custom_output'}
        def interact(self, session):
            while (event := session.next_input()) is not None:
                session.emit(StreamEvent('custom_output', event.payload, event.t_ms))
    @contextmanager
    def factory(trial):
        yield InMemorySession([StreamEvent('custom_input', payload, 5)])
    result = Experiment(subject=Echo(), protocol=SessionProtocol(domain, factory,
        input_channels=['custom_input'], output_channels=['custom_output']),
        tools=[RecordInputsOutputs()], output_dir=tmp_path/domain).run()
    output = next(e for e in rows(result.directory) if e['kind'] == 'output')
    assert output['payload'].t_ms == 5
    np.testing.assert_equal(output['payload'].payload, payload)


def test_legacy_return_and_instance_method_preserved(tmp_path):
    subject = SimpleNamespace(identifier='legacy', look_at=lambda x: np.asarray(x) * 2)
    original = subject.look_at
    result = Experiment(subject=subject, protocol=CallableProtocol('legacy-score',
        lambda model, context: float(model.look_at([2, 3]).mean()), methods=['look_at']),
        tools=[RecordInputsOutputs()], output_dir=tmp_path/'run').run()
    assert result.value == 5 and subject.look_at is original
    assert [e['kind'] for e in rows(result.directory)] == ['trial_start', 'input', 'output', 'trial_end']


def test_external_artifact_producer_and_hash(tmp_path):
    video = tmp_path/'upstream.mp4'
    video.write_bytes(b'fixture, not a video')
    def evaluate(subject, context):
        context.import_artifact(video, name='trial.mp4', producer='external-evaluator',
                                description='Fixture testing provenance')
        return subject.process(1)
    result = Experiment(subject=SimpleNamespace(process=lambda x: x),
        protocol=CallableProtocol('external', evaluate), output_dir=tmp_path/'run').run()
    assert result.manifest['artifacts'][0]['producer'] == 'external-evaluator'
    assert result.manifest['artifacts'][0]['bytes'] == len(video.read_bytes())


def test_caught_error_fails_record_and_experiment(tmp_path):
    def fail(x):
        raise RuntimeError('model failed')
    def evaluate(subject, context):
        try:
            subject.process(1)
        except RuntimeError:
            pass
    with pytest.raises(RuntimeError, match='caught model'):
        Experiment(subject=SimpleNamespace(process=fail),
            protocol=CallableProtocol('caught', evaluate), tools=[RecordInputsOutputs()],
            output_dir=tmp_path/'run').run()
    assert json.loads((tmp_path/'run/experiment.json').read_text())['status'] == 'failed'
    assert RunRecord(tmp_path/'run/inputs_outputs').manifest['status'] == 'failed'


def test_hook_cleanup_on_model_failure_and_original_hooks_retained(tmp_path):
    subject = FixtureSubject()
    handle = subject.layer.register_forward_hook(lambda *args: None)
    def fail(session):
        subject.layer(torch.ones(2))
        raise RuntimeError('trial failure')
    subject.interact = fail
    with pytest.raises(RuntimeError, match='trial failure'):
        Experiment(subject=subject, protocol=protocol(), tools=[RecordInputsOutputs(),
            Ablate(['x'])], instrumentation=TorchInstrumentation({'x': subject.layer}),
            output_dir=tmp_path/'run').run()
    assert len(subject.layer._forward_hooks) == 1
    assert subject.state == 0
    assert RunRecord(tmp_path/'run/inputs_outputs').manifest['status'] == 'failed'
    handle.remove()


@pytest.mark.parametrize('failure', ['no_provider', 'missing_target', 'order', 'channel', 'missing_method'])
def test_preflight_has_no_output_or_inference(tmp_path, failure):
    subject = FixtureSubject()
    p = protocol()
    tools = [Ablate(['unknown'])]
    provider = None
    if failure == 'missing_target':
        provider = TorchInstrumentation({'known': subject.layer})
    elif failure == 'order':
        tools = [Ablate(['known']), RecordInputsOutputs()]
        provider = TorchInstrumentation({'known': subject.layer})
    elif failure == 'channel':
        tools = []
        p.input_channels = {'unsupported'}
    elif failure == 'missing_method':
        tools = []
        p = CallableProtocol('bad', lambda s, c: None, methods=['absent'])
    with pytest.raises(ValueError):
        Experiment(subject=subject, protocol=p, tools=tools, instrumentation=provider,
                   output_dir=tmp_path/'run').run()
    assert not (tmp_path/'run').exists()
    assert subject.resets == 0


def test_external_tool_and_cleanup_failure(tmp_path):
    class ExternalTool(Tool):
        name = 'external'
        @contextmanager
        def attach(self, context):
            try:
                yield
            finally:
                raise RuntimeError('cleanup failed')
    with pytest.raises(RuntimeError, match='cleanup failed'):
        Experiment(subject=FixtureSubject(), protocol=protocol(),
            tools=[RecordInputsOutputs(), ExternalTool()], output_dir=tmp_path/'run').run()
    assert json.loads((tmp_path/'run/experiment.json').read_text())['status'] == 'failed'
    assert RunRecord(tmp_path/'run/inputs_outputs').manifest['status'] == 'failed'


def test_replay_and_comparison_detect_intervention(tmp_path):
    from brainscore.experiments import replay_sessions, compare_outputs
    subject = FixtureSubject()
    baseline = Experiment(subject=subject, protocol=protocol(), tools=[RecordInputsOutputs()],
                          output_dir=tmp_path/'baseline').run()
    candidate = Experiment(subject=subject, protocol=replay_sessions(baseline.directory),
        tools=[RecordInputsOutputs()], output_dir=tmp_path/'replay').run()
    assert compare_outputs(baseline.directory, candidate.directory)['equal']
    changed = Experiment(subject=subject, protocol=replay_sessions(baseline.directory),
        tools=[RecordInputsOutputs(), Ablate(['x'])],
        instrumentation=TorchInstrumentation({'x': subject.layer}), output_dir=tmp_path/'changed').run()
    assert compare_outputs(baseline.directory, changed.directory)['different_indices'] == [0, 1]


def test_saved_calls_explicit_allowlist_and_reset(tmp_path):
    from brainscore.experiments import replay_calls, compare_outputs
    subject = SimpleNamespace(process=lambda x: x * 2)
    baseline = Experiment(subject=subject, protocol=CallableProtocol('original',
        lambda s, c: [s.process(1), s.process(2)]), tools=[RecordInputsOutputs()],
        output_dir=tmp_path/'baseline').run()
    with pytest.raises(ValueError, match='allowed'):
        replay_calls(baseline.directory, methods=['other'], reset=lambda s: None)
    resets = []
    replay = Experiment(subject=subject, protocol=replay_calls(baseline.directory,
        methods=['process'], reset=lambda s: resets.append(True)), tools=[RecordInputsOutputs()],
        output_dir=tmp_path/'replay').run()
    assert compare_outputs(baseline.directory, replay.directory)['equal']
    assert resets == [True, True]


def test_cross_domain_demonstrations(tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1]/'examples/experiment_toolbox/run.py'
    spec = importlib.util.spec_from_file_location('toolbox_demo', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert all(v['equal'] for v in module.run(tmp_path).values())


@pytest.mark.parametrize('exception', [RuntimeError('failure'), KeyboardInterrupt()])
def test_cancellation_and_resource_cleanup(tmp_path, exception):
    subject = FixtureSubject()
    closed = []
    @contextmanager
    def factory(trial):
        try:
            yield InMemorySession([])
        finally:
            closed.append(True)
    subject.interact = lambda session: (_ for _ in ()).throw(exception)
    with pytest.raises(type(exception)):
        Experiment(subject=subject, protocol=SessionProtocol('fail', factory),
            tools=[RecordInputsOutputs(), Ablate(['x'])],
            instrumentation=TorchInstrumentation({'x': subject.layer}),
            output_dir=tmp_path/'run').run()
    assert closed == [True] and subject.resets == 2
    assert not subject.layer._forward_hooks
    assert RunRecord(tmp_path/'run/inputs_outputs').manifest['status'] == 'failed'


def test_no_overwrite_or_silent_activity_loss(tmp_path):
    subject = FixtureSubject()
    with pytest.raises(FileExistsError):
        Experiment(subject=subject, protocol=protocol(), output_dir=tmp_path).run()
    with pytest.raises(ValueError, match='persist'):
        Experiment(subject=subject, protocol=protocol(), tools=[RecordActivity(['x'])],
            instrumentation=TorchInstrumentation({'x': subject.layer}),
            output_dir=tmp_path/'run').validate()


def test_non_tensor_output_rejected_and_hooks_removed(tmp_path):
    class TupleLayer(torch.nn.Module):
        def forward(self, value):
            return (value, value)
    subject = SimpleNamespace(process=TupleLayer())
    with pytest.raises(TypeError, match='tensor'):
        Experiment(subject=subject, protocol=CallableProtocol('tuple',
            lambda s, c: s.process(torch.ones(2))),
            tools=[Ablate(['x'])], instrumentation=TorchInstrumentation({'x': subject.process}),
            output_dir=tmp_path/'run').run()
    assert not subject.process._forward_hooks


def test_corrupt_or_incomplete_records_cannot_replay(tmp_path):
    from brainscore.experiments import replay_sessions
    result = Experiment(subject=FixtureSubject(), protocol=protocol(),
        tools=[RecordInputsOutputs()], output_dir=tmp_path/'run').run()
    log = result.directory/'inputs_outputs/events.jsonl'
    log.write_text(log.read_text()+'{}\n')
    with pytest.raises(ValueError, match='checksum'):
        replay_sessions(result.directory)
    manifest = result.manifest
    manifest['status'] = 'failed'
    (result.directory/'experiment.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='completed experiment'):
        replay_sessions(result.directory)


def test_external_tool_uses_only_public_api(tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1]/'examples/experiment_toolbox/partner_tool.py'
    spec = importlib.util.spec_from_file_location('partner_tool', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = Experiment(subject=FixtureSubject(), protocol=protocol(),
        tools=[module.CountEvents()], output_dir=tmp_path/'run').run()
    assert json.loads((result.directory/'event_counts.json').read_text())['output'] == 2
    assert any(a['producer'] == 'partner_tool.CountEvents' for a in result.manifest['artifacts'])


def test_snapshot_before_mutation_and_multiple_outputs(tmp_path):
    class Mutating(FixtureSubject):
        def interact(self, session):
            event = session.next_input()
            event.payload[:] = [9, 9]
            session.emit(StreamEvent('behavior', event.payload, 12))
            event.payload[:] = [8, 8]
    result = Experiment(subject=Mutating(), protocol=protocol(),
        tools=[RecordInputsOutputs()], output_dir=tmp_path/'run').run()
    events = rows(result.directory)
    assert next(e['payload'].payload for e in events if e['kind'] == 'input') == [1, 2]
    assert next(e['payload'].payload for e in events if e['kind'] == 'output') == [9, 9]


def test_partial_provider_attachment_cleanup(tmp_path):
    subject = FixtureSubject()
    second = torch.nn.Linear(2, 2)
    def fail(*args, **kwargs):
        raise RuntimeError('attachment failed')
    second.register_forward_hook = fail
    with pytest.raises(RuntimeError, match='attachment failed'):
        Experiment(subject=subject, protocol=protocol(), tools=[RecordInputsOutputs(),
            Ablate(['first', 'second'])],
            instrumentation=TorchInstrumentation({'first': subject.layer, 'second': second}),
            output_dir=tmp_path/'run').run()
    assert not subject.layer._forward_hooks
    assert RunRecord(tmp_path/'run/inputs_outputs').manifest['status'] == 'failed'


def test_failed_run_still_lists_record_artifacts(tmp_path):
    def fail(subject, context):
        raise RuntimeError('failed evaluator')
    with pytest.raises(RuntimeError):
        Experiment(subject=SimpleNamespace(process=lambda x: x),
            protocol=CallableProtocol('failed', fail), tools=[RecordInputsOutputs()],
            output_dir=tmp_path/'run').run()
    manifest = json.loads((tmp_path/'run/experiment.json').read_text())
    assert {a['path'] for a in manifest['artifacts']} == {
        'inputs_outputs/manifest.json', 'inputs_outputs/events.jsonl'}


def test_replay_records_source_checksum(tmp_path):
    from brainscore.experiments import replay_sessions
    run = Experiment(subject=FixtureSubject(), protocol=protocol(),
        tools=[RecordInputsOutputs()], output_dir=tmp_path/'run').run()
    source = RunRecord(run.directory/'inputs_outputs').manifest
    assert replay_sessions(run.directory).metadata['source_events_sha256'] == source['events_sha256']


@pytest.mark.parametrize('domain', ['vision', 'language'])
def test_existing_adapters_and_scientific_score_passthrough(tmp_path, domain):
    from brainscore_core.metrics import Score
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    from brainscore.run_record import PayloadCodec
    assembly = NeuroidAssembly(np.array([[1., 2.], [3., 4.]]),
        dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ['a', 'b']),
            'neuroid_id': ('neuroid', [0, 1])})
    score = Score(.75)
    score.attrs['raw'] = Score(.5)
    if domain == 'vision':
        from brainscore_vision.compat.unified_adapter import VisionModelAdapter
        legacy = SimpleNamespace(identifier='legacy-vision', look_at=lambda x, n=1: assembly)
        subject, method = VisionModelAdapter(legacy), 'look_at'
    else:
        from brainscore_language.compat.unified_adapter import LanguageModelAdapter
        legacy = SimpleNamespace(identifier=lambda: 'legacy-language',
                                 digest_text=lambda text: {'neural': assembly})
        subject, method = LanguageModelAdapter(legacy), 'digest_text'
    original = getattr(subject, method)
    def benchmark(subject, context):
        response = getattr(subject, method)(['a', 'b'])
        actual = response['neural'] if domain == 'language' else response
        assert actual is assembly
        return score
    result = Experiment(subject=subject, protocol=CallableProtocol('legacy-scoring',
        benchmark, methods=[method]), tools=[RecordInputsOutputs()],
        output_dir=tmp_path/domain).run()
    assert result.value is score and getattr(subject, method) == original
    decoded = PayloadCodec(result.directory).decode(json.loads((result.directory/'result.json').read_text()))
    assert decoded.equals(score) and decoded.attrs['raw'].equals(score.attrs['raw'])
