import pytest
from brainscore_core import io_catalog
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore.capabilities import Capability
from brainscore_core.io_catalog import CatalogEntry, register_channel
from brainscore_core.streaming import StreamEvent
from brainscore import BrainScoreModel
from brainscore_core.streaming import Session


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    monkeypatch.setattr(io_catalog, '_STIMULUS_COLUMNS', dict(io_catalog._STIMULUS_COLUMNS))
    yield


def test_external_input_output_state_reset_and_session():
    for name, direction in (('test_request', 'input'), ('test_result', 'output')):
        register_channel(CatalogEntry(name, direction, 'StreamEvent', 'int', 'test',
                                     shape_validator=lambda x: isinstance(x, int)))
    resets = []
    class Tool(Capability):
        """A session capability: it runs the session and answers each request."""
        identifier = 'external-test'
        input_channels = {'test_request'}
        output_channels = {'test_result'}
        def __init__(self):
            self.calls = 0
        def reset(self, subject):
            resets.append(subject.identifier)
            self.calls = 0
        def supports_session(self, subject, channels):
            return set(channels) == {'test_result'}
        def participate(self, subject, session):
            while (event := session.next_input()) is not None:
                self.calls += 1
                session.respond(StreamEvent('test_result', self.calls + event.payload,
                                            event.t_ms))
    model = BrainScoreModel('external', capabilities=[Tool()])

    def run(payload):
        session = Session([StreamEvent('test_request', payload, 5)], ('test_result',))
        model.participate(session)
        return session.collect('test_result')[0].payload

    assert run(10) == 11
    model.reset()
    assert resets == ['external']
    assert run(10) == 11
    assert 'test_result' in model.out_channels
    assert 'test_result' not in BrainScoreModel('other').out_channels
    with pytest.raises(ValueError):
        Session([StreamEvent('test_request', 'bad', 5)], ('test_result',))
    with pytest.raises(ExecutionNotFeasibleError, match='cannot answer'):
        model.check_session_support({'test_result', 'motor'})


def test_external_modality_routes_without_host_modification():
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    register_channel(CatalogEntry('depth_sensor', 'input', 'StimulusSet', 'depth', 'test'),
                     columns=['depth_sample'])
    calls = []
    def extract(stimuli, layers=None, **kwargs):
        calls.append(stimuli)
        return 'measured'
    model = BrainScoreModel('depth', extractor={'depth_sensor': extract})
    assert model.process(StimulusSet({'depth_sample': ['fixture']})) == 'measured'
    assert len(calls) == 1
    with pytest.raises(ValueError, match='already belongs'):
        register_channel(CatalogEntry('another_sensor', 'input', 'StimulusSet', 'depth', 'test'),
                         columns=['depth_sample'])
    with pytest.raises(ValueError, match='already belongs'):
        register_channel(CatalogEntry('another_sensor', 'input', 'StimulusSet', 'depth', 'test'),
                         columns=['image_path'])


def test_a_support_refusal_at_session_start_is_an_error_of_the_run():
    import brainscore
    from brainscore_core.compatibility import check_channel_compatibility
    from brainscore_core.streaming import Session
    steps = []
    model = BrainScoreModel('motor-only', policy=lambda step: steps.append(step) or {'motor': 0})

    class _Benchmark:
        identifier = 'session'
        required_input_channels = {'observation'}
        requested_output_channels = {'motor'}

        def __call__(self, subject):
            subject.participate(Session([], ['motor']))
            raise AssertionError('the session must not run')

    check_channel_compatibility(model, _Benchmark())     # declarations pass
    def refuse(channels):
        raise ExecutionNotFeasibleError('no driver', channels=channels)
    model.check_session_support = refuse
    with pytest.raises(ExecutionNotFeasibleError, match=r"Session refused \(task type 'none'\): no driver") as caught:
        brainscore.score(model, _Benchmark(), check_mem=False)
    assert caught.value.channels == {'motor'}
    assert steps == []
