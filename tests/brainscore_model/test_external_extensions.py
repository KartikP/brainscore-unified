import pytest
from brainscore_core import io_catalog
from brainscore.capabilities import Capability, capability_registry, register_capability
from brainscore_core.extensions import CatalogEntry, StreamEvent, register_channel
from brainscore import BrainScoreModel
from brainscore_core.streaming import Session


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    monkeypatch.setattr(io_catalog, '_STIMULUS_COLUMNS', dict(io_catalog._STIMULUS_COLUMNS))
    previous = dict(capability_registry)
    yield
    capability_registry.clear()
    capability_registry.update(previous)


def test_external_input_output_state_reset_and_session():
    for name, kind in (('test_request', 'input'), ('test_result', 'output')):
        register_channel(CatalogEntry(name, kind, 'StreamEvent', 'int', 'test',
                                     shape_validator=lambda x: isinstance(x, int)))
    setups = []
    class Tool(Capability):
        """A session capability: it runs the session and answers each request."""
        identifier = 'external-test'
        input_channels = {'test_request'}
        output_channels = {'test_result'}
        def enabled_for(self, subject):
            return bool(subject.capability_config.get(self.identifier))
        def setup(self, subject):
            setups.append(subject.identifier)
            return {'calls': 0}
        def supports_session(self, subject, channels):
            return set(channels) == {'test_result'}
        def participate(self, subject, session):
            state = subject.capability_state(self.identifier)
            while (event := session.next_input()) is not None:
                state['calls'] += 1
                session.respond(StreamEvent('test_result', state['calls'] + event.payload,
                                            event.t_ms))
    register_capability(Tool())
    model = BrainScoreModel('external', capability_config={'external-test': True})

    def run(payload):
        session = Session([StreamEvent('test_request', payload, 5)], ('test_result',))
        model.participate(session)
        return session.collect('test_result')[0].payload

    assert run(10) == 11
    assert len(setups) == 1
    model.reset()
    assert run(10) == 11
    assert len(setups) == 2 and 'test_result' in model.out_channels
    with pytest.raises(ValueError):
        Session([StreamEvent('test_request', 'bad', 5)], ('test_result',))
    with pytest.raises(NotImplementedError):
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


def test_session_benchmark_checks_execution_support_before_compute():
    from types import SimpleNamespace
    from brainscore_core.compatibility import check_channel_compatibility, CompatibilityError
    model = BrainScoreModel('motor-only', policy=lambda step: None)
    benchmark = SimpleNamespace(identifier='session', uses_session=True,
        required_input_channels={'observation'}, requested_output_channels={'motor'})
    check_channel_compatibility(model, benchmark)
    model.check_session_support = lambda channels: (_ for _ in ()).throw(NotImplementedError('no driver'))
    with pytest.raises(CompatibilityError, match='no driver'):
        check_channel_compatibility(model, benchmark)
