"""Run the capability and benchmark templates as shipped.

The capability template is registered with isolated registries and driven
through ``Session`` + ``subject.participate``. The benchmark template
scores a subject that implements only ``participate``, and scores a
``BrainScoreModel`` exactly as the earlier ``start_recording`` + ``process``
code did. All inputs are small synthetic fixtures; no scores here are evidence
about any model. The model template needs Torch; it is exercised by
``test_api_cohesion.py::test_new_model_template_supports_public_recording_workflow``.
"""
import importlib.util
import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from brainscore_core import io_catalog
from brainscore.capabilities import capability_registry
from brainscore_core.compatibility import check_compatibility
from brainscore import BrainScoreModel
from brainscore_core.model_interface import Subject
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import neural_response
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

pytestmark = pytest.mark.unit

TEMPLATES = Path(__file__).parents[1] / 'templates'
OLD_SLOT_WORDS = re.compile(r'\b(generation_fn|action_fn|state_change_fn|activations_model|preprocessors)\b')


def _load(relative, name):
    spec = importlib.util.spec_from_file_location(name, TEMPLATES / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_templates_teach_only_public_slot_names():
    for path in sorted(TEMPLATES.rglob('*')):
        if path.suffix in ('.py', '.md'):
            hits = [line for line in path.read_text(encoding='utf-8').splitlines()
                    if OLD_SLOT_WORDS.search(line)]
            assert not hits, (path, hits)


# --- new_capability ---------------------------------------------------------------

@pytest.fixture
def capability(monkeypatch):
    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    monkeypatch.setattr(io_catalog, '_STIMULUS_COLUMNS', dict(io_catalog._STIMULUS_COLUMNS))
    previous = dict(capability_registry)
    module = _load('new_capability/capability.py', 'umi_template_capability')
    assert module.IDENTIFIER not in capability_registry      # importing registers nothing
    assert not io_catalog.has(module.REQUEST)
    module.register()
    yield module
    capability_registry.clear()
    capability_registry.update(previous)


def _requests(module, *texts):
    return [StreamEvent(channel=module.REQUEST, payload=module.WordRequest(text=text),
                        t_ms=10.0 * index, meta={'stimulus_id': f'r{index}'})
            for index, text in enumerate(texts)]


def test_capability_runs_through_participate_with_state_and_reset(capability):
    subject = capability.build_model(min_length=3)
    assert subject.check_session_support({capability.RESULT}) is capability_registry[capability.IDENTIFIER]
    for _ in range(2):   # the second round reuses the same subject after reset
        session = Session(_requests(capability, 'a red ball rolls', 'two cats'), [capability.RESULT])
        subject.participate(session)
        answers = session.collect(capability.RESULT)
        assert [event.payload for event in answers] == [
            capability.WordCount(words=3, words_so_far=3),    # 'a' is shorter than 3
            capability.WordCount(words=2, words_so_far=5)]
        assert [(event.t_ms, event.meta['stimulus_id']) for event in answers] == [(0.0, 'r0'), (10.0, 'r1')]
        subject.reset()
        assert capability.IDENTIFIER not in subject._capability_setup_done


def test_capability_is_opt_in_per_model(capability):
    plain = BrainScoreModel(identifier='plain')
    assert not {capability.REQUEST, capability.RESULT} & (plain.in_channels | plain.out_channels)
    with pytest.raises(NotImplementedError):
        plain.check_session_support({capability.RESULT})
    opted = capability.build_model()
    assert capability.REQUEST in opted.in_channels and capability.RESULT in opted.out_channels


@pytest.mark.parametrize('payload', ['text', {'text': 'a b'}, None])
def test_capability_rejects_wrong_payload_types(capability, payload):
    with pytest.raises(ValueError, match='channel catalog'):
        Session([StreamEvent(channel=capability.REQUEST, payload=payload, t_ms=0.0)], [capability.RESULT])


@pytest.mark.parametrize('text', [42, None, b'a b', ['a', 'b']])
def test_capability_rejects_invalid_request_fields_before_compute(capability, text):
    request = capability.WordRequest(text=text)
    with pytest.raises(ValueError, match='channel catalog'):
        Session([StreamEvent(channel=capability.REQUEST, payload=request, t_ms=0.0)], [capability.RESULT])
    subject = capability.build_model()
    with pytest.raises(ValueError, match='shape_validator'):
        subject.process(StreamEvent(channel=capability.REQUEST, payload=request, t_ms=0.0))
    assert capability.IDENTIFIER not in subject._capability_state   # nothing was set up or counted


@pytest.mark.parametrize('words,words_so_far', [(-1, 0), (2, 1), (True, 1), (1.0, 1), (1, None)])
def test_capability_result_validator_rejects_invalid_counts(capability, words, words_so_far):
    result = capability.WordCount(words=words, words_so_far=words_so_far)
    assert io_catalog.validate(capability.RESULT, result, io_catalog.OUTPUT)


def test_capability_model_refuses_an_invalid_result(capability):
    for valid in (capability.WordCount(words=0, words_so_far=0), capability.WordCount(words=2, words_so_far=5)):
        assert not io_catalog.validate(capability.RESULT, valid, io_catalog.OUTPUT)
    subject = capability.build_model()
    request = StreamEvent(channel=capability.REQUEST, payload=capability.WordRequest(text='a b'), t_ms=0.0)
    subject.process(request)
    subject.capability_state(capability.IDENTIFIER)['words_so_far'] = -10   # corrupt the running total
    with pytest.raises(ValueError, match='shape_validator'):
        subject.process(request)


@pytest.mark.parametrize('settings', [{'min_length': True}, {'min_length': False}, {'min_length': None},
                                      {'min_length': 0}, {'min_length': -1}, {'min_length': 1.5},
                                      {'min_length': '2'}, {}, None, False])
def test_capability_checks_settings_in_setup_before_compute(capability, settings):
    subject = BrainScoreModel(identifier='direct', capability_config={capability.IDENTIFIER: settings})
    session = Session(_requests(capability, 'one two'), [capability.RESULT])
    with pytest.raises(ValueError, match='min_length'):
        subject.participate(session)
    assert session.next_input() is not None and session.emitted == []   # nothing consumed or answered
    with pytest.raises(ValueError, match='min_length'):
        subject.process(_requests(capability, 'one two')[0])


@pytest.mark.parametrize('min_length', [True, 0, None, 2.0])
def test_capability_builder_checks_settings(capability, min_length):
    with pytest.raises(ValueError, match='min_length'):
        capability.build_model(min_length=min_length)


def test_capability_registers_once(capability):
    with pytest.raises(ValueError, match='already registered'):
        capability.register()
    with pytest.raises(ValueError, match='already registered'):
        from brainscore.capabilities import register_capability
        register_capability(capability.WordCountCapability())


def test_capability_readme_example_runs():
    text = (TEMPLATES / 'new_capability/README.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```python\n(.*?)```', text, flags=re.S)
    assert blocks
    namespace = {}
    for block in blocks:
        exec(compile(block, 'new_capability/README.md', 'exec'), namespace)
    response = namespace['model'].process({'observation': {}})
    assert response == {'motor': 0}


# --- new_benchmark -----------------------------------------------------------------

@pytest.fixture
def benchmark(monkeypatch):
    import brainscore
    # Restores the registry afterwards; the template registers 'your-benchmark' on import.
    monkeypatch.setitem(brainscore.benchmark_registry, 'your-benchmark', None)
    _load('new_benchmark/benchmark.py', 'umi_template_benchmark')
    return brainscore.benchmark_registry['your-benchmark']()


class ParticipateOnlySubject(Subject):
    """Deterministic subject with no start_recording or process."""
    identifier = 'participate-only'
    in_channels = {'vision:image'}
    out_channels = {'neural:IT'}

    def participate(self, session):
        event = session.next_input()
        while event is not None:
            stimulus_id = event.meta['stimulus_id']
            values = np.random.RandomState(zlib.crc32(str(stimulus_id).encode())).randn(1, 16)
            response = NeuroidAssembly(
                values,
                coords={'stimulus_id': ('presentation', [stimulus_id]),
                        'stimulus_index': ('presentation', [event.meta['stimulus_index']]),
                        'neuroid_id': ('neuroid', [f'IT.{unit}' for unit in range(16)]),
                        'region': ('neuroid', ['IT'] * 16)},
                dims=['presentation', 'neuroid'])
            session.respond(StreamEvent(channel='neural:IT', payload=response,
                                        t_ms=event.t_ms, meta=dict(event.meta)))
            event = session.next_input()


class StimulusExtractor:
    """Deterministic extractor returning a real assembly that carries stimulus metadata."""

    def __init__(self):
        self.calls = []

    def __call__(self, stimuli, layers):
        self.calls.append((list(stimuli['stimulus_id']), list(layers)))
        values = np.array([np.random.RandomState(zlib.crc32(sid.encode())).randn(8)
                           for sid in stimuli['stimulus_id']], dtype=np.float32)
        coords = {column: ('presentation', list(stimuli[column].values)) for column in stimuli.columns}
        coords['neuroid_id'] = ('neuroid', [f'unit{i}' for i in range(8)])
        coords['layer'] = ('neuroid', [layers[0]] * 8)
        return NeuroidAssembly(values, coords=coords, dims=['presentation', 'neuroid'])


def test_benchmark_keeps_its_identity_and_scores_a_participate_only_subject(benchmark):
    assert (benchmark.identifier, benchmark.version, float(benchmark.ceiling)) == ('your-benchmark', 1, 1.0)
    assert benchmark.parent == 'neural'
    assert len(benchmark._stimulus_set) == 32
    subject = ParticipateOnlySubject()
    assert not hasattr(subject, 'process') and not hasattr(subject, 'start_recording')
    check_compatibility(subject, benchmark)
    responses = neural_response(subject=subject, stimuli=benchmark._stimulus_set,
                                outputs=['neural:IT'], time_bins=[(70, 170)])
    assert dict(responses.sizes) == {'presentation': 32, 'neuroid': 16}
    assert list(responses['stimulus_id'].values) == list(benchmark._stimulus_set['stimulus_id'])
    score = benchmark(subject)
    assert np.isfinite(float(score))
    assert float(score.attrs['ceiling']) == 1.0 and 'raw' in score.attrs
    assert float(score) == float(benchmark(ParticipateOnlySubject()))   # deterministic


def test_benchmark_scores_a_model_as_the_earlier_template_did(benchmark):
    """Same stimuli, region, time bins, metric and ceiling as start_recording + process."""
    def subject():
        return BrainScoreModel(identifier='template-fixture', region_layer_map={'IT': 'layer2'},
                               extractor={'vision': StimulusExtractor()})
    earlier = subject()
    earlier.start_recording('IT', time_bins=[(70, 170)])
    expected = benchmark._metric(earlier.process(benchmark._stimulus_set), benchmark._assembly)

    current = subject()
    check_compatibility(current, benchmark)
    score = benchmark(current)
    assert float(score) == float(expected) / float(benchmark.ceiling)
    np.testing.assert_array_equal(score.attrs['raw'].values, expected.values)
    extractor = current._extractors['vision']
    assert extractor.calls == [(list(benchmark._stimulus_set['stimulus_id']), ['layer2'])]
    assert current._recorder.time_bins == [(70, 170)]


def test_benchmark_template_dummy_candidate_is_scorable(benchmark):
    template_tests = _load('new_benchmark/test_benchmark.py', 'umi_template_benchmark_tests')
    subject = template_tests.DummySubject()
    score = benchmark(subject)
    assert np.isfinite(float(score))
    assert subject.requested == ('neural:IT',)
