"""The provenance entry of a score (rule 6.3): one note per session, through the public score path.

Each test asserts the complete entry. A part states kind, settings, dtype, device and batch
size; a fact it cannot give is 'not stated'. No response is changed for it.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import brainscore
from brainscore import BrainScoreModel, LayerSelector
from brainscore.benchmarks import benchmark_score, not_estimated
from brainscore.model_helpers.provenance import ProvenanceSubject
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.contract import TaskContext
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from tests import pseudo_benchmarks as pseudo
from tests.test_wrapper_cache_guard import _audio_files, _stimuli, audio_wrapper, text_wrapper

pytestmark = pytest.mark.integration

NS = 'not stated'
STORE_OFF = 'not hashed (store off)'
TEXT = 'brainscore.model_helpers.text_wrapper.TextWrapper'
AUDIO = 'brainscore.model_helpers.audio_wrapper.AudioWrapper'


class _Neural(BenchmarkBase):
    """The mean recorded response of the requested regions to a table."""

    def __init__(self, stimuli, outputs):
        self.stimulus_set, self.outputs = stimuli, sorted(outputs)
        self.requested_output_channels = frozenset(outputs)
        super().__init__(identifier='provenance-neural', version=1, parent='neural', ceiling=not_estimated())

    def __call__(self, subject):
        session = stimulus_session(self.stimulus_set, self.outputs, TaskContext(task_type='passive'))
        subject.participate(session)
        self.emitted = list(session.emitted)
        return benchmark_score(self, raw=float(np.mean([e.payload.values.mean() for e in session.emitted])),
                               metric='mean')


def _sentences():
    return _stimuli('provenance-text', sentence=['the cat sat on the mat', 'dog ran', 'the dog sat'])


def _score(subject, benchmark):
    return brainscore.score(subject, benchmark, check_mem=False)


@pytest.fixture(autouse=True)
def _store_off(tmp_path, monkeypatch):
    """Store off by default; a test that turns it on writes only to this test's folder.

    The store reads its folder when the wrapper class is imported, so the folder of
    each wrapper's store is set here, not through RESULTCACHING_HOME.
    """
    import result_caching
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    from brainscore.model_helpers.text_wrapper import TextWrapper
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    for method in (TextWrapper._from_texts_stored, AudioWrapper._from_paths_stored):
        storage = next(cell.cell_contents for cell in method.__closure__
                       if isinstance(cell.cell_contents, result_caching._XarrayStorage))
        monkeypatch.setattr(storage, '_storage_directory', str(tmp_path / 'store'))


def _text_model(**settings):
    return BrainScoreModel('provenance-text', extractor={'text': text_wrapper(**settings)},
                           region_layer_map={'IT': 'proj'})


def test_a_neural_score_from_one_extractor_states_it():
    score = _score(_text_model(batch_size=2), _Neural(_sentences(), {'neural:IT'}))
    assert score.attrs['provenance'] == ({
        'subject': 'provenance-text', 'activation_store': 'off',
        'parts': [{'serves': 'neural:IT', 'part': 'extractor', 'kind': TEXT, 'settings': STORE_OFF,
                   'parameter_devices': 'cpu', 'parameter_dtypes': 'torch.float32', 'batch_size': 2}]},)


def test_with_the_store_on_the_settings_are_the_cache_key_without_inputs(monkeypatch):
    from brainscore_core.extraction_cache import extraction_fingerprint, weight_fingerprint_scope
    monkeypatch.delenv('RESULTCACHING_DISABLE')
    model = _text_model(batch_size=2)
    score = _score(model, _Neural(_sentences(), {'neural:IT'}))
    with weight_fingerprint_scope():
        expected = extraction_fingerprint(model._extractors['text'].cache_config())
    assert expected.startswith('v2-')
    assert score.attrs['provenance'] == ({
        'subject': 'provenance-text', 'activation_store': 'on',
        'parts': [{'serves': 'neural:IT', 'part': 'extractor', 'kind': TEXT, 'settings': expected,
                   'parameter_devices': 'cpu', 'parameter_dtypes': 'torch.float32', 'batch_size': 2}]},)
    # Another batch size is another setting; the value does not move.
    other = _score(_text_model(batch_size=3), _Neural(_sentences(), {'neural:IT'}))
    assert other.attrs['provenance'][0]['parts'][0]['settings'] != expected
    assert float(other) == float(score)


def _picture(stimuli, layers):
    """A picture tower that states nothing about itself (like the legacy PytorchWrapper)."""
    ids = list(stimuli['stimulus_id'])
    return NeuroidAssembly(np.ones((len(ids), 2)), dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['p0', 'p1']),
        'layer': ('neuroid', [layers[0]] * 2)})


def test_a_picture_and_sound_score_lists_both_towers(tmp_path):
    from PIL import Image
    images = []
    for i in range(3):
        images.append(str(tmp_path / f'picture{i}.png'))
        Image.new('RGB', (4, 4), (i * 60, 0, 0)).save(images[-1])
    stimuli = _stimuli('provenance-av', image_file_name=images, audio_path=_audio_files())
    model = BrainScoreModel(
        'provenance-av', extractor={'vision': _picture, 'audio': audio_wrapper(layer_aggregation='mean',
                                                                                batch_size=2)},
        region_layer_map={'IT': LayerSelector(name='picture', modality='vision'),
                          'A1': LayerSelector(name='head', modality='audio')})
    score = _score(model, _Neural(stimuli, {'neural:IT', 'neural:A1'}))
    assert score.attrs['provenance'] == ({
        'subject': 'provenance-av', 'activation_store': 'off',
        'parts': [{'serves': 'neural:A1', 'part': 'extractor', 'kind': AUDIO, 'settings': STORE_OFF,
                   'parameter_devices': 'cpu', 'parameter_dtypes': 'torch.float32', 'batch_size': 2},
                  {'serves': 'neural:IT', 'part': 'extractor', 'kind': f'{__name__}._picture',
                   'settings': NS, 'parameter_devices': NS, 'parameter_dtypes': NS, 'batch_size': NS}]},)


def _part(name, kind):
    return {'serves': 'behavior:choice', 'part': name, 'kind': kind, 'settings': NS, 'parameter_devices': NS,
            'parameter_dtypes': NS, 'batch_size': NS}


def _answer_model():
    def answer(stimulus_row, instruction, label_set):
        return pseudo._truth(stimulus_row['stimulus_id'])
    return BrainScoreModel('pseudo-answer', in_channels={'text'}, answer=answer), answer


@pytest.mark.parametrize('part', ['reply', 'likelihood', 'answer', 'readout'])
def test_a_behavior_score_states_the_part_that_answered(part):
    if part == 'answer':
        subject, function = _answer_model()
        expected = _part('answer', f'{function.__module__}.{function.__qualname__}')
    else:
        subject = {'reply': pseudo.reply_model, 'likelihood': pseudo.likelihood_model,
                   'readout': pseudo.readout_model}[part]()
        wired = {'reply': subject._reply, 'likelihood': subject._likelihood,
                 'readout': subject._extractors.get('text')}[part]
        expected = _part(part, f'{wired.__module__}.{wired.__qualname__}')
        if part == 'readout':
            expected = {**expected, 'layer': 'late'}
    score = _score(subject, pseudo.PseudoChoice())
    assert score.attrs['provenance'] == (
        {'subject': subject.identifier, 'activation_store': 'off', 'parts': [expected]},)


def test_a_registration_adds_its_own_entry_and_keeps_the_parts():
    registration = {'dtype': 'torch.bfloat16', 'revision': 'fixture'}
    score = _score(ProvenanceSubject(_text_model(batch_size=2), registration), _Neural(_sentences(), {'neural:IT'}))
    assert score.attrs['provenance'] == (
        {'subject': 'provenance-text', 'activation_store': 'off',
         'parts': [{'serves': 'neural:IT', 'part': 'extractor', 'kind': TEXT, 'settings': STORE_OFF,
                    'parameter_devices': 'cpu', 'parameter_dtypes': 'torch.float32', 'batch_size': 2}]},
        {'subject': 'provenance-text', 'activation_store': 'off', 'registration': registration})


def test_an_old_model_through_the_adapter_states_nothing(tmp_path):
    subject = pseudo.legacy_vision_model()
    score = _score(subject, pseudo.PseudoChoice('vision:image', tmp_path))
    assert score.attrs['provenance'] == (
        {'subject': 'legacy-vision-stand-in', 'parts': NS, 'activation_store': 'off'},)


def test_a_custom_subject_states_nothing():
    subject = pseudo.CellSubject()
    score = _score(subject, pseudo.PseudoChoice())
    assert score.attrs['provenance'] == ({'subject': subject.identifier, 'parts': NS, 'activation_store': 'off'},)


def test_an_na_score_runs_nothing_and_has_no_provenance(tmp_path):
    """N/A is made before any compute: no session ran, so there is no run to describe."""
    score = _score(pseudo.readout_model(), pseudo.PseudoChoice('vision:image', tmp_path))
    assert score.attrs['status'] == 'N/A'
    assert 'provenance' not in score.attrs and 'software' not in score.attrs


def test_responses_are_not_changed():
    """The recorded payload has no provenance attribute, store off or on: replays compare equal."""
    benchmark = _Neural(_sentences(), {'neural:IT'})
    _score(_text_model(batch_size=2), benchmark)
    (event,) = benchmark.emitted
    assert 'provenance' not in event.payload.attrs and 'provenance' not in event.meta


# The weights are not hashed for provenance (no more weight hashes than without the note).

def _counting(monkeypatch):
    from brainscore_core import extraction_cache
    calls, original = [], extraction_cache._hash_tensor
    monkeypatch.setattr(extraction_cache, '_hash_tensor', lambda tensor: calls.append(1) or original(tensor))
    return calls


def _store_in(monkeypatch, folder):
    import result_caching
    from brainscore.model_helpers.text_wrapper import TextWrapper
    storage = next(cell.cell_contents for cell in TextWrapper._from_texts_stored.__closure__
                   if isinstance(cell.cell_contents, result_caching._XarrayStorage))
    monkeypatch.setattr(storage, '_storage_directory', str(folder))


def _without_the_note(monkeypatch):
    """The cache signature alone, as before the note: the baseline count."""
    from brainscore.model_helpers._extractors import CachingExtractor, cache_signature
    monkeypatch.setattr(CachingExtractor, '_signature_and_facts', lambda self, stored, inputs, identifier:
                        cache_signature(self, stored, self._key_inputs(inputs), identifier))


def _table(model):
    _score(model, _Neural(_sentences(), {'neural:IT'}))


def _stream(model):
    session = stimulus_session(_sentences(), ['neural:IT'], TaskContext(task_type='passive'), streaming=True)
    model.participate(session)


def _direct(model):
    wrapper = model._extractors['text']
    wrapper(_sentences(), layers=['proj'])
    wrapper(_sentences(), layers=['proj'])


def _experiment(model, tmp_path):
    from contextlib import contextmanager
    from brainscore.experiments import Experiment, SessionProtocol

    @contextmanager
    def factory(repetition):
        yield stimulus_session(_sentences(), ['neural:IT'], TaskContext(task_type='passive'))
    Experiment(subject=model, protocol=SessionProtocol(
        identifier='provenance-repetitions', session_factory=factory, repetitions=(0, 1, 2),
        required_input_channels=('text',), requested_output_channels=('neural:IT',)),
        tools=[], output_dir=tmp_path / f'experiment-{len(list(tmp_path.iterdir()))}').run()


RUNS = {'table': lambda model, tmp: _table(model), 'stream': lambda model, tmp: _stream(model),
        'direct': lambda model, tmp: _direct(model), 'experiment': _experiment}


@pytest.mark.parametrize('run', sorted(RUNS))
def test_with_the_store_off_no_weight_is_hashed(monkeypatch, tmp_path, run):
    calls = _counting(monkeypatch)
    RUNS[run](_text_model(batch_size=2), tmp_path)
    assert calls == []


def _model(inference):
    """The text model; with ``inference``, its weights are made under torch.inference_mode, so they
    have no version counters and core never memoizes their hashes."""
    import contextlib
    import torch
    with torch.inference_mode() if inference else contextlib.nullcontext():
        return _text_model(batch_size=2)


@pytest.mark.parametrize('inference', [False, True], ids=['ordinary', 'inference'])
@pytest.mark.parametrize('run', sorted(RUNS))
def test_with_the_store_on_the_note_adds_no_weight_hash(monkeypatch, tmp_path, run, inference):
    monkeypatch.delenv('RESULTCACHING_DISABLE')
    with monkeypatch.context() as patch:
        _without_the_note(patch)
        _store_in(patch, tmp_path / 'store-before')
        before = _counting(patch)
        RUNS[run](_model(inference), tmp_path)
    _store_in(monkeypatch, tmp_path / 'store-after')
    after = _counting(monkeypatch)
    RUNS[run](_model(inference), tmp_path)
    assert len(before) > 0 and len(after) == len(before)


@pytest.mark.parametrize('run', sorted(RUNS))
def test_with_the_store_off_no_inference_weight_is_hashed(monkeypatch, tmp_path, run):
    calls = _counting(monkeypatch)
    RUNS[run](_model(True), tmp_path)
    assert calls == []


def test_a_store_off_recording_replays_equal_with_the_store_on(monkeypatch, tmp_path):
    """Exact output comparison (payload attrs included) holds across the store state."""
    from contextlib import contextmanager
    from brainscore.experiments import (Experiment, RecordInputsOutputs, SessionProtocol, compare_outputs,
                                        replay_sessions)

    stimuli = _stimuli('provenance-replay', sentence=['the mat sat', 'the cat ran', 'dog'])   # its own store entry

    @contextmanager
    def factory(repetition):
        yield stimulus_session(stimuli, ['neural:IT'], TaskContext(task_type='passive'))
    recorded = Experiment(subject=_text_model(batch_size=2), protocol=SessionProtocol(
        'provenance-replay', factory, repetitions=(0, 1), required_input_channels=('text',),
        requested_output_channels=('neural:IT',)), tools=[RecordInputsOutputs()],
        output_dir=tmp_path / 'store-off').run()
    from brainscore.model_helpers.text_wrapper import TextWrapper
    computed, compute = [], TextWrapper._from_texts
    monkeypatch.setattr(TextWrapper, '_from_texts', lambda self, *a, **k: computed.append(1) or compute(self, *a, **k))
    monkeypatch.delenv('RESULTCACHING_DISABLE')
    counts = []
    for name in ('cold', 'warm'):
        replay = Experiment(subject=_text_model(batch_size=2), protocol=replay_sessions(recorded.directory),
                            tools=[RecordInputsOutputs()], output_dir=tmp_path / name).run()
        assert compare_outputs(recorded.directory, replay.directory)['equal']
        counts.append(len(computed))
    assert counts[0] > 0 and counts[1] == counts[0]       # the warm replay read the store


# Where a part's parameters are (no input device is recorded as the part's device).

class _Placed:
    def __init__(self, *placements):
        self._parameters = [SimpleNamespace(device=torch.device(device), dtype=dtype) for device, dtype in placements]

    def parameters(self):
        return iter(self._parameters)


@pytest.mark.parametrize('placements, devices, dtypes', [
    ([('cuda:0', torch.float16), ('cuda:1', torch.float16)], ['cuda:0', 'cuda:1'], 'torch.float16'),
    ([('cuda:0', torch.bfloat16), ('cuda:0', torch.float32)], 'cuda:0', ['torch.bfloat16', 'torch.float32']),
    ([('cuda:0', torch.float16), ('cpu', torch.float16)], NS, 'torch.float16'),        # offloaded to the CPU
    ([('cuda:0', torch.float16), ('meta', torch.float16)], NS, 'torch.float16'),       # offloaded to disk
    ([], NS, NS),
], ids=['two GPUs', 'mixed precision', 'cpu offload', 'meta offload', 'no parameters'])
def test_a_part_states_where_its_parameters_are(placements, devices, dtypes):
    from brainscore.model_helpers.provenance import _extractor_facts
    wrapper = text_wrapper(batch_size=2)
    wrapper._model, wrapper._device = _Placed(*placements), 'cuda:0'     # the input device is not the part's
    facts = _extractor_facts(wrapper, {'not hashed (store off)'})
    assert (facts['parameter_devices'], facts['parameter_dtypes']) == (devices, dtypes)
    assert 'device' not in facts and 'dtype' not in facts
