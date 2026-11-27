"""The provenance entry of a score (rule 6.3): the parts each session called, through the public score path.

Each test asserts the complete entry. Each item is a part that ran, with the requested outputs its calls
served, the layers they read, the conversion applied to its answer and its facts (settings, where its
parameters are, their dtypes, batch size); a fact a part cannot give is 'not stated'. No response is
changed for it.
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
from tests.test_model_helpers.test_wrapper_cache_guard import _audio_files, _stimuli, audio_wrapper, text_wrapper

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


def _item(part, kind, serves, layers=(), conversion=None, settings=NS, devices=NS, dtypes=NS, batch=NS,
          max_length=NS):
    """One complete part item of an entry."""
    return {'part': part, 'kind': kind, 'serves': list(serves), 'layers': sorted(layers),
            'conversion': conversion or {}, 'settings': settings, 'parameter_devices': devices,
            'parameter_dtypes': dtypes, 'batch_size': batch, 'max_length': max_length}


def _text(serves, layers=('proj',), settings=STORE_OFF, batch=2, max_length=64):
    """A text extractor item; 64 is the stand-in tokenizer's limit (the model's own window)."""
    return _item('extractor', TEXT, serves, layers, settings=settings, devices='cpu', dtypes='torch.float32',
                 batch=batch, max_length=max_length)


def _text_model(**settings):
    return BrainScoreModel('provenance-text', extractor={'text': text_wrapper(**settings)},
                           region_layer_map={'IT': 'proj'})


def test_a_neural_score_from_one_extractor_states_it():
    score = _score(_text_model(batch_size=2), _Neural(_sentences(), {'neural:IT'}))
    assert score.attrs['provenance'] == ({
        'subject': 'provenance-text', 'activation_store': 'off',
        'parts': [_text(['neural:IT'])]},)
    # A stated limit is the one on the score.
    stated = _score(_text_model(batch_size=2, max_length=5), _Neural(_sentences(), {'neural:IT'}))
    assert stated.attrs['provenance'][0]['parts'][0]['max_length'] == 5


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
        'parts': [_text(['neural:IT'], settings=expected)]},)
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
        'parts': [_item('extractor', AUDIO, ['neural:A1'], ['head'], settings=STORE_OFF, devices='cpu',
                        dtypes='torch.float32', batch=2),
                  _item('extractor', f'{__name__}._picture', ['neural:IT'], ['picture'])]},)


def _qualified(function):
    return f'{function.__module__}.{function.__qualname__}'


def _readout_kind():
    from brainscore._model.fitted_readout import ProbabilitiesClassifier
    return _qualified(ProbabilitiesClassifier)


READOUT = _readout_kind()


def _answer_model():
    def answer(stimulus_row, instruction, label_set):
        return pseudo._truth(stimulus_row['stimulus_id'])
    return BrainScoreModel('pseudo-answer', in_channels={'text'}, answer=answer), answer


@pytest.mark.parametrize('part', ['reply', 'likelihood', 'answer', 'readout'])
def test_a_behavior_score_states_the_part_that_answered(part):
    choice = ['behavior:choice']
    if part == 'answer':
        subject, function = _answer_model()
        expected = [_item('answer', _qualified(function), choice)]
    elif part == 'reply':
        subject = pseudo.reply_model()
        expected = [_item('reply', _qualified(subject._reply), choice)]
    elif part == 'likelihood':
        subject = pseudo.likelihood_model()
        expected = [_item('likelihood', _qualified(subject._likelihood), choice,
                          conversion={'behavior:choice': 'softmax_argmax'})]
    else:
        subject = pseudo.readout_model()
        extract = _qualified(subject._extractors['text'])
        expected = [_item('extractor', extract, [], ['late']),                       # the fit's features
                    _item('readout', READOUT, [], ['late']),                         # the fit
                    _item('extractor', extract, choice, ['late']),
                    _item('readout', READOUT, choice, ['late'], conversion={'behavior:choice': 'argmax'})]
    score = _score(subject, pseudo.PseudoChoice())
    assert score.attrs['provenance'] == (
        {'subject': subject.identifier, 'activation_store': 'off', 'parts': expected},)


def test_a_registration_adds_its_own_entry_and_keeps_the_parts():
    registration = {'dtype': 'torch.bfloat16', 'revision': 'fixture'}
    score = _score(ProvenanceSubject(_text_model(batch_size=2), registration), _Neural(_sentences(), {'neural:IT'}))
    assert score.attrs['provenance'] == (
        {'subject': 'provenance-text', 'activation_store': 'off',
         'parts': [_text(['neural:IT'])]},
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
    from brainscore.model_helpers.extractors import CachingExtractor, cache_signature
    monkeypatch.setattr(CachingExtractor, '_signature_and_facts', lambda self, log, stored, inputs, identifier:
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
    from brainscore.model_helpers.provenance import _facts
    wrapper = text_wrapper(batch_size=2)
    wrapper._model, wrapper._device = _Placed(*placements), 'cuda:0'     # the input device is not the part's
    facts = _facts(wrapper, {'not hashed (store off)'})
    assert (facts['parameter_devices'], facts['parameter_dtypes']) == (devices, dtypes)
    assert 'device' not in facts and 'dtype' not in facts


# A conversation: a model that states its parts and a custom subject in one score.

class _Conversation(BenchmarkBase):
    """Two seats talk for two turns; the scored subject takes the first seat."""
    requested_output_channels = frozenset({'behavior:text'})

    def __init__(self, partner):
        self.partner = partner
        self.required_input_channels = frozenset({'text'})
        super().__init__(identifier='provenance-conversation', version=1, parent='behavior',
                         ceiling=not_estimated())

    def __call__(self, subject):
        from brainscore.harnesses.multi_agent import DialogueMediator, multi_agent_rollout
        transcript = multi_agent_rollout([subject, self.partner], DialogueMediator(start='hi'), n_turns=2)
        return benchmark_score(self, raw=float(len(transcript)), metric='answers')


def _say_hello(row):
    return {'behavior:text': 'hello'}


def test_a_conversation_states_the_model_parts_and_the_silent_partner():
    from brainscore.harnesses.multi_agent import EchoAgent
    model = BrainScoreModel('provenance-talker', policy=_say_hello, policy_in_channels=('text',),
                            policy_out_channels=('behavior:text',))
    score = _score(model, _Conversation(EchoAgent('provenance-partner')))
    assert float(score) == 4.0
    assert score.attrs['provenance'] == (
        {'subject': 'provenance-partner', 'parts': NS, 'activation_store': 'off'},
        {'subject': 'provenance-talker', 'activation_store': 'off',
         'parts': [_item('policy', f'{__name__}._say_hello', ['behavior:text'])]})


def test_a_policy_that_another_callable_wraps_states_its_settings_on_the_score(tmp_path):
    """Settings a screen policy states, its own and those of the reply it asks, reach the score whatever
    callable the registration wraps it in (rule 6.3)."""
    from brainscore.harnesses.multi_agent import EchoAgent
    from brainscore.model_helpers.api_behavioral import build_api_reply
    from brainscore.model_helpers.provenance import state_settings_as
    from brainscore_core.responses import Prompt, Text

    reply = build_api_reply(lambda model, system, messages, settings: 'hello', 'snapshot-2026-01-01',
                            max_tokens=16, cache_dir=tmp_path / 'store')

    def screen(row):
        reply(Prompt(parts=(('text', 'hi'),), names=('turn',), response=Text(max_tokens=16)))
        state_settings_as(reply, {'position_grid': 4})
        return {'behavior:text': 'hello'}

    model = BrainScoreModel('provenance-wrapped', policy=lambda row: screen(row), policy_in_channels=('text',),
                            policy_out_channels=('behavior:text',))
    score = _score(model, _Conversation(EchoAgent('provenance-partner')))
    (item,) = [item for item in score.attrs['provenance'][1]['parts'] if item['part'] == 'policy']
    assert '{"position_grid": 4}' in item['settings']
    assert any('"model": "snapshot-2026-01-01"' in setting for setting in item['settings'])


# Chunk B: each case where a second resolution named a part that did not run, or named none.

class _Sessions(BenchmarkBase):
    """Runs the given sessions in order (each made by a callable); a session that fails is caught when
    ``catch`` says so. The score is the number of answers."""

    def __init__(self, makers, outputs, inputs=('text',), catch=False):
        self.makers, self.catch = makers, catch
        self.requested_output_channels = frozenset(outputs)
        self.required_input_channels = frozenset(inputs)
        super().__init__(identifier='provenance-sessions', version=1, parent='behavior',
                         ceiling=not_estimated())

    def __call__(self, subject):
        answers = 0
        for make in self.makers:
            session = make()
            try:
                subject.participate(session)
            except RuntimeError:
                if not self.catch:
                    raise
                continue
            answers += len(session.emitted)
        return benchmark_score(self, raw=float(answers), metric='answers')


def _entry(score):
    (entry,) = score.attrs['provenance']
    return entry


def test_a_plain_choice_names_the_answer_part_when_answer_and_reply_are_wired():
    from brainscore_core.responses import Choice

    def answer(stimulus_row, instruction, label_set):
        return pseudo._truth(stimulus_row['stimulus_id'])

    def reply(prompt):
        raise AssertionError('a plain choice does not call the reply')
    model = BrainScoreModel('legacy-both', in_channels={'text'}, answer=answer, reply=reply)
    task = TaskContext(response=Choice(pseudo.OPTIONS), instruction='Is this a real word?')
    score = _score(model, _Sessions([lambda: stimulus_session(pseudo.text_stimuli(), ['behavior'], task)],
                                    {'behavior'}))
    assert _entry(score) == {'subject': 'legacy-both', 'activation_store': 'off',
                             'parts': [_item('answer', _qualified(answer), ['behavior'])]}


def test_a_verbatim_choice_with_choice_and_log_probabilities_names_the_likelihood():
    from brainscore_core.responses import Choice
    def reply(prompt):
        raise AssertionError('a joint request with log probabilities is answered by the likelihood')
    likelihood = pseudo.likelihood_model()._likelihood
    model = BrainScoreModel('joint', in_channels={'text'}, reply=reply, likelihood=likelihood)
    outputs = ['behavior:choice', 'behavior:log_probabilities']
    task = TaskContext(response=Choice(pseudo.OPTIONS, verbatim=True))
    score = _score(model, _Sessions([lambda: stimulus_session(pseudo.text_stimuli(), outputs, task)], set(outputs)))
    assert _entry(score) == {'subject': 'joint', 'activation_store': 'off', 'parts': [
        _item('likelihood', _qualified(likelihood), outputs, conversion={'behavior:choice': 'softmax_argmax'})]}


def test_a_readout_that_owns_the_probabilities_is_named_for_them_beside_the_reply():
    from brainscore_core.responses import Choice
    reading = pseudo.readout_model()
    reply = pseudo.reply_model()._reply
    model = BrainScoreModel('reply-and-readout', extractor={'text': reading._extractors['text']},
                            in_channels={'text'}, reply=reply, behavioral_readout_layer='late',
                            answer_owners={'behavior:rating': 'generation'})
    outputs = ['behavior:choice', 'behavior:probabilities']
    task = TaskContext(instruction='Is this a real word?', response=Choice(pseudo.OPTIONS),
                       fitting_stimuli=pseudo.FITTING)
    score = _score(model, _Sessions([lambda: stimulus_session(pseudo.text_stimuli(), outputs, task)], set(outputs)))
    extract = _qualified(reading._extractors['text'])
    assert _entry(score) == {'subject': 'reply-and-readout', 'activation_store': 'off', 'parts': [
        _item('extractor', extract, [], ['late']), _item('readout', READOUT, [], ['late']),
        _item('reply', _qualified(reply), ['behavior:choice']),
        _item('extractor', extract, ['behavior:probabilities'], ['late']),
        _item('readout', READOUT, ['behavior:probabilities'], ['late'])]}


def test_a_composite_region_lists_its_extractor_with_every_layer_it_read():
    from brainscore import CompositeSelector
    model = BrainScoreModel('composite', extractor={'text': text_wrapper(batch_size=2)},
                            region_layer_map={'IT': CompositeSelector((('enc', (0,)), ('proj', (0, 1))))})
    score = _score(model, _Neural(_sentences(), {'neural:IT'}))
    assert _entry(score) == {'subject': 'composite', 'activation_store': 'off',
                             'parts': [_text(['neural:IT'], ['enc', 'proj'])]}


def test_a_fused_extractor_is_listed_once_for_the_region_it_served(tmp_path):
    from tests.brainscore_model.test_fused_extractor import Fused
    fused = Fused()
    model = BrainScoreModel('fused', extractor={('vision', 'text'): fused}, region_layer_map={'A': 'layers.3'})
    from PIL import Image
    paths = []
    for name in 'ab':
        paths.append(str(tmp_path / f'{name}.png'))
        Image.new('RGB', (4, 4)).save(paths[-1])
    stimuli = _stimuli('provenance-fused', image_path=paths, sentence=['one', 'two'])
    score = _score(model, _Sessions([lambda: stimulus_session(stimuli, ['neural:A'])], {'neural:A'},
                                    inputs=('text', 'vision:image')))
    assert len(fused.calls) == 1
    assert _entry(score) == {'subject': 'fused', 'activation_store': 'off', 'parts': [
        _item('extractor', 'tests.brainscore_model.test_fused_extractor.Fused', ['neural:A'], ['layers.3'])]}


def _tower(name):
    def extract(stimuli, layers):
        ids = list(stimuli['stimulus_id'])
        return NeuroidAssembly(np.ones((len(ids), 2)), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', [f'{name}0', f'{name}1']),
            'layer': ('neuroid', [layers[0]] * 2)})
    extract.__qualname__ = f'tower_{name}'
    return extract


def test_a_stream_with_no_table_lists_both_towers_that_served_one_region(tmp_path):
    """Image events reach the image extractor and video events the video one: one region, two parts."""
    from brainscore_core.streaming import Session, StreamEvent
    image, video = _tower('image'), _tower('video')
    model = BrainScoreModel('two-towers', extractor={'vision': image, 'video': video},
                            region_layer_map={'IT': LayerSelector(name='L', modality='vision')})
    picture, clip = tmp_path / 'a.png', tmp_path / 'b.mp4'
    from PIL import Image
    Image.new('RGB', (4, 4)).save(picture)
    clip.write_bytes(b'not decoded by this extractor')
    events = lambda: [StreamEvent('vision:image', str(picture), 0.0, {'stimulus_id': 'a'}),
                      StreamEvent('vision:video', str(clip), 1.0, {'stimulus_id': 'b'})]
    score = _score(model, _Sessions([lambda: Session(iter(events()), ['neural:IT'])], {'neural:IT'},
                                    inputs=('vision:image', 'vision:video')))
    assert _entry(score) == {'subject': 'two-towers', 'activation_store': 'off', 'parts': [
        _item('extractor', f'{__name__}.tower_image', ['neural:IT'], ['L']),
        _item('extractor', f'{__name__}.tower_video', ['neural:IT'], ['L'])]}


def test_a_custom_session_with_a_table_lists_what_ran():
    from brainscore_core.streaming import Session, StreamEvent
    table = _sentences()
    events = [StreamEvent('text', sentence, float(i)) for i, sentence in enumerate(table['sentence'])]
    model = _text_model(batch_size=2)
    score = _score(model, _Sessions([lambda: Session(events, ['neural:IT'], stimuli=table)], {'neural:IT'}))
    assert _entry(score) == {'subject': 'provenance-text', 'activation_store': 'off',
                             'parts': [_text(['neural:IT'])]}


def test_two_sessions_in_sequence_give_one_entry_each():
    model = BrainScoreModel('two-regions', extractor={'text': text_wrapper(batch_size=2)},
                            region_layer_map={'IT': 'proj', 'V4': 'enc'})
    makers = [lambda: stimulus_session(_sentences(), ['neural:IT']), lambda: stimulus_session(_sentences(), ['neural:V4'])]
    score = _score(model, _Sessions(makers, {'neural:IT', 'neural:V4'}))
    assert score.attrs['provenance'] == (      # sorted by content
        {'subject': 'two-regions', 'activation_store': 'off', 'parts': [_text(['neural:V4'], ['enc'])]},
        {'subject': 'two-regions', 'activation_store': 'off', 'parts': [_text(['neural:IT'], ['proj'])]})


def test_a_session_that_fails_halfway_leaves_none_of_its_parts_on_the_score():
    calls = []

    def flaky(stimuli, layers):
        calls.append(layers[0])
        if layers[0] == 'broken':
            raise RuntimeError('the extractor failed')
        return _tower('ok')(stimuli, layers)
    model = BrainScoreModel('flaky', extractor={'text': flaky},
                            region_layer_map={'IT': 'broken', 'V4': 'fine'})
    makers = [lambda: stimulus_session(_sentences(), ['neural:IT']), lambda: stimulus_session(_sentences(), ['neural:V4'])]
    score = _score(model, _Sessions(makers, {'neural:IT', 'neural:V4'}, catch=True))
    assert calls == ['broken', 'fine']           # the failed session called the extractor before it failed
    assert _entry(score) == {'subject': 'flaky', 'activation_store': 'off', 'parts': [
        _item('extractor', f'{__name__}.{flaky.__qualname__}', ['neural:V4'], ['fine'])]}


# A shared extraction pass states its settings for every output it served.

def _two_layer_entry(monkeypatch, passes):
    monkeypatch.setenv('BRAINSCORE_REGIONS_PER_PASS', passes)
    model = BrainScoreModel('shared-pass', extractor={'text': text_wrapper(batch_size=2)},
                            region_layer_map={'IT': 'proj', 'V4': 'enc'})
    return _entry(_score(model, _Neural(_sentences(), {'neural:IT', 'neural:V4'})))


@pytest.mark.parametrize('store', ['off', 'on'])
def test_a_shared_pass_states_the_settings_of_the_pass_for_each_output(monkeypatch, tmp_path, store):
    if store == 'on':
        monkeypatch.delenv('RESULTCACHING_DISABLE')
    entries = {}
    for passes in ('all', 'one'):
        if store == 'on':
            _store_in(monkeypatch, tmp_path / f'store-{passes}')
        entries[passes] = _two_layer_entry(monkeypatch, passes)
    settings = {passes: [item['settings'] for item in entry['parts']] for passes, entry in entries.items()}
    assert settings['all'] == settings['one']
    (value,) = set(settings['all'])
    assert value == STORE_OFF if store == 'off' else value.startswith('v2-')
    assert entries['all'] == {'subject': 'shared-pass', 'activation_store': store, 'parts': [
        _text(['neural:IT'], ['proj'], settings=value), _text(['neural:V4'], ['enc'], settings=value)]}


# A score run inside a benchmark keeps its own subject.

class _InnerScore(BenchmarkBase):
    """Scores another subject on the made-up choice benchmark and returns that score; the subject this
    benchmark is given never takes part."""
    requested_output_channels = frozenset({'behavior:choice'})

    def __init__(self, inner):
        self.inner = inner
        self.required_input_channels = frozenset({'text'})
        super().__init__(identifier='provenance-inner-score', version=1, parent='behavior',
                         ceiling=not_estimated())

    def __call__(self, subject):
        inner = brainscore.score(self.inner, pseudo.PseudoChoice(), check_mem=False)
        return benchmark_score(self, raw=float(inner), metric='the inner score')


def test_an_inner_score_keeps_its_silent_subject_and_names_no_part_that_never_ran():
    inner = pseudo.CellSubject()
    outer = pseudo.reply_model(calls := [])
    score = _score(outer, _InnerScore(inner))
    assert calls == []                                   # the outer model's part never ran
    assert score.attrs['provenance'] == ({'subject': inner.identifier, 'parts': NS, 'activation_store': 'off'},)


def test_the_facts_of_a_part_are_read_once_per_session(monkeypatch):
    """One TextWrapper serves IT and V4: two items, one reading of its parameters."""
    monkeypatch.setenv('BRAINSCORE_REGIONS_PER_PASS', 'one')
    wrapper = text_wrapper(batch_size=2)
    reads, parameters = [], wrapper._model.parameters
    monkeypatch.setattr(wrapper._model, 'parameters', lambda *a, **k: reads.append(1) or parameters(*a, **k))
    model = BrainScoreModel('read-once', extractor={'text': wrapper}, region_layer_map={'IT': 'proj', 'V4': 'enc'})
    from brainscore.model_helpers import provenance
    build = provenance.session_entry

    def counted(*a):
        before = len(reads)
        entry = build(*a)
        counts.append(len(reads) - before)
        return entry
    counts = []
    monkeypatch.setattr(provenance, 'session_entry', counted)
    score = _score(model, _Neural(_sentences(), {'neural:IT', 'neural:V4'}))
    assert len(_entry(score)['parts']) == 2
    assert counts == [1]


@pytest.mark.parametrize('part', ['answer', 'reply'])
def test_an_api_part_states_its_provider_snapshot_and_generation_settings(part, tmp_path):
    # A stand-in client: no network. The provider of a callable client is its name.
    from brainscore.model_helpers.api_behavioral import api_answer, build_api_reply

    def stand_in_client(model, system, messages, settings):
        return pseudo.OPTIONS[0]
    build = api_answer if part == 'answer' else build_api_reply
    function = build(stand_in_client, 'snapshot-2026-01-01', max_tokens=16, cache_dir=tmp_path / 'store')
    subject = BrainScoreModel(f'api-{part}', in_channels={'text'}, **{part: function})
    (item,) = _entry(_score(subject, pseudo.PseudoChoice()))['parts']
    assert item['part'] == part
    for fact in ('"provider": "stand_in_client"', '"model": "snapshot-2026-01-01"', '"temperature": 0',
                 '"max_tokens": 16'):
        assert fact in item['settings']


@pytest.mark.parametrize('wrapped', [False, True], ids=['as given', 'in another callable'])
def test_an_api_policy_states_its_provider_snapshot_and_generation_settings(tmp_path, wrapped):
    # The policy part of the same clients: a closed-loop score names its settings too, also when the
    # registration calls it from a callable of its own.
    from brainscore.benchmarks.grid_game.benchmark import GridGameBenchmark, policy_model
    from brainscore.model_helpers.api_behavioral import api_policy

    def stand_in_client(model, system, messages, settings):
        return 'Action: 0'
    policy = api_policy(stand_in_client, 'snapshot-2026-01-01', obs_mode='ascii', max_tokens=16,
                        cache_dir=tmp_path / 'store')
    part = (lambda row: policy(row)) if wrapped else policy
    score = _score(policy_model(part, 'api-policy'), GridGameBenchmark(size=5, n_episodes=1, max_steps=2))
    (entry,) = [entry for entry in score.attrs['provenance'] if entry['subject'] == 'api-policy']
    (item,) = entry['parts']
    assert item['part'] == 'policy'
    for fact in ('"provider": "stand_in_client"', '"model": "snapshot-2026-01-01"', '"temperature": 0',
                 '"max_tokens": 16'):
        assert fact in item['settings']
