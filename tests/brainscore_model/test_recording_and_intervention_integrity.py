"""A recording or an intervention gives the full result, or refuses before compute.

Each test is one case: interventions while the network is intact, overlapping
weight edits, layers of different dtypes in one call, recordings during a task
(audio chunks, video windows, the video clock), a task the subject has no part
for, the regions-per-pass plan in every session kind, and the input forms a
region declares.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from brainscore import BrainScoreModel, LayerSelector
from brainscore.perturbation import build_pytorch_ablation_fn
from brainscore_core.compatibility import CompatibilityError, check_compatibility
from brainscore_core.contract import TaskContext
from brainscore_core.events import Perturbation, Selection, StateChange
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming_helpers import apply_state_change, stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

PLAN = 'BRAINSCORE_REGIONS_PER_PASS'


# ── Interventions: a fixed set while intact, and composed weight edits ──

def _weight_subject(value=1.0, size=1):
    net = torch.nn.Linear(size, 1, bias=False)
    with torch.no_grad():
        net.weight.fill_(value)
    return net, BrainScoreModel('weights', model=net, intervention=build_pytorch_ablation_fn(net))


def _edit(subject, kind='scale', indices=None, **values):
    return apply_state_change(subject, StateChange(
        kind='weights', target=Selection('weight', indices=indices),
        perturbation=Perturbation(kind, **values))).handle_id


def _undo(subject, handle):
    apply_state_change(subject, StateChange(kind='reset', handle_id=handle))


@pytest.mark.parametrize('change', ['undo', 'reset', 'add'])
def test_a_change_of_interventions_inside_intact_is_refused_and_the_entry_state_returns(change):
    net, subject = _weight_subject()
    handle = _edit(subject, scale=2)
    with pytest.raises(RuntimeError, match='inside intact'):
        with subject._interventions.intact():
            assert net.weight.item() == 1.0
            if change == 'undo':
                _undo(subject, handle)
            elif change == 'reset':
                subject._interventions.reset()
            else:
                _edit(subject, scale=3)
    assert net.weight.item() == 2.0
    assert list(subject._interventions.active_interventions) == [handle]


def test_an_exception_inside_intact_resumes_the_entry_state():
    net, subject = _weight_subject()
    _edit(subject, scale=2)
    with pytest.raises(KeyError):
        with subject._interventions.intact():
            raise KeyError('inside')
    assert net.weight.item() == 2.0 and len(subject._interventions.active_interventions) == 1


def test_only_the_outermost_intact_block_switches_a_hook_on_again():
    net = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        net[0].weight.fill_(1.0)
    subject = BrainScoreModel('hook', model=net, intervention=build_pytorch_ablation_fn(net))
    apply_state_change(subject, StateChange(kind='ablation', target=Selection('0'),
                                            perturbation=Perturbation('zero')))
    one = torch.ones(1, 1)
    assert net(one).item() == 0.0
    with subject._interventions.intact():
        with subject._interventions.intact():
            assert net(one).item() == 1.0
        assert net(one).item() == 1.0            # the outer block still needs the intact network
    assert net(one).item() == 0.0


@pytest.mark.parametrize('first_undone', [0, 1])
def test_overlapping_weight_edits_undo_one_and_keep_the_other(first_undone):
    net, subject = _weight_subject()
    handles = [_edit(subject, scale=2), _edit(subject, scale=3)]
    assert net.weight.item() == 6.0
    _undo(subject, handles[first_undone])
    assert net.weight.item() == (3.0 if first_undone == 0 else 2.0)
    _undo(subject, handles[1 - first_undone])
    assert net.weight.item() == 1.0


def test_reset_of_overlapping_weight_edits_gives_back_the_first_value():
    net, subject = _weight_subject()
    _edit(subject, scale=2)
    _edit(subject, scale=3)
    subject.reset()
    assert net.weight.item() == 1.0 and not subject._interventions.active_interventions


def test_overlapping_slices_and_edits_that_do_not_commute_are_recomposed_in_order():
    net, subject = _weight_subject(size=4)
    with torch.no_grad():
        net.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    replace = _edit(subject, 'replace', indices=[1, 2], replacement=torch.tensor([[10.0, 20.0]]))
    zero = _edit(subject, 'zero', indices=[2, 3])
    scale = _edit(subject, 'scale', indices=[0, 1, 2], scale=-1.0)
    assert net.weight.tolist() == [[-1.0, -10.0, 0.0, 0.0]]
    _undo(subject, zero)                       # replace, then scale
    assert net.weight.tolist() == [[-1.0, -10.0, -20.0, 4.0]]
    with subject._interventions.intact():
        assert net.weight.tolist() == [[1.0, 2.0, 3.0, 4.0]]
    assert net.weight.tolist() == [[-1.0, -10.0, -20.0, 4.0]]
    _undo(subject, replace)                    # scale alone
    assert net.weight.tolist() == [[-1.0, -2.0, -3.0, 4.0]]
    _undo(subject, scale)
    assert net.weight.tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_a_single_weight_edit_and_its_undo_are_exact():
    net, subject = _weight_subject(value=0.1, size=3)
    before = net.weight.detach().clone()
    handle = _edit(subject, 'drive', indices=[1], amount=0.7)
    expected = before.clone()
    expected[:, 1] += 0.7
    assert torch.equal(net.weight.detach(), expected)
    _undo(subject, handle)
    assert torch.equal(net.weight.detach(), before)


# ── Audio: layers of different dtypes, and a recording during a task ──

class _AudioNet(torch.nn.Module):
    """A float16 layer and a float64 layer of the waveform."""

    def __init__(self):
        super().__init__()
        self.low = torch.nn.Identity()
        self.high = torch.nn.Identity()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_values):
        low = self.low(input_values.to(torch.float16)[:, :, None])
        return self.high(low.to(torch.float64))


class _Processor:
    sampling_rate = 4

    def __call__(self, waves, **kwargs):
        return {'input_values': torch.tensor(np.stack(waves))}


def _audio_wrapper(loader):
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    return AudioWrapper(_AudioNet(), _Processor(), identifier='fake-audio', batch_size=1,
                        device='cpu', max_duration_sec=1, audio_loader=loader)


def _audio_stimuli():
    stimuli = StimulusSet({'stimulus_id': ['a', 'b', 'c'], 'audio_path': ['0.1', '0.2', '0.3']})
    stimuli.identifier = 'audio-dtypes'
    return stimuli


def _neural(subject, stimuli, channels, plan, monkeypatch):
    monkeypatch.setenv(PLAN, plan)
    session = stimulus_session(stimuli, channels)
    subject.participate(session)
    return {channel: session.collect(channel) for channel in channels}


def test_a_float16_and_a_float64_layer_give_the_same_region_under_both_plans(monkeypatch):
    wrapper = _audio_wrapper(lambda path, rate: np.full(4, float(path), dtype=np.float32))
    subject = BrainScoreModel('fake', extractor={'audio': wrapper},
                              region_layer_map={'low': 'low', 'high': 'high'})
    channels = ['neural:low', 'neural:high']
    shared = _neural(subject, _audio_stimuli(), channels, 'all', monkeypatch)
    alone = _neural(subject, _audio_stimuli(), channels, 'one', monkeypatch)
    for channel, dtype in (('neural:low', np.float16), ('neural:high', np.float64)):
        a, b = shared[channel], alone[channel]
        assert a.dtype == b.dtype == dtype
        np.testing.assert_array_equal(a.values, b.values)
        assert float(a.values.mean()) == float(b.values.mean())
        assert set(a.coords) == set(b.coords)
        for name in a.coords:
            assert list(a[name].values) == list(b[name].values)
    assert float(shared['neural:low'].values.mean()) == pytest.approx(0.199951171875)


def test_a_direct_call_for_layers_of_different_dtypes_is_refused():
    from brainscore.model_helpers._extractors import LayerDtypesDiffer
    wrapper = _audio_wrapper(lambda path, rate: np.full(4, 0.5, dtype=np.float32))
    with pytest.raises(LayerDtypesDiffer, match='different dtypes'):
        wrapper._from_paths(['x'], ['low', 'high'])


def _audio_reply_subject(wrapper, chunked):
    """A reply that runs the network on the clip, as the wrapper does (chunked or whole)."""
    passes = []

    def reply(prompt):
        wave = np.array([1.0] * 4 + [3.0] * 4, dtype=np.float32)
        pieces = [wave[:4], wave[4:]] if chunked else [wave]
        for piece in pieces:
            passes.append(len(piece))
            wrapper._model(input_values=torch.tensor(piece)[None])
        return 'yes'
    subject = BrainScoreModel('fake', extractor={'audio': wrapper}, region_layer_map={'low': 'low'},
                              reply=reply)
    return subject, passes


@pytest.mark.parametrize('chunked', [True, False])
def test_an_audio_clip_the_passive_path_splits_is_refused_during_a_task(chunked):
    loader = lambda path, rate: np.array([1] * 4 + [3] * 4, dtype=np.float32)
    wrapper = _audio_wrapper(loader)
    stimuli = StimulusSet({'stimulus_id': ['a'], 'audio_path': ['clip']})
    stimuli.identifier = 'long-clip'
    passive = wrapper._from_paths(['clip'], ['low'])
    assert passive.values.tolist() == [[2.0]]       # two chunks, combined
    subject, passes = _audio_reply_subject(wrapper, chunked)
    session = stimulus_session(stimuli, ['behavior:text', 'neural:low'],
                               TaskContext(response=Text()))
    with pytest.raises(ValueError, match='chunk'):
        subject.participate(session)
    # Refused before the pass that the recording could not keep.
    assert passes == ([4, 4] if chunked else [8])


def test_an_audio_clip_of_one_chunk_is_recorded_during_a_task_as_in_the_passive_path():
    wave = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)
    wrapper = _audio_wrapper(lambda path, rate: wave)
    stimuli = StimulusSet({'stimulus_id': ['a'], 'audio_path': ['clip']})
    stimuli.identifier = 'short-clip'

    def reply(prompt):
        wrapper._model(input_values=torch.tensor(wave)[None])
        wrapper._model(input_values=torch.tensor(wave)[None])     # the same input again
        return 'yes'
    subject = BrainScoreModel('fake', extractor={'audio': wrapper}, region_layer_map={'low': 'low'},
                              reply=reply)
    session = stimulus_session(stimuli, ['behavior:text', 'neural:low'], TaskContext(response=Text()))
    subject.participate(session)
    recorded = session.collect('neural:low')
    passive = wrapper._from_paths(['clip'], ['low'])
    assert recorded.dtype == passive.dtype
    np.testing.assert_array_equal(recorded.values, passive.values)


# ── Video: windows and the clip clock during a task ──

class _VideoNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Identity()
        self.unused = torch.nn.Parameter(torch.zeros(1))

    def forward(self, video):                  # (B, T, C, H, W) -> (B, T, 1)
        return self.block(video.mean(dim=(2, 3, 4))[..., None])


def _video_wrapper(**kwargs):
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    wrapper = VideoWrapper(_VideoNet(), preprocessing=lambda frames: torch.tensor(
        np.stack(frames), dtype=torch.float32), identifier='fake-video', num_frames=2,
        device='cpu', **kwargs)
    clips = {'a.mp4': [1.0, 3.0], 'b.mp4': [5.0, 7.0]}
    wrapper._frame_sampler = lambda path, num_frames, **_: [np.full((1, 1, 1), v) for v in clips[path]]
    wrapper._clip_duration_ms = lambda path: (1000.0, 2, 2.0)
    return wrapper, clips


def _video_stimuli():
    stimuli = StimulusSet({'stimulus_id': ['a', 'b'], 'video_path': ['a.mp4', 'b.mp4']})
    stimuli.identifier = 'video-clock'
    return stimuli


def test_a_video_recording_during_a_task_has_the_values_and_clock_of_the_passive_recording(monkeypatch):
    wrapper, clips = _video_wrapper()

    def reply(prompt):
        path = reply.paths.pop(0)
        wrapper._model(wrapper._preprocessing(wrapper._frame_sampler(path, 2))[None])
        return 'left'
    reply.paths = ['a.mp4', 'b.mp4']
    subject = BrainScoreModel('v', extractor={'video': wrapper},
                              region_layer_map={'IT': LayerSelector('block', modality='video')},
                              reply=reply)
    passive = _neural(subject, _video_stimuli(), ['neural:IT'], 'all', monkeypatch)['neural:IT']
    session = stimulus_session(_video_stimuli(), ['behavior:choice', 'neural:IT'],
                               TaskContext(response=Choice(['left', 'right'])))
    subject.participate(session)
    task = session.collect('neural:IT')
    np.testing.assert_array_equal(task.values, passive.values)
    for name in ('time_bin_center_ms', 'time_bin_start_ms', 'time_bin_end_ms', 'time_bin_idx'):
        assert list(task[name].values) == list(passive[name].values), name


def test_a_video_wrapper_that_stitches_windows_is_refused_during_a_task_before_compute():
    wrapper, _ = _video_wrapper(context_window_ms=500.0)
    calls = []
    subject = BrainScoreModel('v', extractor={'video': wrapper},
                              region_layer_map={'IT': LayerSelector('block', modality='video')},
                              reply=lambda prompt: calls.append(prompt) or 'left')
    session = stimulus_session(_video_stimuli(), ['behavior:choice', 'neural:IT'],
                               TaskContext(response=Choice(['left', 'right'])))
    with pytest.raises(ValueError, match='windows'):
        subject.participate(session)
    assert calls == []


def test_a_later_video_pass_with_another_clip_is_refused_before_it_runs():
    wrapper, _ = _video_wrapper()
    seen = []
    wrapper._model.block.register_forward_hook(lambda *args: seen.append(1))

    def reply(prompt):
        for path in ('a.mp4', 'b.mp4'):
            wrapper._model(wrapper._preprocessing(wrapper._frame_sampler(path, 2))[None])
        return 'left'
    subject = BrainScoreModel('v', extractor={'video': wrapper},
                              region_layer_map={'IT': LayerSelector('block', modality='video')},
                              reply=reply)
    session = stimulus_session(_video_stimuli(), ['behavior:choice', 'neural:IT'],
                               TaskContext(response=Choice(['left', 'right'])))
    with pytest.raises(ValueError, match='other input'):
        subject.participate(session)
    assert seen == [1]


# ── A task the subject has no part for ──

def _counting_subject():
    calls = []

    def extract(stimuli, layers, **kwargs):
        calls.append(list(layers))
        ids = list(stimuli['stimulus_id'])
        return NeuroidAssembly(np.ones((len(ids), 1)), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['n']),
            'layer': ('neuroid', [layers[0]])})
    stimuli = StimulusSet({'stimulus_id': ['a', 'b'], 'image_path': ['a.png', 'b.png']})
    stimuli.identifier = 'counting'
    return BrainScoreModel('m', extractor={'vision': extract}, region_layer_map={'IT': 'x', 'V4': 'y'}), \
        stimuli, calls


@pytest.mark.parametrize('task', [TaskContext(response=Text()),
                                  TaskContext(response=Choice(['a', 'b']))])
def test_a_task_the_subject_has_no_part_for_is_not_applicable_with_zero_passes(task):
    subject, stimuli, calls = _counting_subject()
    with pytest.raises(CompatibilityError, match='cannot give a'):
        subject.participate(stimulus_session(stimuli, ['neural:IT'], task))
    assert calls == []


def test_a_passive_task_still_records_without_an_answering_part():
    subject, stimuli, calls = _counting_subject()
    session = stimulus_session(stimuli, ['neural:IT'], TaskContext(task_type='passive'))
    subject.participate(session)
    assert session.collect('neural:IT').sizes['presentation'] == 2 and calls == [['x']]


# ── BRAINSCORE_REGIONS_PER_PASS in every session kind ──

@pytest.mark.parametrize('kind', ['neural under a task', 'behavior', 'policy', 'intervention'])
def test_an_invalid_regions_per_pass_stops_every_session_kind(kind, monkeypatch):
    subject, stimuli, calls = _counting_subject()
    replies = []
    subject._reply = lambda prompt: replies.append(prompt) or 'a'
    monkeypatch.setenv(PLAN, 'invalid')
    if kind in ('policy', 'intervention'):
        subject = (BrainScoreModel('p', policy=lambda row: replies.append(row) or {'motor': 0.0},
                                   policy_in_channels={'proprioception'}, policy_out_channels={'motor'})
                   if kind == 'policy' else _weight_subject()[1])
        session = SimpleNamespace(outputs=['motor' if kind == 'policy' else 'perturbation'])
    else:
        channels = ['neural:IT'] if kind == 'neural under a task' else ['behavior:text']
        session = stimulus_session(stimuli, channels, TaskContext(response=Text()))
    with pytest.raises(ValueError, match=PLAN):
        subject.participate(session)
    assert calls == [] and replies == []


@pytest.mark.parametrize('plan, expected', [('all', [['r', 'x', 'y']]), ('one', [['r'], ['x'], ['y']])])
def test_a_readout_task_honors_regions_per_pass(plan, expected, monkeypatch):
    calls = []

    def extract(stimuli, layers, **kwargs):
        calls.append(list(layers))
        ids = list(stimuli['stimulus_id'])
        values = [[float(ord(i[0])) + k for k in range(len(layers))] for i in ids]
        return NeuroidAssembly(np.array(values), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', list(layers)),
            'layer': ('neuroid', list(layers))})
    stimuli = StimulusSet({'stimulus_id': ['a', 'b', 'c', 'd'], 'sentence': ['w'] * 4,
                           'label': ['cat', 'dog', 'cat', 'dog']})
    stimuli.identifier = 'readout-plan'
    subject = BrainScoreModel('m', extractor={'text': extract}, behavioral_readout_layer='r',
                              region_layer_map={'IT': 'x', 'V4': 'y'})
    task = TaskContext(task_type='probabilities', label_set=['cat', 'dog'], fitting_stimuli=stimuli)
    monkeypatch.setenv(PLAN, plan)
    session = stimulus_session(stimuli, ['behavior', 'neural:IT', 'neural:V4'], task=task)
    subject.participate(session)
    assert calls[1:] == expected                     # calls[0] fits the readout
    assert list(session.collect('neural:V4')['layer'].values) == ['y']


# ── The input forms a region declares ──

def _forms_extractor(value):
    def extract(stimuli, layers, **kwargs):
        ids = list(stimuli['stimulus_id'])
        return NeuroidAssembly(np.full((len(ids), 1), value), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['n']),
            'layer': ('neuroid', [layers[0]])})
    return extract


@pytest.mark.parametrize('modality', ['vision', 'video'])
def test_a_region_of_a_model_with_image_and_video_extractors_declares_both_forms(modality):
    from brainscore_core.streaming_helpers import neural_response
    model = BrainScoreModel('m', extractor={'vision': _forms_extractor(1.0), 'video': _forms_extractor(2.0)},
                            region_layer_map={'IT': LayerSelector('x', modality=modality)})
    assert model.output_inputs == {'neural:IT': frozenset({'vision:image', 'vision:video'})}
    benchmark = SimpleNamespace(identifier='clips', required_input_channels={'vision:video'},
                                requested_output_channels={'neural:IT'})
    assert check_compatibility(model, benchmark).kind == 'full'
    clips = StimulusSet({'stimulus_id': ['s'], 'video_path': ['s.mp4']})
    clips.identifier = 'clips'
    stills = StimulusSet({'stimulus_id': ['s'], 'image_path': ['s.png']})
    stills.identifier = 'stills'
    assert neural_response(model, clips, ['neural:IT']).values.max() == 2.0
    assert neural_response(model, stills, ['neural:IT']).values.max() == 1.0


class _Layers:
    """An extractor that lists its layers and refuses any other."""

    def __init__(self, names, value):
        self.names, self.value = list(names), value

    def layer_names(self):
        return self.names

    def __call__(self, stimuli, layers, **kwargs):
        unknown = [layer for layer in layers if layer not in self.names]
        if unknown:
            raise ValueError(f'unknown layers {unknown}')
        return _forms_extractor(self.value)(stimuli, layers)


def _clip_benchmark():
    return SimpleNamespace(identifier='clips', required_input_channels={'vision:video'},
                           requested_output_channels={'neural:IT'})


@pytest.mark.parametrize('layer, modality, forms', [
    ('x', 'vision', {'vision:image'}),       # only the image extractor has 'x'
    ('y', 'vision', {'vision:video'}),       # only the video extractor has 'y'
    ('x', 'video', {'vision:image'}),        # an explicit route to the video family
    ('y', 'video', {'vision:video'}),
])
def test_a_region_lists_only_the_forms_whose_extractor_has_its_layer(layer, modality, forms):
    from brainscore_core.streaming_helpers import neural_response
    model = BrainScoreModel('m', extractor={'vision': _Layers(['x'], 1.0), 'video': _Layers(['y'], 2.0)},
                            region_layer_map={'IT': LayerSelector(layer, modality=modality)})
    assert model.output_inputs == {'neural:IT': frozenset(forms)}
    clips = StimulusSet({'stimulus_id': ['s'], 'video_path': ['s.mp4']})
    clips.identifier = 'clips'
    stills = StimulusSet({'stimulus_id': ['s'], 'image_path': ['s.png']})
    stills.identifier = 'stills'
    for form, stimuli, value in (('vision:image', stills, 1.0), ('vision:video', clips, 2.0)):
        if form in forms:              # a listed form runs
            assert neural_response(model, stimuli, ['neural:IT']).values.max() == value
        else:                          # an unlisted form is refused
            with pytest.raises(ValueError, match='unknown layers'):
                neural_response(model, stimuli, ['neural:IT'])
    if 'vision:video' in forms:
        assert check_compatibility(model, _clip_benchmark()).kind == 'full'
    else:
        with pytest.raises(CompatibilityError, match='only from the inputs'):
            check_compatibility(model, _clip_benchmark())


def test_a_region_lists_both_forms_when_both_extractors_have_its_layer():
    model = BrainScoreModel('m', extractor={'vision': _Layers(['x'], 1.0), 'video': _Layers(['x'], 2.0)},
                            region_layer_map={'IT': LayerSelector('x', modality='vision')})
    assert model.output_inputs == {'neural:IT': frozenset({'vision:image', 'vision:video'})}
    assert check_compatibility(model, _clip_benchmark()).kind == 'full'


def test_a_fused_extractor_lists_each_input_it_takes():
    fused = _Layers(['x'], 1.0)
    model = BrainScoreModel('m', extractor={('vision', 'text'): fused}, region_layer_map={'IT': 'x'})
    assert model.output_inputs == {'neural:IT': frozenset({'vision:image', 'text'})}


def test_a_region_whose_layers_no_one_extractor_has_is_refused_at_registration():
    from brainscore import CompositeSelector
    with pytest.raises(ValueError, match='no one extractor has all of the layers'):
        BrainScoreModel('m', extractor={'vision': _Layers(['x'], 1.0), 'video': _Layers(['y'], 2.0)},
                        region_layer_map={'IT': CompositeSelector(((('x', None)), ('y', None)),
                                                                  modality='vision')})


# ── Which objects reset() reaches ──

class _Owner:
    """Owns several parts as bound methods and counts its resets."""

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def act(self, row):
        return {'motor': 0.0}

    def answer(self, stimulus_row, instruction, label_set):
        return label_set[0]

    def reply(self, prompt):
        return 'a'

    def likelihood(self, prompt, labels):
        return {label: 0.0 for label in labels}


def _parts(owner, names):
    parts = {name: getattr(owner, 'act' if name == 'policy' else name) for name in names}
    if 'policy' in names:
        parts.update(policy_in_channels={'observation'}, policy_out_channels={'motor'})
    return parts


@pytest.mark.parametrize('names', [('policy',), ('answer',), ('reply',), ('likelihood',),
                                   ('reply', 'likelihood'), ('policy', 'answer', 'reply', 'likelihood')])
def test_the_owner_of_bound_method_parts_is_reset_once_per_episode_and_reset(names):
    owner = _Owner()
    model = BrainScoreModel('m', in_channels={'text'}, **_parts(owner, names))
    model._begin_episode()
    assert owner.resets == 1
    model.reset()
    assert owner.resets == 2


def test_a_bound_policy_starts_each_episode_and_reset_empty():
    class Counter:
        def __init__(self):
            self.n = 0

        def reset(self):
            self.n = 0

        def act(self, row):
            self.n += 1
            return {'motor': self.n}
    counter = Counter()
    model = BrainScoreModel('m', policy=counter.act, policy_in_channels={'observation'},
                            policy_out_channels={'motor'})
    actions = []
    for _ in range(2):
        model._begin_episode()
        actions.append(model.process({'observation': 0})['motor'])
    assert actions == [1, 1]
    model.process({'observation': 0})
    model.reset()
    assert counter.n == 0


def test_a_part_object_and_the_owner_of_another_part_are_each_reset_once():
    class Scorer(_Owner):
        def __call__(self, prompt, labels):
            return {label: 0.0 for label in labels}
    owner, scorer = _Owner(), Scorer()
    model = BrainScoreModel('m', in_channels={'text'}, reply=owner.reply, answer=owner.answer,
                            likelihood=scorer)
    model.reset()
    assert (owner.resets, scorer.resets) == (1, 1)


# ── reset() inside intact() changes nothing ──

def test_a_reset_refused_inside_intact_leaves_every_state_as_it_was():
    net, _ = _weight_subject()
    owner = _Owner()
    subject = BrainScoreModel('weights', model=net, intervention=build_pytorch_ablation_fn(net),
                              extractor={'vision': _forms_extractor(1.0)}, region_layer_map={'IT': 'x'},
                              answer=owner.answer)
    handle = _edit(subject, scale=2)
    subject.start_recording('IT')
    subject.start_task(TaskContext(task_type='probabilities', instruction='i', label_set=['a', 'b']))
    subject._behavioral.history = ['earlier trial']

    def state():
        return (net.weight.item(), list(subject._interventions.active_interventions), subject._behavioral.task,
                subject._behavioral.response_path, list(subject._behavioral.history),
                subject._recorder.snapshot(), dict(subject._capability_state),
                set(subject._capability_setup_done), owner.resets)
    with subject._interventions.intact():
        before = state()
        with pytest.raises(RuntimeError, match='inside intact'):
            subject.reset()
        assert state() == before
    assert net.weight.item() == 2.0 and list(subject._interventions.active_interventions) == [handle]
    subject.reset()
    assert net.weight.item() == 1.0 and subject._behavioral.task is None and owner.resets == 1


# ── Applicability uses the part choice of the run ──

_LEGACY = dict(task_type='probabilities', instruction='Which?', label_set=['a', 'b'])
_TASKS = {
    'legacy': lambda: TaskContext(**_LEGACY),
    'choice': lambda: TaskContext(instruction='Which?', response=Choice(['a', 'b'])),
    'choice with probabilities': lambda: TaskContext(
        instruction='Which?', response=Choice(['a', 'b'], probabilities=True)),
    'choice with raw scores': lambda: TaskContext(
        instruction='Which?', response=Choice(['a', 'b'], raw_scores=True)),
    'choice with confidence': lambda: TaskContext(
        instruction='Which?', response=Choice(['a', 'b'], confidence=Scale(0, 1))),
    'scale': lambda: TaskContext(instruction='How?', response=Scale(1, 7)),
    'text': lambda: TaskContext(instruction='Say.', response=Text()),
}
_PARTS = {
    'none': {},
    'readout without fitting': {'behavioral_readout_layer': 'x'},
    'answer': {'answer': lambda row, instruction, labels: labels[0]},
    'likelihood': {'likelihood': lambda prompt, labels: {label: 0.0 for label in labels}},
    'reply': {'reply': lambda prompt: 'a'},
}
# Each task kind against each combination of parts that cannot answer it.
_INSUFFICIENT = [
    ('legacy', 'none'), ('legacy', 'readout without fitting'), ('legacy', 'likelihood'), ('legacy', 'reply'),
    ('choice', 'none'), ('choice', 'readout without fitting'),
    ('choice with probabilities', 'none'), ('choice with probabilities', 'answer'),
    ('choice with probabilities', 'reply'),
    ('choice with raw scores', 'none'), ('choice with raw scores', 'answer'),
    ('choice with raw scores', 'reply'),
    ('choice with confidence', 'none'), ('choice with confidence', 'answer'),
    ('choice with confidence', 'likelihood'),
    ('scale', 'none'), ('scale', 'answer'), ('scale', 'likelihood'),
    ('scale', 'readout without fitting'),
    ('text', 'none'), ('text', 'answer'), ('text', 'likelihood'), ('text', 'readout without fitting'),
]


# A subject with no part declares no behavior output, so that pair has no session at all.
@pytest.mark.parametrize('task_kind, parts, outputs', [
    (task_kind, parts, outputs) for task_kind, parts in _INSUFFICIENT
    for outputs in (['neural:IT'], ['behavior']) if not (parts == 'none' and outputs == ['behavior'])])
def test_a_task_no_part_can_answer_is_not_applicable_before_any_pass(task_kind, parts, outputs):
    calls = []

    def extract(stimuli, layers, **kwargs):
        calls.append(list(layers))
        return _forms_extractor(1.0)(stimuli, layers)
    stimuli = StimulusSet({'stimulus_id': ['a', 'b'], 'image_path': ['a.png', 'b.png'],
                           'label': ['a', 'b']})
    stimuli.identifier = 'applicability'
    subject = BrainScoreModel('m', extractor={'vision': extract}, region_layer_map={'IT': 'x'},
                              **_PARTS[parts])
    task = _TASKS[task_kind]()
    with pytest.raises(CompatibilityError, match='cannot'):
        subject.participate(stimulus_session(stimuli, outputs, task))
    assert calls == [] and subject._behavioral.task is None
