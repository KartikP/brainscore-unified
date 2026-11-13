"""A recording during a task equals the passive recording of the same stimuli, or is refused before any pass.

Each wrapper of ``brainscore.model_helpers`` records a task through the same
functions as its passive call. Every cell of the grid below ends in one of two
outcomes: the task recording has the values, dtype and coordinates of the
passive recording, or the session is refused before any forward pass
completes. The answering part shows each stimulus to the network as the
wrapper does. Forward passes are counted after they complete.
"""
import numpy as np
import pytest
import torch

from brainscore import BrainScoreModel, LayerSelector
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming import Session
from brainscore_core.streaming_helpers import EnvironmentSource, stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from tests.brainscore_model import tiny_wrappers

pytestmark = pytest.mark.unit
pytest.importorskip('cv2')

PLAN = 'BRAINSCORE_REGIONS_PER_PASS'
TASK = TaskContext(instruction='Is it there?', response=Choice(['yes', 'no']))


def _names(assembly, dim):
    index = assembly.indexes.get(dim)
    names = [name for name in getattr(index, 'names', ()) if name and name != dim]
    return names + [name for name, coord in assembly.coords.items()
                    if coord.dims == (dim,) and name != dim and name not in names]


def assert_equal(task, passive):
    """Values, dtype, dimensions and every coordinate of the passive recording."""
    assert task.dims == passive.dims and task.dtype == passive.dtype
    np.testing.assert_array_equal(np.asarray(task.values), np.asarray(passive.values))
    for dim in passive.dims:
        for name in _names(passive, dim):
            assert list(task[name].values) == list(passive[name].values), name


def _counted(network):
    passes = []
    network.register_forward_hook(lambda *_: passes.append(1))
    return passes


def _first(stimuli, count, tag):
    rows = StimulusSet(stimuli.iloc[:count].reset_index(drop=True))
    rows.identifier = f'{stimuli.identifier}-{count}-{tag}'
    paths = getattr(stimuli, 'stimulus_paths', None)
    if paths:
        rows.stimulus_paths = {sid: paths[sid] for sid in rows['stimulus_id']}
    return rows


def _sessions(model, stimuli, regions, monkeypatch, plan, task=TASK):
    """``(task recording, passive recording)`` per region, each from its own session."""
    monkeypatch.setenv(PLAN, plan)
    outputs = [f'neural:{region}' for region in regions]
    passive = stimulus_session(stimuli, outputs)
    model.participate(passive)
    during = stimulus_session(stimuli, ['behavior:choice', *outputs], task)
    model.participate(during)
    return {o: (during.collect(o), passive.collect(o)) for o in outputs}


# ── The common grid: five wrappers, one or two presentations, passive batch 1 or 3,
#    one or two regions, both region plans, a one-pass answer or a generated one ──

@pytest.mark.parametrize('answer', ['one pass', 'generated'])
@pytest.mark.parametrize('plan', ['all', 'one'])
@pytest.mark.parametrize('regions', [('V4',), ('V1', 'IT')])
@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('count', [1, 2])
@pytest.mark.parametrize('kind', tiny_wrappers.KINDS)
def test_a_task_recording_equals_the_passive_recording(tmp_path, monkeypatch, kind, count, batch, regions,
                                                        plan, answer):
    wrapper, stimuli, selectors, network = tiny_wrappers.FACTORIES[kind](tmp_path)
    wrapper._batch_size = batch
    stimuli = _first(stimuli, count, f'{batch}{plan}')
    layer = selectors[regions[0]].name

    def reply(prompt):
        tiny_wrappers.show(wrapper, stimuli, prompt.stimulus_id, layer)
        if answer == 'generated':           # later passes: a new token, or the same input again
            if kind == 'text':
                with torch.no_grad():
                    network(input_ids=torch.tensor([[3]]))
            else:
                tiny_wrappers.show(wrapper, stimuli, prompt.stimulus_id, layer)
        return 'yes'
    model = BrainScoreModel(f'tiny-{kind}', extractor={tiny_wrappers.MODALITY[kind]: wrapper},
                            region_layer_map={r: selectors[r] for r in regions}, reply=reply)
    for task, passive in _sessions(model, stimuli, regions, monkeypatch, plan).values():
        assert_equal(task, passive)


class _Rows:
    def __init__(self, ids):
        self.ids, self.index = list(ids), 0

    def reset(self):
        self.index = 0
        return {'observation': {'stimulus_id': self.ids[0]}}

    def step(self, outputs):
        self.index += 1
        done = self.index == len(self.ids)
        return {'observation': {'stimulus_id': self.ids[min(self.index, len(self.ids) - 1)]}}, (
            'terminal' if done else None)


@pytest.mark.parametrize('kind', tiny_wrappers.KINDS)
def test_a_policy_step_records_as_the_passive_recording_or_is_refused_before_any_pass(tmp_path, kind):
    wrapper, stimuli, selectors, network = tiny_wrappers.FACTORIES[kind](tmp_path)
    passes = _counted(network)

    def policy(row):
        tiny_wrappers.show(wrapper, stimuli, row['observation']['stimulus_id'], selectors['V4'].name)
        return {'motor': 0}
    model = BrainScoreModel(f'tiny-{kind}', extractor={tiny_wrappers.MODALITY[kind]: wrapper},
                            region_layer_map={'V4': selectors['V4']}, policy=policy)
    session = Session(EnvironmentSource(_Rows(stimuli['stimulus_id'])), ('motor', 'neural:V4'))
    if kind == 'video':                     # a step presents no clip, so there is no clock
        with pytest.raises(ValueError, match='clip for its clock'):
            model.participate(session)
        assert passes == [] and not session.emitted
        return
    model.participate(session)
    passive = stimulus_session(stimuli, ['neural:V4'])
    model.participate(passive)
    recorded, expected = session.collect('neural:V4'), passive.collect('neural:V4')
    assert recorded.dtype == expected.dtype
    np.testing.assert_array_equal(recorded.values, expected.values)


# ── Text: padding on either side, and the attention mask of the pass ──

class _Ids(torch.nn.Module):
    """Token ids as values: no position term, so padding on either side keeps each token's value."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Identity()

    def forward(self, input_ids, attention_mask=None):
        return self.layer(input_ids[..., None].float())


class _Tokens:
    """One token per character (``a``=1, ``b``=2, ...), padded with 0 on ``side``."""

    def __init__(self, side):
        self.side, self.truncation_side = side, 'left'

    def __call__(self, texts, padding=True, truncation=False, max_length=None, return_tensors='pt', **_):
        ids = [[ord(c) - 96 for c in text] for text in texts]
        width = max(map(len, ids))
        pad = lambda row, fill: (row + [fill] * (width - len(row)) if self.side == 'right'
                                 else [fill] * (width - len(row)) + row)
        return {'input_ids': torch.tensor([pad(row, 0) for row in ids]),
                'attention_mask': torch.tensor([pad([1] * len(row), 0) for row in ids])}


@pytest.mark.parametrize('aggregation', ['last_token', 'mean_tokens'])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_text_of_unequal_length_padded_on_either_side(monkeypatch, side, aggregation):
    from brainscore.model_helpers.text_wrapper import TextWrapper
    network, tokens = _Ids(), _Tokens(side)
    wrapper = TextWrapper(network, tokens, identifier='ids', layer_aggregation=aggregation, batch_size=3,
                          device='cpu')
    stimuli = StimulusSet({'stimulus_id': ['x', 'y', 'z'], 'sentence': ['ab', 'abc', 'e']})
    stimuli.identifier = f'ids-{side}-{aggregation}'
    texts = dict(zip(stimuli['stimulus_id'], stimuli['sentence']))

    def reply(prompt):
        network(**tokens([texts[prompt.stimulus_id]]))
        return 'yes'
    model = BrainScoreModel('ids', extractor={'text': wrapper}, region_layer_map={'L': 'layer'}, reply=reply)
    task, passive = _sessions(model, stimuli, ['L'], monkeypatch, 'all')['neural:L']
    assert_equal(task, passive)
    expected = {'last_token': [2.0, 3.0, 5.0], 'mean_tokens': [1.5, 2.0, 5.0]}[aggregation]
    assert passive.values[:, 0].tolist() == expected


def test_the_mask_of_the_pass_decides_the_text_reduction():
    """Token values [1, 2, 0] with mask [1, 1, 0]: last_token 2.0, mean_tokens 1.5, as __call__ gives."""
    from brainscore.model_helpers.text_wrapper import TextWrapper
    for aggregation, expected in (('last_token', 2.0), ('mean_tokens', 1.5)):
        network = _Ids()
        wrapper = TextWrapper(network, _Tokens('right'), identifier='ids', layer_aggregation=aggregation,
                              device='cpu')
        assert wrapper(['ab', 'abc'], layers=['layer']).values[0, 0] == expected
        with wrapper.record(['layer']) as recording:
            network(input_ids=torch.tensor([[1, 2, 0]]), attention_mask=torch.tensor([[1, 1, 0]]))
            assert recording.take()['layer'].values[0, 0] == expected


def test_a_text_pass_of_several_prompts_is_refused_before_it_runs():
    from brainscore.model_helpers.text_wrapper import TextWrapper
    network = _Ids()
    passes = _counted(network)
    wrapper = TextWrapper(network, _Tokens('right'), identifier='ids', device='cpu')
    with wrapper.record(['layer']):
        with pytest.raises(ValueError, match='one prompt per pass'):
            network(**_Tokens('right')(['ab', 'cd']))
    assert passes == []


# ── Audio: clips of one length and one chunk, and layers of different dtypes ──

class _Sound(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.low = torch.nn.Identity()
        self.high = torch.nn.Identity()

    def forward(self, input_values, attention_mask=None):
        low = self.low(input_values.to(torch.float16)[:, :, None])
        return self.high(low.to(torch.float64))


class _Pad:
    """Pads to the longest clip of the call, with no attention mask."""
    sampling_rate = 4

    def __call__(self, waves, **_):
        width = max(map(len, waves))
        return {'input_values': torch.tensor(np.stack([np.pad(w, (0, width - len(w))) for w in waves]))}


def _sound_subject(waves, **settings):
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    network = _Sound()
    wrapper = AudioWrapper(network, _Pad(), identifier='sound', device='cpu', batch_size=2,
                           audio_loader=lambda path, rate: waves[path], **settings)

    def reply(prompt):
        network(**_Pad()([waves[prompt.stimulus_id]]))
        return 'yes'
    model = BrainScoreModel('sound', extractor={'audio': wrapper}, reply=reply,
                            region_layer_map={'low': 'low', 'high': 'high'})
    stimuli = StimulusSet({'stimulus_id': list(waves), 'audio_path': list(waves)})
    stimuli.identifier = 'sound-' + '-'.join(f'{k}{len(v)}' for k, v in waves.items())
    return model, stimuli, _counted(network)


def test_audio_clips_of_unequal_length_are_refused_before_any_pass(monkeypatch):
    """Clips [1, 1] and [3, 3, 3, 3]: __call__ pads the first to [1, 1, 0, 0] (mean 0.5); one pass
    per clip gives 1.0, so the task recording is refused."""
    waves = {'a': np.array([1, 1], np.float32), 'b': np.array([3, 3, 3, 3], np.float32)}
    model, stimuli, passes = _sound_subject(waves)
    monkeypatch.setenv(PLAN, 'all')
    passive = stimulus_session(stimuli, ['neural:low'])
    model.participate(passive)
    assert passive.collect('neural:low').values.tolist() == [[0.5], [3.0]]
    passes.clear()
    with pytest.raises(ValueError, match='different lengths'):
        model.participate(stimulus_session(stimuli, ['behavior:choice', 'neural:low'], TASK))
    assert passes == []


def test_an_audio_clip_of_several_chunks_is_refused_before_any_pass():
    waves = {'a': np.ones(8, np.float32)}
    model, stimuli, passes = _sound_subject(waves, max_duration_sec=1)
    with pytest.raises(ValueError, match='chunks'):
        model.participate(stimulus_session(stimuli, ['behavior:choice', 'neural:low'], TASK))
    assert passes == []


@pytest.mark.parametrize('plan', ['all', 'one'])
@pytest.mark.parametrize('aggregation', ['mean_time', 'time_series'])
def test_audio_layers_of_different_dtypes_keep_their_dtype_during_a_task(monkeypatch, plan, aggregation):
    waves = {'a': np.array([0.1, 0.2, 0.3, 0.4], np.float32), 'b': np.array([1, 2, 3, 4], np.float32)}
    model, stimuli, passes = _sound_subject(waves, layer_aggregation=aggregation)
    recorded = _sessions(model, stimuli, ['low', 'high'], monkeypatch, plan)
    for channel, dtype in (('neural:low', np.float16), ('neural:high', np.float64)):
        task, passive = recorded[channel]
        assert task.dtype == dtype
        assert_equal(task, passive)


# ── Video: clocks, frames and windows decided from the stimuli ──

class _Clips(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Identity()
        self.unused = torch.nn.Parameter(torch.zeros(1))         # the wrapper reads the network dtype

    def forward(self, video):                  # (B, T, C, H, W) -> (B, T, 1)
        return self.block(video.mean(dim=(2, 3, 4))[..., None])


def _clip_subject(durations, frames=None, **settings):
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    network = _Clips()
    wrapper = VideoWrapper(network, preprocessing=lambda f: torch.tensor(np.stack(f), dtype=torch.float32),
                           identifier='clips', num_frames=2, device='cpu', **settings)
    frames = frames or {path: [1.0, 3.0] for path in durations}
    wrapper._frame_sampler = lambda path, num_frames, **_: [np.full((1, 1, 1), v) for v in frames[path]]
    wrapper._clip_duration_ms = lambda path: (durations[path], 2, 2.0)
    wrapper._sample_window = lambda path, win, nf, total, fps: (
        [np.full((1, 1, 1), v) for v in frames[path]], [False] * nf)

    def reply(prompt):
        path = prompt.stimulus_id
        network(torch.tensor(np.stack([np.full((1, 1, 1), v) for v in frames[path]]),
                             dtype=torch.float32)[None])
        return 'yes'
    model = BrainScoreModel('clips', extractor={'video': wrapper}, reply=reply,
                            region_layer_map={'IT': LayerSelector('block', modality='video')})
    stimuli = StimulusSet({'stimulus_id': list(durations), 'video_path': list(durations)})
    stimuli.identifier = 'clips-' + '-'.join(f'{d:g}' for d in durations.values())
    return model, stimuli, _counted(network)


@pytest.mark.parametrize('durations', [{'a': 1000.0, 'b': 1000.0}, {'a': 1000.0, 'b': 1030.0}])
def test_video_clips_of_one_clock_record_with_the_first_clip_clock(monkeypatch, durations):
    model, stimuli, _ = _clip_subject(durations, frames={'a': [1.0, 3.0], 'b': [5.0, 7.0]})
    task, passive = _sessions(model, stimuli, ['IT'], monkeypatch, 'all')['neural:IT']
    assert_equal(task, passive)
    assert list(task['time_bin_center_ms'].values) == [250.0, 750.0]


def test_video_clips_of_different_clocks_are_refused_before_any_pass():
    """Two clips of two steps, 1000 and 2000 ms: __call__ refuses them; so does the task."""
    model, stimuli, passes = _clip_subject({'a': 1000.0, 'b': 2000.0})
    with pytest.raises(ValueError, match='different clocks'):
        model.participate(stimulus_session(stimuli, ['neural:IT']))
    passes.clear()
    with pytest.raises(ValueError, match='different clocks'):
        model.participate(stimulus_session(stimuli, ['behavior:choice', 'neural:IT'], TASK))
    assert passes == []


def test_video_clips_of_different_frame_counts_are_refused_before_any_pass():
    model, stimuli, passes = _clip_subject({'a': 1000.0, 'b': 1000.0},
                                           frames={'a': [1.0, 3.0], 'b': [1.0, 3.0, 5.0]})
    with pytest.raises(ValueError, match='numbers of frames'):
        model.participate(stimulus_session(stimuli, ['behavior:choice', 'neural:IT'], TASK))
    assert passes == []


@pytest.mark.parametrize('strategy', ['block', 'causal'])
def test_a_clip_shorter_than_its_window_is_one_window_and_records_as_the_passive_path(monkeypatch, strategy):
    model, stimuli, _ = _clip_subject({'a': 1000.0, 'b': 1000.0}, frames={'a': [1.0, 3.0], 'b': [5.0, 7.0]},
                                      context_window_ms=2000.0, context_strategy=strategy)
    task, passive = _sessions(model, stimuli, ['IT'], monkeypatch, 'all')['neural:IT']
    assert_equal(task, passive)


def test_a_clip_of_several_windows_is_refused_before_any_pass():
    model, stimuli, passes = _clip_subject({'a': 3000.0}, context_window_ms=2000.0)
    with pytest.raises(ValueError, match='windows'):
        model.participate(stimulus_session(stimuli, ['behavior:choice', 'neural:IT'], TASK))
    assert passes == []


def test_vlm_clips_of_different_time_steps_are_refused_before_any_pass(tmp_path):
    wrapper, stimuli, selectors, network = tiny_wrappers.vlm_video(tmp_path)
    import cv2
    short = str(tmp_path / 'short.avi')
    writer = cv2.VideoWriter(short, cv2.VideoWriter_fourcc(*'MJPG'), 4.0, (16, 16))
    for _ in range(4):                                     # 1 s: one time step, the others have two
        writer.write(np.zeros((16, 16, 3), np.uint8))
    writer.release()
    clips = StimulusSet({'stimulus_id': ['c0', 'short'], 'video_path': [stimuli['video_path'][0], short]})
    clips.identifier = 'vlm-clips-of-two-lengths'
    passes = _counted(network)
    model = BrainScoreModel('v', extractor={'video': wrapper}, region_layer_map={'V4': selectors['V4']},
                            reply=lambda prompt: 'yes')
    with pytest.raises(ValueError, match='numbers of time steps'):
        model.participate(stimulus_session(clips, ['behavior:choice', 'neural:V4'], TASK))
    assert passes == []


# ── VLM batches: each image of a pass, and one clip per pass ──

class _Patches(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Identity()

    def forward(self, pixel_values, grid_thw=None):
        return self.layer(pixel_values)


def test_a_vlm_pass_over_two_images_gives_each_image_as_the_passive_path():
    """Two images of two patches, batched (2, 2, 1) with means 2 and 6: one part per image."""
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    network = _Patches()
    wrapper = VLMVisionWrapper(network, None, identifier='patches', device='cpu')
    images = torch.tensor([[[1.0], [3.0]], [[5.0], [7.0]]])
    with wrapper.record(['layer']) as recording:
        network(images)
        taken = recording.take()['layer']
    assert taken.dims == ('presentation', 'part', 'neuroid')
    assert taken.values[0].tolist() == [[2.0], [6.0]]
    assert wrapper._per_image({'layer': images.numpy()}, 2, None)['layer'].tolist() == [[2.0], [6.0]]


def test_a_vlm_clip_pass_over_two_clips_is_refused_before_it_runs():
    from brainscore.model_helpers.vlm_video_wrapper import VLMVideoWrapper
    network = _Patches()
    passes = _counted(network)
    wrapper = VLMVideoWrapper(network, tiny_wrappers._ClipProcessors(), identifier='clips')
    with wrapper.record(['layer']):
        with pytest.raises(ValueError, match='one grid_thw row'):
            network(torch.tensor([[2.0]] * 4 + [[6.0]] * 4), grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]))
    assert passes == []


# ── A later pass of one presentation: the same input is recorded once, other input is refused ──

@pytest.mark.parametrize('kind', ['audio', 'video', 'vlm_vision', 'vlm_video'])
def test_an_input_changed_in_place_between_passes_is_refused_before_the_later_pass(tmp_path, kind):
    wrapper, stimuli, selectors, network = tiny_wrappers.FACTORIES[kind](tmp_path)
    shown = {}
    original = network.forward

    def keep(*args, **kwargs):
        shown.setdefault('args', (args, kwargs))
        return original(*args, **kwargs)
    network.forward = keep
    passes = _counted(network)
    with wrapper.record([selectors['V4'].name], stimuli.iloc[:1]) as recording:
        tiny_wrappers.show(wrapper, stimuli, stimuli['stimulus_id'][0], selectors['V4'].name)
        args, kwargs = shown['args']
        tensor = next(v for v in (*args, *kwargs.values()) if torch.is_tensor(v) and v.is_floating_point())
        with torch.no_grad():
            tensor.add_(1.0)                                    # the same object, other values
            with pytest.raises(ValueError, match='other input'):
                network(*args, **kwargs)
        assert len(passes) == 1
        recording.take(stimuli.iloc[0])
