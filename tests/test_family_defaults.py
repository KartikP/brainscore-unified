"""Family plumbing lives once: the defaults and shared helpers give the values of the copies they replace.

The guard references (``tests/test_wrapper_cache_guard.py``) pin what each
wrapper extracts. The tests here pin the rest: a recording through the Qwen-VL
defaults of ``VLMVisionWrapper``, and each shared model helper against a frozen
copy of the code it replaced.
"""
import numpy as np
import pytest
import torch
import xarray
from PIL import Image

from tests.test_wrapper_cache_guard import _PatchProcessor, _image_files, _stimuli, vlm_vision_wrapper

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    monkeypatch.chdir(tmp_path)


# -- VLMVisionWrapper: the Qwen-VL layout is the default -------------------------

def test_a_qwen_pass_is_recorded_as_the_passive_call_gives_it():
    wrapper = vlm_vision_wrapper(batch_size=2)
    stimuli = _stimuli('family-vlm', image_file_name=_image_files()[1:2])      # a 4x6 image: 6 patches
    passive = wrapper(stimuli, layers=['blocks.0', 'blocks.1'])
    processed = _PatchProcessor()(images=[Image.open(stimuli['image_file_name'][0]).convert('RGB')])
    with wrapper.record(['blocks.0', 'blocks.1'], stimuli) as recording, torch.no_grad():
        wrapper._model(processed['pixel_values'], grid_thw=processed['image_grid_thw'])
        taken = recording.take(stimuli.iloc[0])
    for layer, assembly in taken.items():
        expected = passive.isel(neuroid=passive['layer'].values == layer)
        np.testing.assert_array_equal(assembly.values, expected.values)
        assert list(assembly['neuroid_id'].values) == list(expected['neuroid_id'].values)


def test_without_a_grid_each_output_row_is_one_image():
    wrapper = vlm_vision_wrapper()
    assert wrapper._patch_counts({'pixel_values': torch.zeros(2, 12)}) is None
    assert wrapper._patch_counts({'image_grid_thw': torch.tensor([[1, 2, 3], [2, 2, 2]])}) == [6, 8]
    with pytest.raises(TypeError):
        type(wrapper)(wrapper._model, None, patch_count_fn=lambda out: None)


# -- tubelet video models -------------------------------------------------------

def _old_tubelet_mean(arr, temporal, spatial):
    """Frozen copy of the V-JEPA 1, V-JEPA 2 and VideoMAE post-hooks before they were shared."""
    B, seq_len, hidden = arr.shape
    if seq_len == temporal * spatial:
        t_steps, s_steps = temporal, spatial
    elif seq_len % spatial == 0:
        t_steps, s_steps = seq_len // spatial, spatial
    elif seq_len % temporal == 0:
        t_steps, s_steps = temporal, seq_len // temporal
    else:
        raise ValueError('unable to infer')
    return arr.reshape(B, t_steps, s_steps, hidden).mean(axis=2)


def _old_fit(frames, count):
    """Frozen copy of the V-JEPA 1 and 2 frame pad and subsample before it was shared."""
    frames = list(frames)
    if len(frames) < count:
        frames = frames + [frames[-1]] * (count - len(frames))
    elif len(frames) > count:
        idxs = [int(round(i * (len(frames) - 1) / (count - 1))) for i in range(count)]
        frames = [frames[i] for i in idxs]
    return frames


@pytest.mark.parametrize('module, hook, temporal, spatial', [
    ('vjepa', '_vjepa_post_hook', 32, 256), ('vjepa_v1', '_vjepa_v1_post_hook', 8, 196),
    ('videomae_base', '_videomae_post_hook', 8, 196)])
def test_each_tubelet_registration_reduces_as_its_old_copy(module, hook, temporal, spatial):
    import importlib
    post_hook = getattr(importlib.import_module(f'brainscore.models.{module}.model'), hook)
    rng = np.random.default_rng(0)
    for seq in (temporal * spatial, 3 * spatial, temporal * 5):
        arr = rng.standard_normal((2, seq, 4)).astype(np.float32)
        result, expected = post_hook(arr), _old_tubelet_mean(arr, temporal, spatial)
        assert result.dtype == expected.dtype and np.array_equal(result, expected)


@pytest.mark.parametrize('count', [16, 64])
def test_fit_frames_gives_the_old_frames(count):
    from brainscore.models._shared import fit_frames
    frames = [np.full((2, 2, 3), i, np.uint8) for i in range(70)]
    for n in (1, 10, count, 70):
        assert [f is g for f, g in zip(fit_frames(frames[:n], count), _old_fit(frames[:n], count))] == [True] * count
    with pytest.raises(ValueError, match='empty frame list'):
        fit_frames([], count)


# -- Wav2Vec2-base audio tower ----------------------------------------------------

@pytest.fixture
def tiny_wav2vec2(monkeypatch):
    """``from_pretrained`` of the config and feature extractor give tiny local objects (no download)."""
    import transformers
    config = transformers.Wav2Vec2Config(hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
                                         intermediate_size=16, conv_dim=(8, 8), conv_stride=(5, 2),
                                         conv_kernel=(10, 3), num_conv_pos_embeddings=4,
                                         num_conv_pos_embedding_groups=2)
    monkeypatch.setattr(transformers.Wav2Vec2Config, 'from_pretrained', classmethod(lambda cls, *a, **k: config))
    monkeypatch.setattr(transformers.Wav2Vec2FeatureExtractor, 'from_pretrained',
                        classmethod(lambda cls, *a, **k: transformers.Wav2Vec2FeatureExtractor()))
    monkeypatch.setattr('brainscore.model_helpers._device.select_device', lambda device=None: torch.device('cpu'))
    return config


def test_the_random_null_has_the_weights_of_its_old_copy(tiny_wav2vec2):
    """The old ``random_wav2vec2`` builder also seeded numpy; Wav2Vec2 initialises from torch only."""
    from transformers import Wav2Vec2Model
    from brainscore.models._shared import build_audio_wrapper
    np.random.seed(123)
    _, shared = build_audio_wrapper('random-wav2vec2-base', random_init=True)
    torch.manual_seed(0)
    np.random.seed(0)
    old = Wav2Vec2Model(tiny_wav2vec2).eval()                       # frozen copy of the old builder
    assert shared.state_dict().keys() == old.state_dict().keys()
    for name, tensor in old.state_dict().items():
        assert torch.equal(shared.state_dict()[name], tensor), name


def _old_wav_loader(audio_path, target_rate):
    """Frozen copy of the wav loaders of the three audio registrations (they differed in message only)."""
    from scipy.io import wavfile
    sr, wav = wavfile.read(audio_path)
    if sr != target_rate:
        raise ValueError('rate')
    if wav.dtype.kind == 'i':
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    else:
        wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav


def test_the_shared_wav_loader_reads_as_the_old_copies(tiny_wav2vec2, tmp_path):
    from scipy.io import wavfile
    from brainscore.models._shared import build_audio_wrapper
    wrapper, _ = build_audio_wrapper('combo', random_init=True)
    assert wrapper.identifier == 'combo-audio' and wrapper._layer_aggregation == 'sequence'
    rng = np.random.default_rng(1)
    for name, data in (('int16', rng.integers(-3000, 3000, 160).astype(np.int16)),
                       ('float', rng.standard_normal(160).astype(np.float32)),
                       ('stereo', rng.integers(-3000, 3000, (160, 2)).astype(np.int16))):
        path = str(tmp_path / f'{name}.wav')
        wavfile.write(path, 16000, data)
        result, expected = wrapper._audio_loader(path, 16000), _old_wav_loader(path, 16000)
        assert result.dtype == expected.dtype and np.array_equal(result, expected)
    with pytest.raises(ValueError):
        wrapper._audio_loader(path, 8000)


# -- HuggingFace image processors (CLIP, BLIP-2) ------------------------------------

def _old_load_preprocess_images(image_filepaths, processor, image_size=224):
    """Frozen copy of the five image loaders before they were shared (``image_size`` was never read)."""
    images = []
    for path in image_filepaths:
        with Image.open(path) as img:
            if img.mode not in ('RGB',):
                img = img.convert('RGB')
            images.append(img.copy())
    processed = processor(images=images, return_tensors='pt')
    return processed['pixel_values'].numpy()


class _PixelProcessor:
    def __call__(self, images, return_tensors='pt'):
        return {'pixel_values': torch.from_numpy(np.stack([np.asarray(image, np.float32) for image in images]))}


def test_the_shared_image_loader_prepares_images_as_the_old_copies(tmp_path):
    from brainscore.models._shared import _load_preprocess_images
    rng = np.random.default_rng(2)
    paths = []
    for mode, shape in (('RGB', (4, 4, 3)), ('L', (4, 4)), ('RGBA', (4, 4, 4))):
        paths.append(str(tmp_path / f'{mode}.png'))
        Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8), mode).save(paths[-1])
    result = _load_preprocess_images(paths, processor=_PixelProcessor())
    expected = _old_load_preprocess_images(paths, processor=_PixelProcessor())
    assert result.dtype == expected.dtype and np.array_equal(result, expected)


# -- one value set for layer_aggregation --------------------------------------------

def test_every_wrapper_takes_the_one_value_set_and_refuses_others_at_construction(tmp_path):
    from brainscore.model_helpers._extractors import LAYER_AGGREGATIONS
    from tests.test_wrapper_cache_guard import audio_wrapper, text_wrapper
    builders = {'text': (text_wrapper, ('last', 'mean', 'sequence')),
                'audio': (audio_wrapper, ('mean', 'sequence')),
                'vlm_vision': (vlm_vision_wrapper, ('mean', 'first', 'sequence'))}
    for build, allowed in builders.values():
        assert set(allowed) <= set(LAYER_AGGREGATIONS)
        for value in allowed:
            assert build(layer_aggregation=value)._layer_aggregation == value
        for value in ('last_token', 'mean_tokens', 'per_token', 'mean_time', 'time_series', 'mean_patches',
                      'first_patch', 'none', 'max'):
            with pytest.raises(ValueError, match=r"layer_aggregation must be one of \(.*\); got"):
                build(layer_aggregation=value)


def test_a_recorded_pass_of_flattened_patches_without_a_grid_is_refused():
    wrapper = vlm_vision_wrapper()
    with wrapper.record(['blocks.0']), torch.no_grad():
        with pytest.raises(ValueError, match='without grid_thw'):
            wrapper._model(torch.ones(6, 12))


# -- input columns from the core catalog ----------------------------------------------

def test_each_wrapper_reads_the_columns_the_catalog_binds_to_its_channel():
    import pandas as pd
    from brainscore_core.io_catalog import stimulus_columns
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    catalog = stimulus_columns()
    readers = {'audio': AudioWrapper._extract_paths,
               'vision:video': lambda table: VideoWrapper._extract_video_paths(None, table),
               'vision:image': lambda table: VLMVisionWrapper._extract_paths(None, table)}
    for channel, read in readers.items():
        for column in [column for column, fed in catalog.items() if fed == channel]:
            assert read(pd.DataFrame({'stimulus_id': ['a'], column: ['x']})) == ['x'], (channel, column)
        others = [column for column, fed in catalog.items() if fed != channel]
        with pytest.raises(ValueError, match='expected one of'):
            read(pd.DataFrame({'stimulus_id': ['a'], **{column: ['x'] for column in others}}))


def test_stored_paths_keep_their_place_in_each_wrapper():
    """Images: ``stimulus_paths`` first (a brainio ``filename`` column holds names, not paths).
    Clips: a clip column first, then ``stimulus_paths``. Sound: a sound column only."""
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    table = _stimuli('family-paths', filename=['a.png'], video_path=['clip.mp4'])
    table.stimulus_paths = {'s0': '/stored/a.png'}
    assert VLMVisionWrapper._extract_paths(None, table) == ['/stored/a.png']
    assert VideoWrapper._extract_video_paths(None, table) == ['clip.mp4']
    del table['video_path']
    table.stimulus_paths = {'s0': '/stored/clip.mp4'}
    assert VideoWrapper._extract_video_paths(None, table) == ['/stored/clip.mp4']


@pytest.mark.parametrize('channel, columns', [
    ('audio', ['audio', 'audio_path']), ('vision:video', ['video', 'video_path']),
    ('vision:image', ['filename', 'image_file_name'])])
def test_of_two_candidate_columns_the_catalog_order_decides(channel, columns):
    """The table lists the less preferred column first; the catalog's order still decides."""
    import pandas as pd
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    read = {'audio': AudioWrapper._extract_paths,
            'vision:video': lambda table: VideoWrapper._extract_video_paths(None, table),
            'vision:image': lambda table: VLMVisionWrapper._extract_paths(None, table)}[channel]
    table = pd.DataFrame({'stimulus_id': ['a'], columns[0]: ['less preferred'], columns[1]: ['preferred']})
    assert read(table) == ['preferred']


@pytest.mark.parametrize('channel, accepted', [
    ('audio', "['audio_path', 'audio_file_name', 'audio_file', 'audio']"),
    ('vision:video', "['video_path', 'video_file_name', 'video']"),
    ('vision:image', "['image_file_name', 'image_path', 'filename', 'image', 'vision']")])
def test_a_table_without_an_accepted_column_is_refused_with_the_accepted_columns(channel, accepted):
    import re
    import pandas as pd
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    read = {'audio': AudioWrapper._extract_paths,
            'vision:video': lambda table: VideoWrapper._extract_video_paths(None, table),
            'vision:image': lambda table: VLMVisionWrapper._extract_paths(None, table)}[channel]
    with pytest.raises(ValueError, match=re.escape(f'expected one of {accepted}')):
        read(pd.DataFrame({'stimulus_id': ['a'], 'sentence': ['x']}))


def test_a_clip_table_without_stimulus_ids_is_refused_before_any_pass():
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    from tests.test_wrapper_cache_guard import _video_files, video_wrapper
    wrapper = video_wrapper()
    passes = []
    wrapper._model.register_forward_pre_hook(lambda *_: passes.append(1))
    table = StimulusSet({'video_path': _video_files()})
    table.identifier = 'no-ids'
    with pytest.raises(ValueError, match='needs a stimulus_id column'):
        wrapper(table, layers=['block'])
    assert passes == []
