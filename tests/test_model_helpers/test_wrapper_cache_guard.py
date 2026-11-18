"""Guard: what each caching wrapper extracts from a tiny network is pinned per ``_VERSION``.

The activation cache key of a wrapper names the ``_VERSION`` of its class, not its
source text. So a change that moves extracted values must raise ``_VERSION`` by
hand, or the cache serves old activations. This test holds, for each wrapper and
its current ``_VERSION``, a digest of the assembly it extracts from a tiny seeded
CPU network (values, dtype, dims, every coordinate). A change of values without a
raise of ``_VERSION`` fails here.

The networks use weights that are multiples of 1/8 and integer inputs, so every
sum is exact in float32 and the digest does not depend on the order of a sum
(machine, BLAS). Paths are relative to a temporary folder, so path coordinates
are the same on every machine.

To record the digests after a raise of ``_VERSION``, or of a new case, run this
file with ``BRAINSCORE_WRAPPER_GUARD_UPDATE=1``. A digest that changed under an
unchanged ``_VERSION`` is refused in that mode too.

The guards are samples: they catch a change of values only in code that their
inputs run. Code they do not run still needs a manual raise of ``_VERSION`` when
a change there can move values. Not run by these cases: text inputs longer than
``max_length`` (chunked ``per_token`` passes); audio layers with a pooled 2-D
output, and a mask at the resolution of the hooked output; video with a
``post_hook_fn``, a custom or non-uniform clock, sampling by ``target_fps``
(``num_frames=None``), and the ``black`` and ``gray`` padding; the VLM vision
``none`` reduction; other devices and reduced-precision models; the ``record``
paths of every wrapper.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

REFERENCE = Path(__file__).parents[1] / 'fixtures' / 'wrapper_guard_reference.json'
UPDATE = os.environ.get('BRAINSCORE_WRAPPER_GUARD_UPDATE') == '1'
MODULES = {
    'text': 'brainscore.model_helpers.text_wrapper',
    'audio': 'brainscore.model_helpers.audio_wrapper',
    'video': 'brainscore.model_helpers.video_wrapper',
    'vlm_vision': 'brainscore.model_helpers.vlm_vision_wrapper',
}
CLASSES = {'text': 'TextWrapper', 'audio': 'AudioWrapper', 'video': 'VideoWrapper', 'vlm_vision': 'VLMVisionWrapper'}


# Case names are the reference keys, recorded under the layer_aggregation names
# of that time; the values are the names of today (one set of names for all wrappers).
CASE_AGGREGATIONS = {
    'text': [('last_token', 'last'), ('mean_tokens', 'mean'), ('per_token', 'sequence')],
    'audio': [('mean_time', 'mean'), ('time_series', 'sequence')],
    'vlm_vision': [('mean_patches', 'mean'), ('first_patch', 'first')],
}


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    monkeypatch.chdir(tmp_path)


def _eighths(module: nn.Module, seed: int) -> nn.Module:
    """``module`` with seeded weights that are multiples of 1/8 in [-1, 1]."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randint(-8, 9, parameter.shape, generator=generator).float() / 8)
    return module.eval()


def _stimuli(identifier: str, **columns) -> StimulusSet:
    n = len(next(iter(columns.values())))
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': [f's{i}' for i in range(n)], **columns}))
    stimuli.identifier = identifier
    return stimuli


# -- text ---------------------------------------------------------------------

class _TextNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Embedding(9, 4)
        self.proj = nn.Linear(4, 3)

    def forward(self, input_ids=None, attention_mask=None, **_):
        return self.proj(self.enc(input_ids))


def _tokenizer():
    tokenizers = pytest.importorskip('tokenizers')
    from transformers import PreTrainedTokenizerFast
    vocab = {'[PAD]': 0, '[UNK]': 1, 'the': 2, 'cat': 3, 'sat': 4, 'on': 5, 'mat': 6, 'dog': 7, 'ran': 8}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]',
                                   model_max_length=64)


def text_wrapper(**settings):
    from brainscore.model_helpers.text_wrapper import TextWrapper
    return TextWrapper(_eighths(_TextNet(), 1), _tokenizer(), identifier='guard-text', device='cpu', **settings)


def _text_cases():
    stimuli = _stimuli('guard-text-stimuli', sentence=['the cat sat on the mat', 'dog ran', 'the dog sat'])
    for case, aggregation in CASE_AGGREGATIONS['text']:
        wrapper = text_wrapper(layer_aggregation=aggregation, batch_size=2)
        yield case, wrapper(stimuli, layers=['enc', 'proj'])


# -- audio --------------------------------------------------------------------

class _AudioNet(nn.Module):
    """Frames the waveform in steps of 8 samples: ``(B, L)`` -> ``(B, L // 8, 3)``."""

    def __init__(self):
        super().__init__()
        self.frames = nn.Linear(8, 4)
        self.head = nn.Linear(4, 3)

    def forward(self, input_values=None, attention_mask=None, **_):
        steps = input_values.reshape(input_values.shape[0], -1, 8)
        return self.head(self.frames(steps))


class _AudioProcessor:
    sampling_rate = 16

    def __call__(self, waveforms, sampling_rate, return_tensors='pt', padding=True):
        longest = max(len(w) for w in waveforms)
        values = np.zeros((len(waveforms), longest), dtype=np.float32)
        mask = np.zeros((len(waveforms), longest), dtype=np.int64)
        for i, waveform in enumerate(waveforms):
            values[i, :len(waveform)], mask[i, :len(waveform)] = waveform, 1
        return {'input_values': torch.from_numpy(values), 'attention_mask': torch.from_numpy(mask)}


def _load_npy(path, _rate):
    return np.load(path)


def audio_wrapper(**settings):
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    return AudioWrapper(_eighths(_AudioNet(), 2), _AudioProcessor(), identifier='guard-audio',
                        max_duration_sec=4.0, audio_loader=_load_npy, device='cpu', **settings)


def _audio_files():
    generator = np.random.default_rng(3)
    names = []
    for i, length in enumerate((32, 48, 80)):            # the last clip has two chunks of 4 s
        names.append(f'clip{i}.npy')
        np.save(names[-1], generator.integers(-3, 4, length).astype(np.float32))
    return names


def _audio_cases():
    stimuli = _stimuli('guard-audio-stimuli', audio_path=_audio_files())
    for case, aggregation in CASE_AGGREGATIONS['audio']:
        wrapper = audio_wrapper(layer_aggregation=aggregation, batch_size=2)
        yield case, wrapper(stimuli, layers=['frames', 'head'])


# -- video --------------------------------------------------------------------

class _VideoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(12, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, video):
        batch, steps = video.shape[:2]
        return self.head(self.block(video.reshape(batch, steps, -1)))


def _frames_to_tensor(frames):
    """Each frame becomes ``(3, 2, 2)`` filled with 0 or 1 (dark or bright): robust to the codec."""
    bright = [float(round(float(np.asarray(frame).mean()) / 255.0)) for frame in frames]
    scale = torch.arange(1, 4, dtype=torch.float32).reshape(3, 1, 1).expand(3, 2, 2)
    return torch.stack([scale * value for value in bright])


def _video_files():
    cv2 = pytest.importorskip('cv2')
    names = []
    for i, (count, period) in enumerate(((20, 3), (20, 4))):
        names.append(f'clip{i}.mp4')
        writer = cv2.VideoWriter(names[-1], cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (16, 16))
        if not writer.isOpened():
            pytest.skip('cv2 VideoWriter could not open the mp4v codec')
        for frame in range(count):
            writer.write(np.full((16, 16, 3), 255 if frame % period == 0 else 0, dtype=np.uint8))
        writer.release()
    return names


def video_wrapper(**settings):
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    return VideoWrapper(_eighths(_VideoNet(), 4), _frames_to_tensor, identifier='guard-video',
                        num_frames=4, device='cpu', **settings)


def _video_cases():
    stimuli = _stimuli('guard-video-stimuli', video_path=_video_files())
    yield 'whole', video_wrapper(batch_size=2)(stimuli, layers=['block', 'head'])
    yield 'block', video_wrapper(context_window_ms=1500)(stimuli, layers=['block', 'head'])
    yield 'causal', video_wrapper(context_window_ms=1500, context_strategy='causal')(
        stimuli, layers=['block', 'head'])


# -- VLM vision (flattened patches, Qwen layout) --------------------------------

class _PatchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(12, 4), nn.Linear(4, 3)])

    def forward(self, pixel_values, grid_thw=None):
        for block in self.blocks:
            pixel_values = block(pixel_values)
        return pixel_values


class _PatchProcessor:
    """Cuts each image into 2x2 patches of pixel quarters (0..3), flattened across images."""

    def __call__(self, images=None, return_tensors='pt'):
        patches, grids = [], []
        for image in images:
            pixels = np.asarray(image, dtype=np.int64) // 64             # (H, W, 3)
            h, w = pixels.shape[0] // 2, pixels.shape[1] // 2
            blocks = pixels[:2 * h, :2 * w].reshape(h, 2, w, 2, 3).transpose(0, 2, 1, 3, 4)
            patches.append(blocks.reshape(h * w, 12))
            grids.append((1, h, w))
        return {'pixel_values': torch.from_numpy(np.concatenate(patches).astype(np.float32)),
                'image_grid_thw': torch.tensor(grids)}


def vlm_vision_wrapper(**settings):
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    return VLMVisionWrapper(_eighths(_PatchNet(), 5), _PatchProcessor(), identifier='guard-vlm',
                            device='cpu', **settings)


class _PerImageProcessor(_PatchProcessor):
    """The same patches, one row per image: ``(images, patches, 12)``, as a model that keeps images apart."""

    def __call__(self, images=None, return_tensors='pt'):
        flat = super().__call__(images=images)
        return {'pixel_values': flat['pixel_values'].reshape(len(images), -1, 12)}


def _image_files(sizes=((4, 4), (4, 6), (2, 2)), seed=6):
    from PIL import Image
    generator = np.random.default_rng(seed)
    names = []
    for i, (height, width) in enumerate(sizes):
        names.append(f'image{seed}-{i}.png' if seed != 6 else f'image{i}.png')
        Image.fromarray(generator.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(names[-1])
    return names


def _vlm_vision_cases():
    stimuli = _stimuli('guard-vlm-stimuli', image_file_name=_image_files())
    for case, aggregation in CASE_AGGREGATIONS['vlm_vision']:
        wrapper = vlm_vision_wrapper(layer_aggregation=aggregation, batch_size=2)
        yield case, wrapper(stimuli, layers=['blocks.0', 'blocks.1'])
    # Images already apart on the first axis (no patch counts): the per-image branch.
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    same_size = _stimuli('guard-vlm-per-image', image_file_name=_image_files(((4, 4),) * 3, seed=7))
    wrapper = VLMVisionWrapper(_eighths(_PatchNet(), 5), _PerImageProcessor(), identifier='guard-vlm',
                               batch_size=2, device='cpu')
    yield 'per_image_mean_patches', wrapper(same_size, layers=['blocks.0', 'blocks.1'])


CASES = {'text': _text_cases, 'audio': _audio_cases, 'video': _video_cases, 'vlm_vision': _vlm_vision_cases}


# -- digest -------------------------------------------------------------------

def _coordinate_names(assembly):
    names = set(assembly.coords)
    for dim, index in assembly.indexes.items():
        if getattr(index, 'nlevels', 1) > 1:
            names.discard(dim)
            names.update(index.names)
    return sorted(names)


def digest(assembly) -> dict:
    """Values, dtype, dims and every coordinate of ``assembly``, in a form that compares exactly."""
    values = np.ascontiguousarray(assembly.values)
    coords = {}
    for name in _coordinate_names(assembly):
        coordinate = assembly[name]
        listed = json.dumps([repr(item) for item in coordinate.values.tolist()])
        coords[name] = [list(coordinate.dims), str(coordinate.values.dtype),
                        hashlib.sha256(listed.encode()).hexdigest()]
    return {'dims': list(assembly.dims), 'shape': list(values.shape), 'dtype': str(values.dtype),
            'values': hashlib.sha256(values.tobytes()).hexdigest(), 'coords': coords}


def current_version(wrapper: str) -> int:
    return vars(wrapper_class(wrapper))['_VERSION']


def wrapper_class(wrapper: str):
    __import__(MODULES[wrapper])
    return getattr(sys.modules[MODULES[wrapper]], CLASSES[wrapper])


def _reference() -> dict:
    return json.loads(REFERENCE.read_text()) if REFERENCE.exists() else {}


@pytest.mark.parametrize('wrapper', sorted(CASES))
def test_values_are_pinned_for_the_current_version(wrapper):
    actual = {case: digest(assembly) for case, assembly in CASES[wrapper]()}
    version, recorded = current_version(wrapper), _reference().get(wrapper)
    changed = recorded is not None and recorded['version'] == version and any(
        recorded['cases'].get(case, digest_) != digest_ for case, digest_ in actual.items())
    raise_message = (f'The values that {MODULES[wrapper]}.{CLASSES[wrapper]} extracts changed, and its '
                     f'_VERSION is still {version}. The activation cache would serve the old values. Raise '
                     f'_VERSION in the body of {CLASSES[wrapper]}, then record the new digests with '
                     f'BRAINSCORE_WRAPPER_GUARD_UPDATE=1.')
    if UPDATE:
        assert not changed, raise_message
        reference = _reference()
        reference[wrapper] = {'version': version, 'cases': actual}
        REFERENCE.write_text(json.dumps(reference, indent=1, sort_keys=True) + '\n')
        return
    assert recorded is not None, f'No reference for {wrapper}; record it with BRAINSCORE_WRAPPER_GUARD_UPDATE=1.'
    assert recorded['version'] == version, (
        f'{CLASSES[wrapper]} has _VERSION {version}; the reference is for {recorded["version"]}. '
        f'Record the digests of the new version with BRAINSCORE_WRAPPER_GUARD_UPDATE=1.')
    assert not changed, raise_message
    assert set(recorded['cases']) == set(actual), (
        f'The guard cases of {wrapper} are {sorted(actual)}; the reference holds {sorted(recorded["cases"])}. '
        f'Record a new case with BRAINSCORE_WRAPPER_GUARD_UPDATE=1.')
