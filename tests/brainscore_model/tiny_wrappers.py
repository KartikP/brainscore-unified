"""The five extractors of ``brainscore.model_helpers`` on tiny seeded networks, CPU, no weights.

Each factory returns ``(wrapper, stimuli, regions, network)``: three stimuli
written to ``folder``, four regions (one per layer, routed to the wrapper's
input) and the network the wrapper runs. Batch size 1 everywhere, so a passive
call and a pass made by another part show the network the same input.
"""
import numpy as np
import torch

from brainscore import LayerSelector
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

KINDS = ('text', 'vlm_vision', 'vlm_video', 'video', 'audio')


def _seeded(seed, build):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return build().eval()


def _regions(modality, layers):
    return {region: LayerSelector(layer, modality=modality)
            for region, layer in zip(('V1', 'V4', 'IT', 'A1'), layers)}


def _table(ids, column, values, identifier):
    stimuli = StimulusSet({'stimulus_id': ids, column: values})
    stimuli.identifier = identifier
    if column.endswith('_path'):
        stimuli.stimulus_paths = dict(zip(ids, values))
    return stimuli


def _write_clip(path, seed):
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 4.0, (16, 16))
    rng = np.random.default_rng(seed)
    for _ in range(8):
        writer.write(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
    writer.release()
    return str(path)


def text(folder):
    import tokenizers
    from transformers import GPT2Config, GPT2Model, PreTrainedTokenizerFast
    from brainscore.model_helpers.text_wrapper import TextWrapper
    vocab = {'[PAD]': 0, '[UNK]': 1, 'the': 2, 'cat': 3, 'dog': 4, 'sat': 5, 'ran': 6}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]',
                                        model_max_length=64)
    network = _seeded(17, lambda: GPT2Model(GPT2Config(vocab_size=7, n_positions=64, n_embd=8,
                                                       n_layer=2, n_head=2)))
    wrapper = TextWrapper(network, tokenizer, identifier='tiny-text', layer_aggregation='last_token',
                          max_length=64, batch_size=1, device='cpu')
    stimuli = _table(['a', 'b', 'c'], 'sentence', ['the cat sat', 'the dog ran the', 'cat'], 'tiny-text-set')
    return wrapper, stimuli, _regions('text', ('h.0', 'h.1', 'ln_f', 'h.1.mlp.c_proj:input')), network


class _Images:
    def __call__(self, images=None, return_tensors='pt'):
        return {'pixel_values': torch.tensor(np.stack(
            [np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255 for image in images]))}


def vlm_vision(folder):
    from PIL import Image
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    rng = np.random.default_rng(3)
    paths = []
    for index in range(3):
        path = folder / f'image{index}.png'
        Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    network = _seeded(5, lambda: torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU(),
                                                     torch.nn.Conv2d(4, 2, 3), torch.nn.Flatten()))
    wrapper = VLMVisionWrapper(network, _Images(), identifier='tiny-vlm', batch_size=1, device='cpu')
    stimuli = _table(['i0', 'i1', 'i2'], 'image_path', paths, 'tiny-images')
    return wrapper, stimuli, _regions('vision', ('0', '1', '2', '3')), network


class _ClipProcessor:
    """Qwen-like: samples at 2 fps, pairs frames into temporal patches of 2x2 positions."""
    fps, temporal_patch_size = 2, 2

    def __call__(self, videos, video_metadata, return_tensors):
        frames = videos[0].astype(np.float32) / 255
        metadata = video_metadata[0]
        fps = metadata.fps if hasattr(metadata, 'fps') else metadata['fps']
        sampled = frames[::int(round(fps / self.fps))]
        patches = []
        for k in range(len(sampled) // self.temporal_patch_size):
            pair = sampled[2 * k:2 * k + 2].mean(axis=0)
            for qh in range(2):
                for qw in range(2):
                    block = pair[qh * 8:(qh + 1) * 8, qw * 8:(qw + 1) * 8]
                    patches.append([block.mean(), block.std(), float(qh), float(qw)])
        steps = len(sampled) // self.temporal_patch_size
        return {'pixel_values_videos': torch.tensor(patches), 'video_grid_thw': torch.tensor([[steps, 2, 2]])}


class _ClipProcessors:
    video_processor = _ClipProcessor()


class _Tower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 5))])

    def forward(self, hidden, grid_thw):
        return self.blocks[0](hidden)


def vlm_video(folder):
    from brainscore.model_helpers.vlm_video_wrapper import VLMVideoWrapper
    network = _seeded(7, _Tower)
    wrapper = VLMVideoWrapper(network, _ClipProcessors(), identifier='tiny-vlm-video')
    paths = [_write_clip(folder / f'vlm_clip{index}.avi', 10 + index) for index in range(3)]
    stimuli = _table(['c0', 'c1', 'c2'], 'video_path', paths, 'tiny-vlm-clips')
    return wrapper, stimuli, _regions('video', ('blocks.0.0', 'blocks.0.1', 'blocks.0.2', 'blocks.0')), network


class _VideoNet(torch.nn.Module):
    """``(B, T, C, H, W)`` -> ``(B, T, features)`` through four named layers."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Linear(3 * 8 * 8, 6)
        self.block0 = torch.nn.Linear(6, 6)
        self.block1 = torch.nn.Linear(6, 5)
        self.head = torch.nn.Linear(5, 4)

    def forward(self, video):
        batch, steps = video.shape[:2]
        hidden = torch.relu(self.block0(self.embed(video.reshape(batch, steps, -1))))
        return self.head(self.block1(hidden))


def _frames_to_tensor(frames):
    small = np.stack([frame[::2, ::2][:8, :8] for frame in frames]).astype(np.float32) / 255
    return torch.from_numpy(small.transpose(0, 3, 1, 2).copy())


def video(folder):
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    network = _seeded(9, _VideoNet)
    wrapper = VideoWrapper(network, _frames_to_tensor, identifier='tiny-video', num_frames=4,
                           batch_size=1, device='cpu')
    paths = [_write_clip(folder / f'clip{index}.avi', 20 + index) for index in range(3)]
    stimuli = _table(['v0', 'v1', 'v2'], 'video_path', paths, 'tiny-clips')
    return wrapper, stimuli, _regions('video', ('embed', 'block0', 'block1', 'head')), network


class _Waveforms:
    """A Wav2Vec2-like processor: pads waveforms to the longest, with an attention mask."""
    sampling_rate = 160

    def __call__(self, waveforms, sampling_rate, return_tensors='pt', padding=True):
        longest = max(len(w) for w in waveforms)
        values = np.zeros((len(waveforms), longest), dtype=np.float32)
        mask = np.zeros((len(waveforms), longest), dtype=np.int64)
        for index, waveform in enumerate(waveforms):
            values[index, :len(waveform)] = waveform
            mask[index, :len(waveform)] = 1
        return {'input_values': torch.from_numpy(values), 'attention_mask': torch.from_numpy(mask)}


class _AudioNet(torch.nn.Module):
    """``(B, samples)`` -> frames of 16 samples -> ``(B, T, features)`` through four named layers."""

    def __init__(self):
        super().__init__()
        self.frontend = torch.nn.Linear(16, 6)
        self.layer0 = torch.nn.Linear(6, 6)
        self.layer1 = torch.nn.Linear(6, 5)
        self.head = torch.nn.Linear(5, 3)

    def forward(self, input_values, attention_mask=None):
        batch, samples = input_values.shape
        frames = input_values[:, :samples // 16 * 16].reshape(batch, samples // 16, 16)
        hidden = torch.relu(self.layer0(self.frontend(frames)))
        return self.head(self.layer1(hidden))


def audio(folder, aggregation='time_series'):
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    network = _seeded(11, _AudioNet)
    rng = np.random.default_rng(13)
    paths = []
    for index in range(3):
        path = folder / f'sound{index}.npy'
        np.save(path, rng.standard_normal(320).astype(np.float32))
        paths.append(str(path))
    wrapper = AudioWrapper(network, _Waveforms(), identifier='tiny-audio', layer_aggregation=aggregation,
                           batch_size=1, audio_loader=lambda path, rate: np.load(path), device='cpu')
    stimuli = _table(['s0', 's1', 's2'], 'audio_path', paths, 'tiny-sounds')
    return wrapper, stimuli, _regions('audio', ('frontend', 'layer0', 'layer1', 'head')), network


FACTORIES = {'text': text, 'vlm_vision': vlm_vision, 'vlm_video': vlm_video, 'video': video, 'audio': audio}
MODALITY = {'text': 'text', 'vlm_vision': 'vision', 'vlm_video': 'video', 'video': 'video', 'audio': 'audio'}


def show(wrapper, stimuli, stimulus_id, layer):
    """Show one stimulus to the network the way the wrapper does: what an answering part runs."""
    row = stimuli[stimuli['stimulus_id'] == stimulus_id]
    one = StimulusSet({column: list(row[column].values) for column in row.columns})
    paths = getattr(stimuli, 'stimulus_paths', None)
    if paths:
        one.stimulus_paths = {stimulus_id: paths[stimulus_id]}
    with torch.no_grad():
        wrapper(one, layers=[layer])
