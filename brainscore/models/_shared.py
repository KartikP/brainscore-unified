"""Family plumbing shared by the model registrations, each part once.

- Tubelet video models (V-JEPA 1 and 2, VideoMAE): ``tubelet_spatial_mean``
  (the ``post_hook_fn``) and ``fit_frames`` (pads or subsamples a clip to the
  frame count of the model).
- Wav2Vec2-base: ``build_audio_wrapper`` and its wav loader, for the audio
  extractors and the random-init null.
- HuggingFace image processors (CLIP, BLIP-2): the image loader of their
  ``PytorchWrapper`` extractors.

A registration then names its network, layers and real choices only.
"""
from typing import List

import numpy as np
from PIL import Image


AUDIO_HF_ID = 'facebook/wav2vec2-base'
AUDIO_LAYER = 'encoder.layers.6'


def tubelet_spatial_mean(arr: np.ndarray, temporal_steps: int, spatial_patches: int) -> np.ndarray:
    """A tubelet model's hook output ``(B, T*S, H)`` as ``(B, T, H)``: the mean over the spatial patches.

    Tokens are ordered all spatial patches at time 0, then time 1, and so on.
    ``temporal_steps`` and ``spatial_patches`` are the model's configuration;
    another sequence length is read as ``T`` steps of ``spatial_patches``, else
    ``temporal_steps`` steps.
    """
    B, seq_len, hidden = arr.shape
    if seq_len == temporal_steps * spatial_patches:
        t_steps, s_steps = temporal_steps, spatial_patches
    elif seq_len % spatial_patches == 0:
        t_steps, s_steps = seq_len // spatial_patches, spatial_patches
    elif seq_len % temporal_steps == 0:
        t_steps, s_steps = temporal_steps, seq_len // temporal_steps
    else:
        raise ValueError(f"tubelet post-hook: unable to infer (T, S) from seq_len={seq_len}.")
    return arr.reshape(B, t_steps, s_steps, hidden).mean(axis=2)


def fit_frames(frames, count: int) -> List[np.ndarray]:
    """Exactly ``count`` frames: the last frame repeated for a short clip, an even subsample of a long one.

    The default frame sampler gives fewer frames for a clip with fewer than
    ``num_frames`` (Lahner2024 has clips at 15 fps x 3 s), and a fixed count
    keeps one number of tubelets per clip.
    """
    frames = list(frames)
    if len(frames) == 0:
        raise ValueError("preprocess received an empty frame list")
    if len(frames) < count:
        return frames + [frames[-1]] * (count - len(frames))
    if len(frames) > count:
        return [frames[int(round(i * (len(frames) - 1) / (count - 1)))] for i in range(count)]
    return frames


def _load_preprocess_images(image_filepaths, processor):
    """The images at ``image_filepaths`` as RGB, prepared by a HuggingFace image ``processor``."""
    images = []
    for path in image_filepaths:
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img.copy())
    processed = processor(images=images, return_tensors='pt')
    return processed['pixel_values'].numpy()


def build_audio_wrapper(combo_identifier: str, random_init: bool = False):
    """Wav2Vec2-base AudioWrapper. Identical contract to the
    multimodal_av_vjepa_wav2vec2 audio path; consolidated here so all
    VLM+audio combos share a single source of truth."""
    from transformers import (
        Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2FeatureExtractor)
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    from scipy.io import wavfile
    import torch

    if random_init:
        config = Wav2Vec2Config.from_pretrained(AUDIO_HF_ID)
        torch.manual_seed(0)
        model = Wav2Vec2Model(config).eval()
    else:
        model = Wav2Vec2Model.from_pretrained(AUDIO_HF_ID).eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_HF_ID)

    def _wav_loader(audio_path: str, target_rate: int) -> np.ndarray:
        sr, wav = wavfile.read(audio_path)
        if sr != target_rate:
            raise ValueError(
                f"WAV at {audio_path} has sr={sr}, expected {target_rate}.")
        if wav.dtype.kind == 'i':
            wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
        else:
            wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        return wav

    wrapper = AudioWrapper(
        model=model,
        processor=processor,
        identifier=f'{combo_identifier}-audio',
        layer_aggregation='sequence',
        audio_loader=_wav_loader,
    )
    return wrapper, model
