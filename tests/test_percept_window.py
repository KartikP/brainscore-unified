"""PerceptWindow: capture + reconstruct what a model actually ingests.

Tests use trivial synthetic torch modules (a few floats through a Module) — not
model forward passes — so they run anywhere. Reconstruction is tested as pure
functions (de-normalize round-trip, detokenize, normalization introspection).
"""
import json
import os
import tempfile

import numpy as np
import pytest
import torch
from torch import nn

from brainscore.percept_window import (
    PerceptWindow,
    infer_modality,
    denormalize_image,
    minmax_image,
    decode_tokens,
    as_waveform,
    find_normalization,
)


# ── toy modules ──────────────────────────────────────────────────────

class Doubler(nn.Module):
    def forward(self, x):
        return x * 2


class KwargModule(nn.Module):
    def forward(self, pixel_values=None):
        return pixel_values + 1


class Wrapper:
    """Stands in for an activations wrapper (exposes ._model)."""
    def __init__(self, m):
        self._model = m
        self.identifier = 'w'


class FakeBSM:
    def __init__(self, activations_model=None, preprocessors=None):
        self._activations_model = activations_model
        self._preprocessors = preprocessors or {}


class HFProc:
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.2, 0.2, 0.2]


# ── capture mechanics ─────────────────────────────────────────────────

def test_captures_positional_input():
    m = Doubler()
    t = torch.arange(6.).reshape(1, 6)
    with PerceptWindow(m) as eye:
        m(t)
    assert len(eye.captures) == 1
    assert torch.equal(eye.captures[0].tensor, t)


def test_captures_keyword_input():
    m = KwargModule()
    t = torch.zeros(1, 3, 8, 8)
    with PerceptWindow(m, modality='vision') as eye:
        m(pixel_values=t)
    assert len(eye.captures) == 1
    assert eye.captures[0].modality == 'vision'
    assert tuple(eye.captures[0].shape) == (1, 3, 8, 8)


def test_stride_captures_every_nth():
    m = Doubler()
    with PerceptWindow(m, every=2) as eye:
        for _ in range(5):
            m(torch.zeros(1, 4))
    assert len(eye.captures) == 3            # calls 1, 3, 5


def test_max_captures_caps():
    m = Doubler()
    with PerceptWindow(m, max_captures=2) as eye:
        for _ in range(5):
            m(torch.zeros(1, 4))
    assert len(eye.captures) == 2


def test_hooks_removed_on_exit():
    m = Doubler()
    with PerceptWindow(m) as eye:
        m(torch.zeros(1, 4))
    assert len(eye.captures) == 1
    m(torch.zeros(1, 4))                     # after the with-block
    assert len(eye.captures) == 1            # no new capture — hook removed


def test_no_hookable_module_raises():
    with pytest.raises(ValueError, match='no torch module'):
        PerceptWindow(object())


# ── target resolution ─────────────────────────────────────────────────

def test_resolves_wrapper_dot_model():
    m = Doubler()
    eye = PerceptWindow(Wrapper(m))
    assert len(eye.targets) == 1 and eye.targets[0][1] is m


def test_resolves_brainscoremodel():
    m = Doubler()
    eye = PerceptWindow(FakeBSM(activations_model=Wrapper(m)))
    assert len(eye.targets) == 1 and eye.targets[0][1] is m


def test_auto_denorm_from_preprocessor():
    m = Doubler()
    eye = PerceptWindow(FakeBSM(activations_model=Wrapper(m),
                                preprocessors={'vision': HFProc()}), denorm='auto')
    assert eye.mean == [0.5, 0.5, 0.5]
    assert eye.std == [0.2, 0.2, 0.2]


# ── pure reconstruction functions ─────────────────────────────────────

def test_infer_modality():
    assert infer_modality(torch.zeros(1, 3, 8, 8)) == 'vision'
    assert infer_modality(torch.zeros(3, 8, 8)) == 'vision'
    assert infer_modality(torch.tensor([[1, 2, 3]])) == 'text'      # int dtype
    assert infer_modality(torch.zeros(1, 1600)) == 'audio'


def test_denormalize_image_round_trip():
    rng = np.random.RandomState(0)
    img = rng.randint(0, 256, (8, 8, 3), dtype=np.uint8)
    mean, std = [0.48, 0.45, 0.40], [0.26, 0.26, 0.27]
    f = img.astype(np.float32) / 255.0
    norm = (f - np.array(mean)) / np.array(std)            # (H, W, C)
    chw = np.transpose(norm, (2, 0, 1))[None]              # (1, 3, H, W)
    recon = denormalize_image(torch.tensor(chw), mean, std)
    assert recon.shape == (1, 8, 8, 3)
    assert np.abs(recon[0].astype(int) - img.astype(int)).max() <= 1


def test_denormalize_image_channel_mismatch_raises():
    with pytest.raises(ValueError, match='channel count'):
        denormalize_image(torch.zeros(1, 3, 4, 4), [0.5], [0.5])


def test_minmax_image_shape():
    out = minmax_image(torch.randn(1, 3, 5, 5))
    assert out.shape == (1, 5, 5, 3)
    assert out.dtype == np.uint8


def test_decode_tokens():
    class FakeTok:
        def decode(self, ids, skip_special_tokens=False):
            return ' '.join(str(i) for i in ids)
    out = decode_tokens(torch.tensor([[1, 2, 3], [4, 5, 6]]), FakeTok())
    assert out == ['1 2 3', '4 5 6']


def test_as_waveform_shapes():
    assert as_waveform(torch.zeros(1600)).shape == (1, 1600)
    assert as_waveform(torch.zeros(2, 1600)).shape == (2, 1600)
    assert as_waveform(torch.zeros(2, 1, 1600)).shape == (2, 1600)


def test_find_normalization():
    assert find_normalization(HFProc()) == ([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])

    class Norm:
        mean, std = [1.0], [2.0]

    class Compose:
        transforms = [object(), Norm()]
    assert find_normalization(Compose()) == ([1.0], [2.0])
    assert find_normalization(None) is None
    assert find_normalization(object()) is None


# ── reconstruct() + save() ─────────────────────────────────────────────

def test_reconstruct_vision_with_and_without_denorm():
    m = Doubler()
    with PerceptWindow(m, modality='vision',
                       denorm=([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])) as eye:
        m(torch.rand(1, 3, 4, 4))
    rec = eye.reconstruct()[0]
    assert rec['kind'] == 'image' and rec['denorm'] == 'mean_std'
    assert rec['data'].shape == (1, 4, 4, 3)

    with PerceptWindow(m, modality='vision') as eye2:      # no denorm
        m(torch.randn(1, 3, 4, 4))
    assert eye2.reconstruct()[0]['denorm'] == 'minmax-fallback'


def test_save_text_artifacts(tmp_path):
    class FakeTok:
        def decode(self, ids, skip_special_tokens=False):
            return 'tok:' + '_'.join(str(i) for i in ids)
    m = Doubler()
    with PerceptWindow(m, modality='text') as eye:
        m(torch.tensor([[10, 11, 12]]))
    res = eye.save(str(tmp_path), tokenizer=FakeTok())
    man = json.load(open(res['manifest']))
    assert man['summary']['n_captures'] == 1
    art = man['captures'][0]['artifacts'][0]
    assert os.path.exists(art)
    assert open(art).read() == 'tok:10_11_12'


def test_save_vision_artifacts(tmp_path):
    pytest.importorskip('PIL')
    m = Doubler()
    with PerceptWindow(m, modality='vision', denorm=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])) as eye:
        m(torch.rand(1, 3, 4, 4))
    res = eye.save(str(tmp_path))
    man = json.load(open(res['manifest']))
    assert man['captures'][0]['modality'] == 'vision'
    assert man['captures'][0]['artifacts'][0].endswith('.png')
    assert os.path.exists(man['captures'][0]['artifacts'][0])
