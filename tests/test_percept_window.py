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


# ── real core BrainScoreModel with two real torch towers ───────────────
# REMOTE CPU qualification only (needs torch); never run on the Mac. Each tower
# is a tiny real nn.Module behind a layer-aware extractor that returns a
# NeuroidAssembly, as PytorchWrapper / TextWrapper do.

import pandas as pd
import xarray as xr

from brainscore_core.model_interface import BrainScoreModel, LayerSelector
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

VISION_X = torch.arange(12.).reshape(1, 3, 2, 2)
TEXT_X = torch.tensor([[4, 5, 6]])
VISION_STIM = pd.DataFrame({'stimulus_id': ['i'], 'image_path': ['i.png']})
TEXT_STIM = pd.DataFrame({'stimulus_id': ['s'], 'sentence': ['four five six']})
REGIONS = {'IT': LayerSelector('vision_tower', modality='vision'),
           'language': LayerSelector('text_tower', modality='text')}


def _assembly(stimuli, layer, values):
    """One assembly row per presentation; MultiIndex presentation per BrainIO."""
    values = np.asarray(values).reshape(len(stimuli), -1)
    n = values.shape[1]
    return NeuroidAssembly(values, dims=('presentation', 'neuroid'), coords={
        'presentation': pd.MultiIndex.from_arrays(
            [np.asarray(stimuli['stimulus_id'])], names=['stimulus_id']),
        'neuroid_id': ('neuroid', np.array([f'{layer}.{i}' for i in range(n)])),
        'layer': ('neuroid', np.array([layer] * n))})


EXPECTED_V = _assembly(VISION_STIM, 'vision_tower', (VISION_X * 2.).numpy())
EXPECTED_T = _assembly(TEXT_STIM, 'text_tower', (TEXT_X * 3).numpy())


def _assert_same(actual, expected):
    xr.testing.assert_identical(actual, expected)
    assert actual.dtype == expected.dtype      # assert_identical ignores dtype


class Tower(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.forwards = 0

    def forward(self, x):
        self.forwards += 1
        return x * self.scale


class TowerExtractor:
    """Layer-aware extractor ``(stimuli, layers=...) -> NeuroidAssembly``."""
    def __init__(self, module, identifier, layer, x, processor=None):
        self._model, self.identifier, self.layer, self.x = module, identifier, layer, x
        if processor is not None:
            self.processor = processor

    def __call__(self, stimuli, layers):
        assert list(layers) == [self.layer], layers   # the recording target routed here
        out = self._model(self.x.clone())
        return _assembly(stimuli, self.layer, out.detach().numpy())


def _two_tower_model(form, vision=None):
    vision, text = vision or Tower(2.), Tower(3)
    towers = {'vision': TowerExtractor(vision, 'v', 'vision_tower', VISION_X),
              'text': TowerExtractor(text, 't', 'text_tower', TEXT_X)}
    kwargs = {'extractor': towers} if form == 'extractor' else {'activations_model': towers}
    return BrainScoreModel('two-tower', region_layer_map=REGIONS, **kwargs), vision, text


def _run_each_tower(model):
    model.start_recording('IT')
    out_v = model.process(VISION_STIM)
    model.start_recording('language')
    out_t = model.process(TEXT_STIM)
    return out_v, out_t


@pytest.mark.unit
@pytest.mark.parametrize('form', ['extractor', 'activations_model'])
def test_brainscoremodel_extractor_mapping_captures_both_towers(form):
    model, vision, text = _two_tower_model(form)
    reference = _run_each_tower(model)
    vision.forwards = text.forwards = 0

    eye = PerceptWindow(model)
    assert eye.targets == [('vision:v', vision), ('text:t', text)]
    with eye:
        out_v, out_t = _run_each_tower(model)     # each tower invoked separately

    assert (vision.forwards, text.forwards) == (1, 1)   # no extra forward pass
    assert [c.module for c in eye.captures] == ['vision:v', 'text:t']  # one hook each
    assert torch.equal(eye.captures[0].tensor, VISION_X)
    assert torch.equal(eye.captures[1].tensor, TEXT_X)
    assert [c.modality for c in eye.captures] == ['vision', 'text']
    for actual, ref, expected in ((out_v, reference[0], EXPECTED_V),
                                  (out_t, reference[1], EXPECTED_T)):
        _assert_same(actual, ref)                  # window changes no output
        _assert_same(actual, expected)
    assert out_v.dtype == np.float32 and out_t.dtype == np.int64

    _run_each_tower(model)
    assert len(eye.captures) == 2                       # hooks removed on exit


@pytest.mark.unit
def test_old_and_new_constructor_forms_return_identical_assemblies():
    new, _, _ = _two_tower_model('extractor')
    old, _, _ = _two_tower_model('activations_model')
    with PerceptWindow(new) as eye_new, PerceptWindow(old) as eye_old:
        outs_new, outs_old = _run_each_tower(new), _run_each_tower(old)
    for a, b in zip(outs_new, outs_old):
        _assert_same(a, b)
    for a, b in zip(eye_new.captures, eye_old.captures):
        assert a.module == b.module and torch.equal(a.tensor, b.tensor)


@pytest.mark.unit
def test_brainscoremodel_hooks_removed_when_body_raises():
    model, vision, text = _two_tower_model('extractor')
    with pytest.raises(RuntimeError, match='mid-benchmark'):
        with PerceptWindow(model) as eye:
            model.start_recording('IT')
            model.process(VISION_STIM)
            raise RuntimeError('mid-benchmark')
    assert len(eye.captures) == 1 and not eye._handles
    _run_each_tower(model)
    assert len(eye.captures) == 1


@pytest.mark.unit
def test_brainscoremodel_hooks_removed_when_forward_raises():
    class Boom(nn.Module):
        def forward(self, x):
            raise ValueError('boom')
    model, _, text = _two_tower_model('extractor', vision=Boom())
    with pytest.raises(ValueError, match='boom'):
        with PerceptWindow(model) as eye:
            model.start_recording('IT')
            model.process(VISION_STIM)
    assert len(eye.captures) == 1 and torch.equal(eye.captures[0].tensor, VISION_X)
    model.start_recording('language')
    _assert_same(model.process(TEXT_STIM), EXPECTED_T)
    assert len(eye.captures) == 1


class RefusesHook(Tower):
    """Deterministic failure while attaching to the second root."""
    def __init__(self, error):
        super().__init__(3)
        self.error = error

    def register_forward_pre_hook(self, *args, **kwargs):
        raise self.error


@pytest.mark.unit
@pytest.mark.parametrize('error', [RuntimeError('no hooks here'), KeyboardInterrupt()])
def test_failed_attachment_removes_earlier_hooks(error):
    first, second = Tower(2.), RefusesHook(error)
    eye = PerceptWindow({'vision': first, 'text': second})
    assert [m for _, m in eye.targets] == [first, second]
    with pytest.raises(type(error)) as raised:
        eye.__enter__()
    assert raised.value is error                       # original failure surfaces
    assert eye._handles == []
    first(VISION_X)
    assert eye.captures == []                          # first hook was removed


@pytest.mark.unit
def test_failed_attachment_on_brainscoremodel_tower_cleans_up():
    model = BrainScoreModel('refuse', region_layer_map=REGIONS, extractor={
        'vision': TowerExtractor(Tower(2.), 'v', 'vision_tower', VISION_X),
        'text': TowerExtractor(RefusesHook(RuntimeError('refused')), 't',
                               'text_tower', TEXT_X)})
    eye = PerceptWindow(model)
    with pytest.raises(RuntimeError, match='refused'):
        with eye:
            pass
    model.start_recording('IT')
    _assert_same(model.process(VISION_STIM), EXPECTED_V)
    assert eye.captures == [] and eye._handles == []


@pytest.mark.unit
def test_legacy_shared_activations_model_keeps_name_and_dedups():
    module = Tower(2.)
    wrapper = TowerExtractor(module, 'v', 'vision_tower', VISION_X)
    model = BrainScoreModel('legacy', region_layer_map={'IT': 'vision_tower'},
                            preprocessors={'vision': lambda x: x},
                            activations_model=wrapper)
    assert model.extractor['vision'] is wrapper         # also reachable publicly
    eye = PerceptWindow(model)
    assert eye.targets == [('activations:v', module)]   # legacy name, no duplicate
    with eye:
        model.start_recording('IT')
        _assert_same(model.process(VISION_STIM), EXPECTED_V)
    assert [c.module for c in eye.captures] == ['activations:v']
    assert module.forwards == 1


@pytest.mark.unit
def test_explicit_mapping_target_dedups_by_identity():
    a, b = Tower(2.), Tower(3)
    eye = PerceptWindow({'vision': a, 'text': b, 'again': a})
    assert eye.targets == [('vision:Tower', a), ('text:Tower', b)]
    with eye:
        a(VISION_X)
        b(TEXT_X)
    assert [c.module for c in eye.captures] == ['vision:Tower', 'text:Tower']
    assert torch.equal(eye.captures[0].tensor, VISION_X)
    assert torch.equal(eye.captures[1].tensor, TEXT_X)


@pytest.mark.unit
def test_cyclic_mapping_target_raises_clearly():
    cyclic = {'vision': Tower(2.)}
    cyclic['self'] = cyclic
    with pytest.raises(ValueError, match='contains itself'):
        PerceptWindow(cyclic)
    shared = Tower(2.)                                  # reuse is not a cycle
    assert len(PerceptWindow({'a': [shared], 'b': [shared]}).targets) == 1


@pytest.mark.unit
def test_auto_denorm_from_public_extractor_processor():
    model = BrainScoreModel('denorm', extractor={'vision': TowerExtractor(
        Tower(2.), 'v', 'vision_tower', VISION_X, processor=HFProc())})
    eye = PerceptWindow(model, denorm='auto')
    assert (eye.mean, eye.std) == ([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    mapped = PerceptWindow({'vision': TowerExtractor(
        Tower(2.), 'v', 'vision_tower', VISION_X, processor=HFProc())}, denorm='auto')
    assert mapped.mean == [0.5, 0.5, 0.5]


@pytest.mark.unit
def test_auto_denorm_does_not_invent_constants():
    model, _, _ = _two_tower_model('extractor')     # no processor anywhere
    eye = PerceptWindow(model, denorm='auto')
    assert eye.mean is None and eye.std is None
    assert eye.summary()['denorm_resolved'] is False


@pytest.mark.unit
def test_auto_denorm_prefers_legacy_candidates_first():
    class Other:
        image_mean, image_std = [0.1, 0.1, 0.1], [0.9, 0.9, 0.9]
    target = FakeBSM(activations_model=Wrapper(Doubler()),
                     preprocessors={'vision': HFProc()})
    target.extractor = {'vision': TowerExtractor(
        Tower(2.), 'v', 'vision_tower', VISION_X, processor=Other())}
    eye = PerceptWindow(target, denorm='auto')
    assert eye.mean == [0.5, 0.5, 0.5]
