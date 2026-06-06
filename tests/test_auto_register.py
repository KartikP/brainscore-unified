"""Tests for brainscore.tools.auto_register — the upstream-of-layer-mapping
tool that infers wrapper, recording layers, and a provisional region_layer_map.

All offline: tiny synthetic torch modules with HuggingFace-style class names /
configs stand in for the real backbones, so block detection + modality
classification are exercised without any weight downloads. The one live path
(auto_register building a real PytorchWrapper) uses a 2-conv CPU module.
"""
import types

import pytest
import torch
import torch.nn as nn

from brainscore.tools.auto_register import (
    find_block_groups, inspect_model, scaffold_registration, space_layers,
    auto_register, BlockGroup, _classify_global, _model_meta,
)


# ── synthetic model factories (mimic HF module trees + configs) ──────────────

def _cfg(model_type='', architectures=None, is_decoder=False,
         is_encoder_decoder=False):
    c = types.SimpleNamespace()
    c.model_type = model_type
    c.architectures = architectures or []
    c.is_decoder = is_decoder
    c.is_encoder_decoder = is_encoder_decoder
    return c


class _FakeParam:
    """Stand-in for a torch parameter — reports a big numel without allocating,
    so the >1B-param warning path is testable without 5GB of RAM."""
    def __init__(self, n, dtype):
        self._n = n
        self.dtype = dtype

    def numel(self):
        return self._n


class _Block(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.attn = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))


def _stack(n, d=8):
    return nn.ModuleList([_Block(d) for _ in range(n)])


def make_gpt(n=6):
    """Causal LM: transformer.h.* + ForCausalLM architecture."""
    m = nn.Module()
    m.transformer = nn.Module()
    m.transformer.h = _stack(n)
    m.config = _cfg(model_type='gpt2', architectures=['GPT2LMHeadModel'])
    return m


def make_bert(n=4):
    """Text encoder: encoder.layer.* + bert model_type."""
    m = nn.Module()
    m.encoder = nn.Module()
    m.encoder.layer = _stack(n)
    m.config = _cfg(model_type='bert', architectures=['BertModel'])
    return m


def make_vit(n=8):
    """Plain vision transformer: encoder.layers.* + vit model_type."""
    m = nn.Module()
    m.encoder = nn.Module()
    m.encoder.layers = _stack(n)
    m.config = _cfg(model_type='vit', architectures=['ViTModel'])
    return m


def make_videomae(n=4):
    m = nn.Module()
    m.encoder = nn.Module()
    m.encoder.layer = _stack(n)
    m.config = _cfg(model_type='videomae', architectures=['VideoMAEModel'])
    return m


def make_wav2vec2(n=4):
    m = nn.Module()
    m.encoder = nn.Module()
    m.encoder.layers = _stack(n)
    m.config = _cfg(model_type='wav2vec2', architectures=['Wav2Vec2Model'])
    return m


def make_clip(nv=6, nt=6):
    """Multi-tower: vision_model + text_model, each its own encoder.layers."""
    m = nn.Module()
    m.vision_model = nn.Module()
    m.vision_model.encoder = nn.Module()
    m.vision_model.encoder.layers = _stack(nv)
    m.text_model = nn.Module()
    m.text_model.encoder = nn.Module()
    m.text_model.encoder.layers = _stack(nt)
    m.config = _cfg(model_type='clip', architectures=['CLIPModel'])
    return m


def make_qwen_vl(nv=4, nt=4):
    """Flattened-patch VLM: visual.blocks.* (vision) + model.layers.* (LM)."""
    m = nn.Module()
    m.visual = nn.Module()
    m.visual.blocks = _stack(nv)
    m.model = nn.Module()
    m.model.layers = _stack(nt)
    m.config = _cfg(model_type='qwen2_5_vl',
                    architectures=['Qwen2_5_VLForConditionalGeneration'])
    return m


# ── block detection ──────────────────────────────────────────────────────────

class TestBlockDetection:
    def test_finds_gpt_stack(self):
        groups = find_block_groups(make_gpt(6))
        assert groups[0].prefix == 'transformer.h'
        assert groups[0].size == 6
        assert groups[0].paths[0] == 'transformer.h.0'
        assert groups[0].paths[-1] == 'transformer.h.5'

    def test_finds_vit_stack(self):
        groups = find_block_groups(make_vit(8))
        assert groups[0].prefix == 'encoder.layers'
        assert groups[0].size == 8

    def test_main_stack_outranks_mlp_subgroups(self):
        """The mlp Sequential has integer children too, but the main stack
        (length 8) must rank first."""
        groups = find_block_groups(make_vit(8))
        assert groups[0].size == 8
        # mlp Sequentials are length-2 groups under each block — present but
        # never first
        assert all(g.size <= 8 for g in groups)

    def test_multi_tower_yields_both_stacks(self):
        groups = find_block_groups(make_clip(6, 6))
        prefixes = {g.prefix for g in groups}
        assert 'vision_model.encoder.layers' in prefixes
        assert 'text_model.encoder.layers' in prefixes

    def test_min_size_drops_singletons(self):
        m = nn.Module()
        m.only = nn.ModuleList([nn.Linear(4, 4)])   # length-1, leaf block
        # the length-1 list produces no group under its own prefix
        assert all(g.prefix != 'only' for g in find_block_groups(m, min_size=2))

    def test_rejects_non_module(self):
        with pytest.raises(TypeError, match="named_modules"):
            find_block_groups(object())


# ── modality classification ──────────────────────────────────────────────────

class TestModalityClassification:
    def test_gpt_is_text_causal(self):
        mod, _ = _classify_global(_model_meta(make_gpt()))
        assert mod == 'text_causal'

    def test_bert_is_text_encoder(self):
        mod, _ = _classify_global(_model_meta(make_bert()))
        assert mod == 'text_encoder'

    def test_vit_is_vision(self):
        mod, _ = _classify_global(_model_meta(make_vit()))
        assert mod == 'vision'

    def test_videomae_is_video(self):
        mod, _ = _classify_global(_model_meta(make_videomae()))
        assert mod == 'video'

    def test_wav2vec2_is_audio(self):
        mod, _ = _classify_global(_model_meta(make_wav2vec2()))
        assert mod == 'audio'

    def test_qwen_vl_global_is_flat_vision(self):
        mod, _ = _classify_global(_model_meta(make_qwen_vl()))
        assert mod == 'vision_flat'


# ── inspect_model end-to-end ─────────────────────────────────────────────────

class TestInspectModel:
    def test_gpt_profile(self):
        p = inspect_model(make_gpt(6), identifier='gpt2-tiny')
        assert not p.is_multimodal
        rec = p.primary
        assert rec.wrapper == 'TextWrapper'
        assert rec.layer_aggregation == 'last_token'
        assert rec.submodule_path is None
        assert len(rec.block_layers) == 6
        assert p.required_modalities() == {'text'}

    def test_vit_profile_provisional_map(self):
        p = inspect_model(make_vit(8), identifier='vit-tiny')
        pmap = p.provisional_region_layer_map()
        # canonical vision regions, V1 shallow, IT at the last block
        assert set(pmap) == {'V1', 'V2', 'V4', 'IT'}
        assert pmap['V1'] == 'encoder.layers.0'
        assert pmap['IT'] == 'encoder.layers.7'

    def test_clip_is_multimodal_two_towers(self):
        p = inspect_model(make_clip(6, 6), identifier='clip-tiny')
        assert p.is_multimodal
        modalities = {r.modality for r in p.recommendations}
        assert 'vision' in modalities
        assert 'text_encoder' in modalities
        # towers wrapped at the sub-module; layers made relative
        vis = next(r for r in p.recommendations if r.modality == 'vision')
        assert vis.submodule_path == 'vision_model'
        assert vis.block_layers[0] == 'encoder.layers.0'
        # multimodal → soft required_modalities
        assert p.required_modalities() == set()

    def test_qwen_vl_two_towers_flat_vision_plus_causal_text(self):
        p = inspect_model(make_qwen_vl(4, 4), identifier='qwen-vl-tiny')
        assert p.is_multimodal
        modalities = {r.modality for r in p.recommendations}
        assert 'vision_flat' in modalities
        assert 'text_causal' in modalities
        vis = next(r for r in p.recommendations if r.modality == 'vision_flat')
        assert vis.wrapper == 'VLMVisionWrapper'
        assert vis.submodule_path == 'visual'
        assert vis.block_layers[0] == 'blocks.0'

    def test_large_model_warns_about_layerpca(self):
        big = make_vit(4)
        # fake a >1B param count WITHOUT allocating (a real 1B Linear is ~5GB)
        big.parameters = lambda: iter([_FakeParam(2_000_000_000, torch.float32)])
        p = inspect_model(big)
        assert any('Large model' in w for w in p.warnings)

    def test_video_note_present(self):
        p = inspect_model(make_videomae(4))
        assert any('post_hook_fn' in n for n in p.notes)

    def test_no_blocks_warns(self):
        m = nn.Module()
        m.fc = nn.Linear(4, 4)
        p = inspect_model(m, identifier='no-blocks')
        assert p.recommendations == []
        assert any('No integer-indexed block' in w for w in p.warnings)

    def test_summary_is_readable_string(self):
        s = inspect_model(make_clip(6, 6), identifier='clip-tiny').summary()
        assert 'clip-tiny' in s
        assert 'tower' in s
        assert 'provisional map' in s


# ── space_layers ─────────────────────────────────────────────────────────────

class TestSpaceLayers:
    def test_single_region_takes_last(self):
        assert space_layers(['a', 'b', 'c'], ['IT']) == {'IT': 'c'}

    def test_even_spacing_endpoints(self):
        layers = [f'l{i}' for i in range(12)]
        m = space_layers(layers, ['V1', 'V2', 'V4', 'IT'])
        assert m['V1'] == 'l0'
        assert m['IT'] == 'l11'
        # monotonically deeper
        idx = [int(m[r][1:]) for r in ('V1', 'V2', 'V4', 'IT')]
        assert idx == sorted(idx)

    def test_empty(self):
        assert space_layers([], ['IT']) == {}


# ── scaffold ─────────────────────────────────────────────────────────────────

class TestScaffold:
    def test_vit_scaffold_has_pytorch_wrapper(self):
        p = inspect_model(make_vit(8), identifier='vit-tiny')
        code = scaffold_registration(p, hf_id='org/vit-tiny',
                                     registry_key='vit-tiny')
        assert 'PytorchWrapper' in code
        assert "model_registry['vit-tiny']" in code
        assert 'REGION_LAYER_MAP' in code
        assert 'org/vit-tiny' in code
        # provisional map present
        assert "'V1'" in code and "'IT'" in code

    def test_gpt_scaffold_imports_text_wrapper(self):
        p = inspect_model(make_gpt(6), identifier='gpt2-tiny')
        code = scaffold_registration(p)
        assert 'from brainscore.model_helpers.text_wrapper import TextWrapper' in code
        assert 'required_modalities' not in code or 'text' in code

    def test_scaffold_is_valid_python(self):
        p = inspect_model(make_vit(8), identifier='vit-tiny')
        code = scaffold_registration(p, registry_key='vit-tiny')
        compile(code, '<scaffold>', 'exec')   # must parse

    def test_large_model_scaffold_carries_warning(self):
        big = make_vit(4)
        big.parameters = lambda: iter([_FakeParam(2_000_000_000, torch.float32)])
        p = inspect_model(big, identifier='big-vit')
        code = scaffold_registration(p, registry_key='big-vit')
        assert 'LayerPCA' in code


# ── live construction (real PytorchWrapper, tiny CPU model) ──────────────────

class _TinyVision(nn.Module):
    """A 2-block conv net with a HF-ish module tree so block detection fires."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([
            nn.Conv2d(3, 4, 3, padding=1), nn.Conv2d(4, 4, 3, padding=1)])
        self.config = _cfg(model_type='vit', architectures=['ViTModel'])

    def forward(self, x):
        for layer in self.encoder.layers:
            x = layer(x)
        return x


class TestAutoRegisterLive:
    def test_builds_brainscore_model_vision(self):
        import numpy as np
        model = _TinyVision()

        def preprocessing(image_filepaths):
            return np.zeros((len(image_filepaths), 3, 8, 8), dtype=np.float32)

        bs_model = auto_register(model, identifier='tiny-vision-test',
                                 preprocessing=preprocessing)
        from brainscore_core.model_interface import BrainScoreModel
        assert isinstance(bs_model, BrainScoreModel)
        # provisional vision map wired
        assert set(bs_model._region_layer_map_dict) == {'V1', 'V2', 'V4', 'IT'}
        assert bs_model._region_layer_map_dict['V1'] == 'encoder.layers.0'
        assert 'vision' in bs_model._preprocessors

    def test_vision_without_preprocessing_errors(self):
        with pytest.raises(ValueError, match="preprocessing"):
            auto_register(_TinyVision(), identifier='x')

    def test_override_region_layer_map(self):
        import numpy as np
        bs = auto_register(
            _TinyVision(), identifier='tiny2',
            preprocessing=lambda f: np.zeros((len(f), 3, 8, 8), np.float32),
            region_layer_map={'IT': 'encoder.layers.1'})
        assert bs._region_layer_map_dict == {'IT': 'encoder.layers.1'}

    def test_multimodal_refuses_live_build(self):
        with pytest.raises(NotImplementedError, match="multi-tower"):
            auto_register(make_clip(6, 6), identifier='clip-tiny',
                          preprocessing=lambda f: f)
