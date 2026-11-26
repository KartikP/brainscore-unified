"""The served Kimi K2.5: the vllm provider reaches the stated address, the local registration stores no
reply, and the vision hooks inactivate and record as stated (on toy blocks, no GPU, no vLLM)."""
import json
import sys
import types

import numpy as np
import pytest
import torch

from brainscore.models.kimi_k25_local.vllm_hooks import (KimiHooks, VisionHooks, apply_lesion, picture_means,
                                                         read_lesion, set_stream)
from brainscore.models.kimi_k25_local.localizer import phase_scrambled, project_out, sdpa_packed, zero


def test_a_listed_channel_is_zero_for_every_token_and_the_others_keep_their_values():
    output = torch.randn(7, 6)
    changed = apply_lesion(output, {'name': 'x', 'block': 0, 'channels': [1, 4]})
    assert changed[:, [1, 4]].abs().sum() == 0
    assert torch.equal(changed[:, [0, 2, 3, 5]], output[:, [0, 2, 3, 5]])
    assert torch.equal(zero(output, [1, 4]), changed)


def test_a_direction_lesion_leaves_no_component_along_the_direction():
    output, direction = torch.randn(9, 5), torch.randn(5)
    changed = apply_lesion(output, {'name': 'x', 'block': 0, 'direction': (3 * direction).tolist()})
    assert torch.allclose(changed @ (direction / direction.norm()), torch.zeros(9), atol=1e-5)
    assert torch.allclose(project_out(output, direction), changed, atol=1e-6)


def test_a_mean_ablation_sets_each_channel_to_its_value_and_a_direction_to_its_value():
    output, direction = torch.randn(7, 6), torch.randn(6)
    changed = apply_lesion(output, {'name': 'x', 'block': 0, 'channels': [1, 4], 'values': [0.5, -2.0]})
    assert torch.equal(changed[:, 1], torch.full((7,), 0.5)) and torch.equal(changed[:, 4], torch.full((7,), -2.0))
    assert torch.equal(changed[:, [0, 2, 3, 5]], output[:, [0, 2, 3, 5]])
    moved = apply_lesion(output, {'name': 'x', 'block': 0, 'direction': direction.tolist(), 'value': 0.7})
    assert torch.allclose(moved @ (direction / direction.norm()), torch.full((7,), 0.7), atol=1e-5)


def test_a_missing_or_malformed_lesion_file_is_an_error_never_intact(tmp_path):
    with pytest.raises(RuntimeError, match='cannot be read'):
        read_lesion(str(tmp_path / 'none.json'))
    (tmp_path / 'bad.json').write_text(json.dumps({'name': 'x', 'block': 3}))
    with pytest.raises(RuntimeError, match='needs'):
        read_lesion(str(tmp_path / 'bad.json'))


def test_picture_means_follow_the_packed_bounds():
    output = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    means = picture_means(output, torch.tensor([0, 2, 6]))
    assert torch.equal(means, torch.stack([output[:2].mean(0), output[2:].mean(0)]))


class _Block(torch.nn.Module):
    def forward(self, hidden, cu_seqlens):
        return hidden + 1


def _worker(n_blocks=4):
    worker = VisionHooks()
    blocks = torch.nn.ModuleList([_Block() for _ in range(n_blocks)])
    model = types.SimpleNamespace(vision_tower=types.SimpleNamespace(encoder=types.SimpleNamespace(blocks=blocks)))
    worker.model_runner, worker.rank = types.SimpleNamespace(get_model=lambda: model), 0
    return worker, blocks


def _encode(blocks, hidden, cu):
    for block in blocks:
        hidden = block(hidden, cu)
    return hidden


def test_the_hook_reads_the_condition_at_each_call_and_records_with_its_name(tmp_path):
    worker, blocks = _worker()
    lesion = tmp_path / 'lesion.json'
    lesion.write_text(json.dumps({'name': 'intact'}))
    worker.kimi_hooks_install(json.dumps({'lesion_file': str(lesion), 'record_dir': str(tmp_path / 'rec'),
                                          'record_blocks': [2], 'full_blocks': [3]}))
    cu = torch.tensor([0, 2, 5])
    intact = _encode(blocks, torch.zeros(5, 3), cu)
    assert torch.equal(intact, torch.full((5, 3), 4.0))
    lesion.write_text(json.dumps({'name': 'c1', 'block': 2, 'channels': [1]}))
    lesioned = _encode(blocks, torch.zeros(5, 3), cu)
    assert torch.equal(lesioned[:, 1], torch.ones(5)) and torch.equal(lesioned[:, [0, 2]], intact[:, [0, 2]])
    calls = [json.loads(l) for l in open(tmp_path / 'rec' / 'calls-rank0.jsonl')]
    assert [c['lesion'] for c in calls] == ['intact', 'c1'] and calls[0]['pictures'] == 2
    record = torch.load(tmp_path / 'rec' / 'rank0-call000002-block02.pt')
    assert record['lesion'] == 'c1' and record['means'].shape == (2, 3) and record['means'][:, 1].abs().sum() == 0
    assert 'output' in torch.load(tmp_path / 'rec' / 'rank0-call000002-block03.pt')
    assert worker.kimi_hooks_status() == {'rank': 0, 'calls': 2, 'lesion': 'c1'}


def test_one_lesion_file_can_change_several_blocks(tmp_path):
    worker, blocks = _worker()
    lesion = tmp_path / 'lesion.json'
    lesion.write_text(json.dumps({'name': 'two', 'lesions': [{'block': 0, 'channels': [0]},
                                                              {'block': 2, 'channels': [2]}]}))
    worker.kimi_hooks_install(json.dumps({'lesion_file': str(lesion)}))
    out = _encode(blocks, torch.zeros(5, 3), torch.tensor([0, 5]))
    assert torch.equal(out[:, 0], torch.full((5,), 3.0)) and torch.equal(out[:, 2], torch.ones(5))
    assert torch.equal(out[:, 1], torch.full((5,), 4.0))
    lesion.write_text(json.dumps({'name': 'twice', 'lesions': [{'block': 1, 'channels': [0]}, {'block': 1, 'channels': [1]}]}))
    with pytest.raises(RuntimeError, match='twice'):
        read_lesion(str(lesion))


def test_installing_again_replaces_the_hooks(tmp_path):
    worker, blocks = _worker()
    lesion = tmp_path / 'lesion.json'
    lesion.write_text(json.dumps({'name': 'c0', 'block': 0, 'channels': [0]}))
    for _ in range(2):
        worker.kimi_hooks_install(json.dumps({'lesion_file': str(lesion)}))
    assert len(blocks[0]._forward_hooks) == 1


class _Layer(torch.nn.Module):
    def forward(self, positions, hidden, residual):
        residual = hidden if residual is None else hidden + residual
        return hidden * 0 + 1, residual


def _language_worker():
    worker = KimiHooks()
    model = types.SimpleNamespace(language_model=types.SimpleNamespace(model=types.SimpleNamespace(
        layers=torch.nn.ModuleList([_Layer() for _ in range(3)]))), embed_input_ids=lambda ids, *a, **k: ids.float()[:, None].repeat(1, 4))
    worker.model_runner, worker.rank = types.SimpleNamespace(get_model=lambda: model), 0
    return worker, model


def _language_step(model, mask):
    hidden, residual = model.embed_input_ids(torch.arange(5), None, is_multimodal=mask), None
    for layer in model.language_model.model.layers:
        hidden, residual = layer(None, hidden, residual)
    return hidden + residual


def test_the_stream_is_set_at_the_picture_positions_only_and_recorded(tmp_path):
    worker, model = _language_worker()
    lesion = tmp_path / 'lesion.json'
    lesion.write_text(json.dumps({'name': 'lang', 'language': [{'layer': 1, 'dims': [2], 'values': [-7.0]}]}))
    worker.kimi_language_install(json.dumps({'lesion_file': str(lesion), 'record_dir': str(tmp_path / 'rec')}))
    mask = torch.tensor([False, True, True, False, False])
    stream = _language_step(model, mask)
    intact = torch.arange(5, dtype=torch.float32)[:, None].repeat(1, 4) + 3      # three layers, each adds 1
    assert torch.equal(stream[~mask], intact[~mask]) and torch.equal(stream[mask][:, [0, 1, 3]], intact[mask][:, [0, 1, 3]])
    assert torch.equal(stream[mask][:, 2], torch.full((2,), -6.0))    # set to -7 after layer 1; layer 2 adds 1
    worker.kimi_language_flush()
    record = torch.load(next((tmp_path / 'rec').glob('lang-step*.pt')))
    assert record['lesion'] == 'lang' and record['picture_tokens'] == 2 and record['picture_mean'].shape == (3, 4)
    assert float(record['picture_mean'][1, 2]) == -7.0


def test_set_stream_moves_a_direction_to_its_value():
    hidden, residual, u = torch.randn(4, 3), torch.randn(4, 3), torch.randn(3)
    mask = torch.tensor([True, False, True, False])
    changed = set_stream(hidden, residual, mask, {'direction': u.tolist(), 'value': 0.3})
    assert torch.allclose((hidden + changed)[mask] @ (u / u.norm()), torch.full((2,), 0.3), atol=1e-5)
    assert torch.equal(changed[~mask], residual[~mask])


def test_sdpa_packed_keeps_the_pictures_apart():
    q, k, v = (torch.randn(5, 2, 4) for _ in range(3))
    out = sdpa_packed(q, k, v, q_cu_seqlens=torch.tensor([0, 2, 5]))
    alone = sdpa_packed(q[2:], k[2:], v[2:], q_cu_seqlens=torch.tensor([0, 3]))
    assert out.shape == (5, 8) and torch.allclose(out[2:], alone, atol=1e-6)


def test_the_scrambled_picture_keeps_the_size_and_loses_the_layout():
    from PIL import Image
    pixels = np.zeros((32, 32, 3), np.uint8)
    pixels[8:24, 8:24] = 200
    image = Image.fromarray(pixels)
    scrambled = np.asarray(phase_scrambled(image, 0), dtype=float)
    assert scrambled.shape == pixels.shape and np.corrcoef(scrambled[..., 0].ravel(), pixels[..., 0].ravel())[0, 1] < 0.5


def test_the_vllm_provider_reaches_the_stated_address_with_no_key(monkeypatch):
    seen = {}

    class _Client:
        def __init__(self, api_key, base_url):
            seen.update(api_key=api_key, base_url=base_url)
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, model, messages, **request):
            seen.update(model=model, request=request)
            message = types.SimpleNamespace(content='ok', reasoning_content=None, reasoning=None)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)

    monkeypatch.setitem(sys.modules, 'openai', types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setenv('BRAINSCORE_VLLM_URL', 'http://gpu-machine:8000/v1')
    monkeypatch.delenv('BRAINSCORE_VLLM_KEY', raising=False)
    from brainscore.model_helpers.api_behavioral import PROVIDERS
    text = PROVIDERS['vllm']('moonshotai/Kimi-K2.5', 'system', [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'hi'}]}], {'max_tokens': 8, 'temperature': 0, 'seed': 1})
    assert text == 'ok' and seen['base_url'] == 'http://gpu-machine:8000/v1' and seen['api_key']
    assert seen['request'] == {'max_tokens': 8, 'temperature': 0, 'seed': 1}

