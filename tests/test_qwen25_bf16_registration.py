"""Precision identities and reply metadata, with tiny stand-ins and no weights."""
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from brainscore import load_model, model_registry
from brainscore_core import TaskContext
from brainscore_core.responses import Text
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.models.qwen25_vl_3b import model as registration


@pytest.fixture
def build(monkeypatch, tmp_path):
    import transformers.utils.hub as hub

    calls, wrappers = [], []
    weights = tmp_path / 'shard.safetensors'
    weights.write_bytes(b'fixture weights')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': weights.name}}))
    class FakeModel(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
            self.model = SimpleNamespace(visual=torch.nn.Identity(), language_model=torch.nn.Identity())
            self.generation_config = SimpleNamespace(to_dict=lambda: {'max_length': 20})
        @property
        def device(self):
            return self.weight.device
        @property
        def dtype(self):
            return self.weight.dtype
        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4]])
    def load(name, **kwargs):
        calls.append((name, kwargs))
        return FakeModel(kwargs['torch_dtype'])
    class Inputs(dict):
        def to(self, device):
            return self
    class Processor:
        tokenizer = SimpleNamespace(decode=lambda *a, **k: 'Yes')
        image_processor = SimpleNamespace(to_dict=lambda: {'do_resize': True})
        def apply_chat_template(self, *a, **k):
            return 'prompt'
        def __call__(self, **kwargs):
            return Inputs(input_ids=torch.tensor([[1, 2, 3]]))
    processor = Processor()
    class Wrapper:
        def __init__(self, **kwargs):
            self.identifier = kwargs['identifier']
            wrappers.append(kwargs)
        def __call__(self, stimuli, layers):
            raise AssertionError('This fixture must not extract activations')
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(__version__='fixture',
        Qwen2_5_VLForConditionalGeneration=SimpleNamespace(from_pretrained=load),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: processor)))
    monkeypatch.setattr(hub, 'cached_file', lambda repo, name, **kwargs: str(tmp_path / name))
    monkeypatch.setattr(registration, 'pin_image_processor', lambda value, *a, **k: value)
    monkeypatch.setattr('brainscore.models._downloads.hf_preflight', lambda *a, **k: {'local_files_only': True, **k})
    monkeypatch.setattr('brainscore.model_helpers._device.select_device', lambda: torch.device('cpu'))
    monkeypatch.setattr('brainscore.model_helpers.vision_wrapper.VisionWrapper', Wrapper)
    monkeypatch.setattr('brainscore.model_helpers.text_wrapper.TextWrapper', Wrapper)
    return calls, wrappers


def test_bf16_is_a_separate_lazy_registration_and_uses_pinned_precision(build):
    calls, wrappers = build
    assert registration.BF16_IDENTIFIER in model_registry
    subject = load_model(registration.BF16_IDENTIFIER)
    assert calls == [('Qwen/Qwen2.5-VL-3B-Instruct', {'torch_dtype': torch.bfloat16,
                     'local_files_only': True, 'revision': registration.BF16_REVISION})]
    assert subject.identifier == 'qwen2.5-vl-3b-bf16'
    assert [entry['identifier'] for entry in wrappers] == ['qwen2.5-vl-3b-bf16-vision', 'qwen2.5-vl-3b-bf16-text']
    assert {'behavior:text', 'neural:IT', 'neural:language_system'} <= subject.out_channels
    assert 'lesion' not in subject.in_channels
    session = stimulus_session(StimulusSet([{'stimulus_id': 'a', 'sentence': 'red?'}]), ['behavior:text'], task_context=TaskContext(response=Text()))
    subject.participate(session)
    response = session.collect('behavior:text')
    assert response.values.tolist() == ['Yes']
    provenance = response.attrs['provenance']
    assert provenance['dtype'] == 'torch.bfloat16' and provenance['device'] == 'cpu'
    assert provenance['revision'] == registration.BF16_REVISION
    assert provenance['weights_sha256'] == {'shard.safetensors': hashlib.sha256(b'fixture weights').hexdigest()}
    assert provenance['reply_generation'] == {'max_new_tokens': 64, 'do_sample': False}


@pytest.mark.parametrize('identifier', ['qwen2.5-vl-3b', 'qwen2.5-vl-7b',
                                      'qwen2.5-vl-3b-vwfa', 'qwen2.5-vl-7b-vwfa'])
def test_existing_registrations_keep_float16_without_new_revision_or_cache_keys(build, identifier):
    calls, wrappers = build
    subject = load_model(identifier)
    assert calls[0][1] == {'torch_dtype': torch.float16, 'local_files_only': True}
    assert subject.identifier == identifier
    assert [entry['identifier'] for entry in wrappers] == [f'{identifier}-vision', f'{identifier}-text']
