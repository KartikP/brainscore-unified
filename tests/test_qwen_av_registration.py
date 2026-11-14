"""The Qwen A+V registration builds and declares the shared vision backbone id."""
from unittest.mock import MagicMock, patch

import pytest
import torch


def test_qwen_av_registration_builds_with_shared_backbone_id():
    pytest.importorskip('transformers')
    import transformers
    from brainscore.models.multimodal_av_qwen_wav2vec2 import model as reg

    qwen = MagicMock()
    visual = torch.nn.Module()
    visual.blocks = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(29))
    qwen.model.visual = visual
    audio = MagicMock()
    audio.layer_names.return_value = ['encoder.layers.6']
    with patch('brainscore.models._downloads.hf_preflight', return_value={}), \
            patch.object(transformers.Qwen2_5_VLForConditionalGeneration,
                         'from_pretrained', return_value=qwen), \
            patch.object(transformers.AutoProcessor, 'from_pretrained',
                         return_value=MagicMock()), \
            patch.object(reg, 'pin_image_processor', side_effect=lambda p, *a, **k: p), \
            patch('brainscore.models._multimodal_av_shared.build_audio_wrapper',
                  return_value=(audio, None)):
        model = reg.get_model('qwen2.5-vl-wav2vec2')
    vision = model.extractor['vision']
    assert model.identifier == 'qwen2.5-vl-wav2vec2'
    assert vision.backbone_id == 'qwen2.5-vl-3b-vision'
