"""The legacy vision wrapper of the CLIP and BLIP-2 audio-video combinations keeps its cache path.

The legacy ``PytorchWrapper`` keys its stored entries by ``backbone_id``. These
registrations pass ``backbone_id='<combination>-vision'``, as at ``9442779``;
the stored-method identifier they give must stay the same string. Stand-in
networks replace the checkpoints.
"""
import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch

pytestmark = pytest.mark.unit

STORED = 'brainscore_vision.model_helpers.activations.core.ActivationsExtractorHelper._from_paths_stored'


class _Tower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layers = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(40))

    def forward(self, x):
        for layer in self.encoder.layers:
            x = layer(x)
        return x


class _Checkpoint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = _Tower()


class _Processor:
    """A stand-in image processor the cache key can describe."""

    def __call__(self, images, return_tensors='pt'):
        return {'pixel_values': torch.zeros(len(images), 2)}


def _audio():
    audio = MagicMock()
    audio.layer_names.return_value = ['encoder.layers.6']
    return audio, None


class _Stop(Exception):
    pass


def _stored_identifier(wrapper, tmp_path, monkeypatch):
    """The ``@store_xarray`` function identifier that the legacy wrapper gives one lookup."""
    import result_caching
    helper = wrapper._extractor
    stored = type(helper)._from_paths_stored
    storage = next(cell.cell_contents for cell in stored.__closure__
                   if isinstance(cell.cell_contents, result_caching._Storage))
    function = next(cell.cell_contents for cell in stored.__closure__ if inspect.isfunction(cell.cell_contents))
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        raise _Stop
    monkeypatch.setenv('RESULTCACHING_DISABLE', '0')
    monkeypatch.setattr(helper, '_from_paths_stored', capture)
    (tmp_path / 'a.png').write_bytes(b'image')
    with pytest.raises(_Stop):
        helper.from_paths([str(tmp_path / 'a.png')], ['encoder.layers.1'], 'stimuli')
    call_args = storage.getcallargs(function, helper, **captured)
    return storage.get_function_identifier(function, call_args)


@pytest.mark.parametrize('identifier', ['clip-wav2vec2', 'random-clip-wav2vec2'])
def test_clip_combination_keeps_its_legacy_cache_path(identifier, tmp_path, monkeypatch):
    from brainscore.models.clip_vit_b_32 import model as family     # the family's one load
    from brainscore.models.multimodal_av_clip_wav2vec2 import model as registration

    class _CLIPModel:                          # CLIPModel(config) and CLIPModel.from_pretrained
        def __new__(cls, *args, **kwargs):
            return _Checkpoint()
        from_pretrained = staticmethod(lambda *args, **kwargs: _Checkpoint())

    with patch.object(family, 'CLIPModel', _CLIPModel), \
            patch.object(family, 'CLIPConfig', MagicMock()), \
            patch.object(family.CLIPProcessor, 'from_pretrained', return_value=_Processor()), \
            patch.object(family, 'pin_image_processor', side_effect=lambda p, *a, **k: p), \
            patch.object(registration, 'build_audio_wrapper', side_effect=lambda *a, **k: _audio()):
        model = registration.get_model(identifier)
    path = _stored_identifier(model.extractor['vision'], tmp_path, monkeypatch)
    assert path.startswith(f'{STORED}/identifier={identifier}-vision,stimuli_identifier=stimuli,')


def test_blip2_combination_keeps_its_legacy_cache_path(tmp_path, monkeypatch):
    import transformers
    from brainscore.models.multimodal_av_blip2_wav2vec2 import model as registration
    with patch('brainscore.models._downloads.hf_preflight', return_value={}), \
            patch.object(transformers.Blip2ForConditionalGeneration, 'from_pretrained',
                         return_value=_Checkpoint()), \
            patch.object(transformers.AutoProcessor, 'from_pretrained', return_value=_Processor()), \
            patch('brainscore.models.blip2_opt_2_7b.model.pin_image_processor',
                  side_effect=lambda p, *a, **k: p), \
            patch.object(registration, 'build_audio_wrapper', side_effect=lambda *a, **k: _audio()), \
            patch('brainscore.model_helpers._device.select_device', return_value=torch.device('cpu')):
        model = registration.get_model('blip2-wav2vec2')
    path = _stored_identifier(model.extractor['vision'], tmp_path, monkeypatch)
    assert path.startswith(f'{STORED}/identifier=blip2-wav2vec2-vision,stimuli_identifier=stimuli,')
