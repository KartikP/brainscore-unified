import pytest

pytestmark = pytest.mark.unit


def test_select_device_honours_an_explicit_device():
    import torch
    from brainscore.model_helpers._device import select_device
    assert select_device('cpu') == torch.device('cpu')
    assert select_device() in {torch.device('cpu'), torch.device('cuda'), torch.device('mps')}
