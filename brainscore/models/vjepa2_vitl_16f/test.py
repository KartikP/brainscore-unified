import numpy as np
import pytest
import torch


def test_the_variant_keeps_the_cortex_blocks_of_vjepa2_vitl():
    from brainscore.models.vjepa.model import CORTEX_LAYERS
    from brainscore.models.vjepa2_vitl_16f.model import NUM_INPUT_FRAMES, region_layers
    assert region_layers()['cortex'].layer_paths == CORTEX_LAYERS and NUM_INPUT_FRAMES == 16


def test_the_bf16_forward_keeps_the_module_names_and_runs_in_float32_off_cuda():
    from brainscore.models.vjepa2_vitl_16f.model import bf16_forward
    net = torch.nn.Sequential(torch.nn.Linear(4, 3))
    names = [n for n, _ in net.named_modules()]
    expected = net(torch.ones(1, 4))
    bf16_forward(net)
    assert [n for n, _ in net.named_modules()] == names
    assert torch.equal(net(torch.ones(1, 4)), expected)          # no CUDA here: float32, unchanged


@pytest.mark.slow
def test_it_answers_cortex_on_the_mario_benchmark():
    from brainscore import load_benchmark, load_model
    model = load_model('vjepa2-vitl-16f')
    assert 'vision:video' in model.in_channels and 'neural:cortex' in model.out_channels
    assert 'vision:video' in load_benchmark('Paugam2025-fMRI-mario-cortex').required_input_channels
