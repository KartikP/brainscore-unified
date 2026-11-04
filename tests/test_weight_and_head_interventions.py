"""Interventions beyond activations: parameter edits and attention-head edits, undone exactly."""
import pytest
import torch

from brainscore_core.brainscore_model import BrainScoreModel
from brainscore_core.events import Perturbation, Selection, StateChange
from brainscore_core.streaming_helpers import apply_state_change
from brainscore.perturbation import build_pytorch_ablation_fn

pytestmark = pytest.mark.unit


class Attention(torch.nn.Module):
    def __init__(self, heads=4, head_dim=2):
        super().__init__()
        self.head_dim = head_dim
        self.o_proj = torch.nn.Linear(heads * head_dim, 3)

    def forward(self, per_head):            # (batch, heads * head_dim), as after attention
        return self.o_proj(per_head)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(3, 8)
        self.attn = Attention()


def _subject(net):
    return BrainScoreModel('net', model=net, supported_modalities={'vision'},
                           intervention=build_pytorch_ablation_fn(net))


def test_a_weight_edit_changes_outputs_and_undo_restores_it_exactly():
    net = Net()
    subject = _subject(net)
    before = net.proj.weight.detach().clone()
    x = torch.ones(1, 3)
    applied = apply_state_change(subject, StateChange(
        kind='weights', target=Selection(layer='proj.weight', indices=[0, 2], axis=-2),
        perturbation=Perturbation(kind='zero')))
    assert torch.all(net.proj.weight[[0, 2]] == 0) and torch.equal(net.proj.weight[1], before[1])
    assert torch.all(net.proj(x)[0, [0, 2]] == net.proj.bias[[0, 2]])
    apply_state_change(subject, StateChange(kind='reset', handle_id=applied.handle_id))
    assert torch.equal(net.proj.weight, before)                     # bit for bit


def test_scaling_a_whole_parameter_and_reset():
    net = Net()
    subject = _subject(net)
    before = net.attn.o_proj.weight.detach().clone()
    apply_state_change(subject, StateChange(kind='weights', target=Selection(layer='attn.o_proj.weight'),
                                            perturbation=Perturbation(kind='scale', scale=0.5)))
    assert torch.allclose(net.attn.o_proj.weight, before * 0.5)
    subject.reset()
    assert torch.equal(net.attn.o_proj.weight, before)


def test_silencing_attention_heads():
    net = Net()
    subject = _subject(net)
    per_head = torch.arange(8.0).reshape(1, 8)
    expected_input = per_head.clone()
    expected_input[:, 2:4] = 0                                       # head 1 of 4, head_dim 2
    applied = apply_state_change(subject, StateChange(
        kind='attention_heads', target=Selection(layer='attn', indices=[1]),
        perturbation=Perturbation(kind='zero')))
    assert torch.allclose(net.attn(per_head), net.attn.o_proj.weight @ expected_input[0] + net.attn.o_proj.bias)
    apply_state_change(subject, StateChange(kind='reset', handle_id=applied.handle_id))
    assert not net.attn.o_proj._forward_pre_hooks


def test_editing_something_that_is_not_a_parameter_fails_before_any_change():
    net = Net()
    with pytest.raises(ValueError, match='not a parameter'):
        apply_state_change(_subject(net), StateChange(kind='weights', target=Selection(layer='proj'),
                                                      perturbation=Perturbation(kind='zero')))
