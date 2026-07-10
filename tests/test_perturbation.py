"""Tests for build_pytorch_ablation_fn — exercises the full state_change
dispatch on a real torch.nn.Module via forward-hook installation/removal.
"""

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from brainscore_core.model_interface import (
    BrainScoreModel,
    Perturbation,
    PerturbationApplied,
    Selection,
    StateChange,
)
from brainscore.perturbation import build_pytorch_ablation_fn


class _TwoLayerNet(torch.nn.Module):
    """Tiny model: linear → relu → linear. Outputs 8-dim feature vector."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8, bias=False)
        self.fc2 = torch.nn.Linear(8, 8, bias=False)
        # Set weights to identity-ish so we can predict outputs by hand
        with torch.no_grad():
            self.fc1.weight.copy_(torch.eye(8, 4))
            self.fc2.weight.copy_(torch.eye(8))

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def model_with_ablation():
    net = _TwoLayerNet()
    bs = BrainScoreModel(
        identifier='two-layer-test',
        model=net,
        region_layer_map={},
        preprocessors={'vision': lambda x: x},
        state_change_fn=build_pytorch_ablation_fn(net),
    )
    return bs, net


# ── Whole-layer ablation ─────────────────────────────────────────────

class TestWholeLayerAblation:

    def test_baseline_then_zero_ablation(self, model_with_ablation):
        bs, net = model_with_ablation
        x = torch.tensor([[1., 2., 3., 4.]])

        # Baseline: input passes through fc1 → relu → fc2
        # With identity weights: fc1(x) = [1,2,3,4,0,0,0,0]
        # relu(...) is unchanged (all >=0)
        # fc2(...) is identity → output = [1,2,3,4,0,0,0,0]
        baseline = net(x).detach().clone()
        assert torch.allclose(baseline, torch.tensor([[1., 2., 3., 4., 0., 0., 0., 0.]]))

        # Apply ablation: zero out fc1's output entirely
        applied = bs.process(StateChange(
            kind='ablation',
            target=Selection(layer='fc1'),
            perturbation=Perturbation(kind='zero'),
        ))
        assert isinstance(applied, PerturbationApplied)

        # Forward pass now produces all zeros (fc1 output is zeroed → relu(0)=0
        # → fc2(0)=0)
        lesioned = net(x).detach()
        assert torch.allclose(lesioned, torch.zeros(1, 8))

        # Reset removes the hook
        bs.reset()
        restored = net(x).detach()
        assert torch.allclose(restored, baseline)

    def test_scale_ablation(self, model_with_ablation):
        bs, net = model_with_ablation
        x = torch.tensor([[1., 2., 3., 4.]])
        baseline = net(x).detach().clone()

        bs.process(StateChange(
            kind='ablation',
            target=Selection(layer='fc1'),
            perturbation=Perturbation(kind='scale', scale=0.5),
        ))
        scaled = net(x).detach()
        assert torch.allclose(scaled, baseline * 0.5)

        bs.reset()


# ── Indexed ablation ─────────────────────────────────────────────────

class TestIndexedAblation:

    def test_zero_only_specific_units(self, model_with_ablation):
        bs, net = model_with_ablation
        x = torch.tensor([[1., 2., 3., 4.]])

        # Ablate fc1 outputs at indices 0 and 2 only
        bs.process(StateChange(
            kind='ablation',
            target=Selection(layer='fc1', indices=[0, 2]),
            perturbation=Perturbation(kind='zero'),
        ))

        out = net(x).detach()
        # fc1 output was [1,2,3,4,0,0,0,0]; after zeroing 0,2 -> [0,2,0,4,0,0,0,0]
        # relu unchanged (all >=0). fc2 identity -> [0,2,0,4,0,0,0,0]
        assert torch.allclose(out, torch.tensor([[0., 2., 0., 4., 0., 0., 0., 0.]]))

        bs.reset()
        # baseline restored
        out = net(x).detach()
        assert torch.allclose(out, torch.tensor([[1., 2., 3., 4., 0., 0., 0., 0.]]))

    def test_scale_specific_units(self, model_with_ablation):
        bs, net = model_with_ablation
        x = torch.tensor([[2., 4., 6., 8.]])

        bs.process(StateChange(
            kind='ablation',
            target=Selection(layer='fc1', indices=[1, 3]),
            perturbation=Perturbation(kind='scale', scale=0.25),
        ))

        out = net(x).detach()
        # fc1 output: [2,4,6,8,0,0,0,0]; after scaling 1,3 by 0.25:
        #             [2,1,6,2,0,0,0,0]
        assert torch.allclose(out, torch.tensor([[2., 1., 6., 2., 0., 0., 0., 0.]]))

        bs.reset()


# ── Multiple concurrent ablations ────────────────────────────────────

class TestMultipleConcurrentAblations:

    def test_two_layers_perturbed_simultaneously(self, model_with_ablation):
        bs, net = model_with_ablation
        x = torch.tensor([[1., 2., 3., 4.]])

        # Ablate fc1 unit 0 AND fc2 unit 7
        a = bs.process(StateChange(
            kind='ablation',
            target=Selection(layer='fc1', indices=[0]),
            perturbation=Perturbation(kind='zero'),
        ))
        b = bs.process(StateChange(
            kind='ablation',
            target=Selection(layer='fc2', indices=[7]),
            perturbation=Perturbation(kind='zero'),
        ))
        assert a.handle_id != b.handle_id
        assert len(bs._active_perturbations) == 2

        # Selectively undo just fc1 ablation; fc2 ablation remains
        bs.process(StateChange(kind='reset', handle_id=a.handle_id))
        assert len(bs._active_perturbations) == 1
        assert b.handle_id in bs._active_perturbations

        bs.reset()
        assert len(bs._active_perturbations) == 0


# ── Layer-resolution edge cases ──────────────────────────────────────

class TestLayerResolution:

    def test_unknown_layer_raises(self, model_with_ablation):
        bs, _ = model_with_ablation
        with pytest.raises(AttributeError, match='Cannot resolve layer path'):
            bs.process(StateChange(
                kind='ablation',
                target=Selection(layer='nonexistent.layer'),
                perturbation=Perturbation(kind='zero'),
            ))

    def test_indexed_layer_path(self):
        # If the model contains a Sequential, dotted-int paths must resolve
        net = torch.nn.Sequential(
            torch.nn.Linear(4, 4, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 4, bias=False),
        )
        bs = BrainScoreModel(
            identifier='seq-test',
            model=net,
            region_layer_map={},
            preprocessors={'vision': lambda x: x},
            state_change_fn=build_pytorch_ablation_fn(net),
        )
        # Sequential's children are accessed by integer index
        applied = bs.process(StateChange(
            kind='ablation',
            target=Selection(layer='0'),
            perturbation=Perturbation(kind='zero'),
        ))
        assert isinstance(applied, PerturbationApplied)
        bs.reset()


# ── Drive (stimulation) ──────────────────────────────────────────────

class TestDrive:
    """The 'drive' kind adds `amount` to the selected units — a stimulation /
    excitation analog, the additive counterpart of ablation."""

    def test_whole_layer_drive(self, model_with_ablation):
        bs, net = model_with_ablation
        x = torch.tensor([[1., 2., 3., 4.]])
        baseline = net(x).detach().clone()

        bs.process(StateChange(
            kind='ablation',
            target=Selection(layer='fc1'),
            perturbation=Perturbation(kind='drive', amount=2.0),
        ))
        # every fc1 unit is driven up by 2 (relu keeps them, fc2 is identity)
        driven = net(x).detach()
        assert torch.allclose(driven, baseline + 2.0)

        bs.reset()
        assert torch.allclose(net(x).detach(), baseline)

    def test_drive_specific_units(self, model_with_ablation):
        bs, net = model_with_ablation
        x = torch.tensor([[1., 2., 3., 4.]])
        baseline = net(x).detach().clone()

        # units 4 and 5 are 0 at baseline; drive them to +3, leave the rest
        bs.process(StateChange(
            kind='ablation',
            target=Selection(layer='fc1', indices=[4, 5]),
            perturbation=Perturbation(kind='drive', amount=3.0),
        ))
        out = net(x).detach()
        assert torch.allclose(out, torch.tensor([[1., 2., 3., 4., 3., 3., 0., 0.]]))

        bs.reset()
        assert torch.allclose(net(x).detach(), baseline)


# ── Index validation (fail at apply, not mid-forward) ────────────────

class TestIndexValidation:

    def test_out_of_range_index_raises_at_apply(self, model_with_ablation):
        bs, net = model_with_ablation  # fc1 has 8 units
        with pytest.raises(ValueError, match="out of range"):
            bs.process(StateChange(
                kind='ablation',
                target=Selection(layer='fc1', indices=[99]),
                perturbation=Perturbation(kind='zero'),
            ))

    def test_negative_index_raises_at_apply(self, model_with_ablation):
        bs, net = model_with_ablation
        with pytest.raises(ValueError, match="non-negative"):
            bs.process(StateChange(
                kind='ablation',
                target=Selection(layer='fc1', indices=[-1]),
                perturbation=Perturbation(kind='zero'),
            ))


def test_perturbation_positional_api_preserved():
    # 'amount' (added for kind='drive') must not shift the (kind, scale,
    # replacement) positional order of the public dataclass.
    p = Perturbation('replace', 0.25, 'payload')
    assert (p.kind, p.scale, p.replacement, p.amount) == ('replace', 0.25, 'payload', 0.0)
    assert Perturbation(kind='drive', amount=8.0).amount == 8.0
