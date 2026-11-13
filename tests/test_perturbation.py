"""Tests for build_pytorch_ablation_fn: interventions sent with
``apply_state_change`` on a real torch.nn.Module, through forward-hook
installation and removal.
"""

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from brainscore import BrainScoreModel
from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
from brainscore.perturbation import build_pytorch_ablation_fn
from brainscore_core.streaming_helpers import apply_state_change


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
        extractor={'vision': lambda stimuli, layers=None: stimuli},
        intervention=build_pytorch_ablation_fn(net),
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
        applied = apply_state_change(bs, StateChange(
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

        apply_state_change(bs, StateChange(
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
        apply_state_change(bs, StateChange(
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

        apply_state_change(bs, StateChange(
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
        a = apply_state_change(bs, StateChange(
            kind='ablation',
            target=Selection(layer='fc1', indices=[0]),
            perturbation=Perturbation(kind='zero'),
        ))
        b = apply_state_change(bs, StateChange(
            kind='ablation',
            target=Selection(layer='fc2', indices=[7]),
            perturbation=Perturbation(kind='zero'),
        ))
        assert a.handle_id != b.handle_id
        assert len(bs._interventions.active_interventions) == 2
        # Both act: fc1 unit 0 and fc2 unit 7 are zero (unit 7 is zero anyway).
        assert torch.equal(net(x).detach(), torch.tensor([[0., 2., 3., 4., 0., 0., 0., 0.]]))

        # Selectively undo just fc1 ablation; fc2 ablation remains
        apply_state_change(bs, StateChange(kind='reset', handle_id=a.handle_id))
        assert len(bs._interventions.active_interventions) == 1
        assert b.handle_id in bs._interventions.active_interventions
        assert torch.equal(net(x).detach(), torch.tensor([[1., 2., 3., 4., 0., 0., 0., 0.]]))

        bs.reset()
        assert len(bs._interventions.active_interventions) == 0


class TestUndoIsExact:

    def test_undo_by_handle_restores_the_same_outputs(self, model_with_ablation):
        bs, net = model_with_ablation
        x = torch.randn(3, 4)
        baseline = net(x).detach().clone()
        applied = apply_state_change(bs, StateChange(
            kind='ablation', target=Selection(layer='fc1', indices=[1, 3]),
            perturbation=Perturbation(kind='scale', scale=0.3)))
        assert not torch.equal(net(x).detach(), baseline)
        apply_state_change(bs, StateChange(kind='reset', handle_id=applied.handle_id))
        assert torch.equal(net(x).detach(), baseline)
        assert bs._interventions.active_interventions == {}

    def test_two_interventions_stack_and_undo_selectively(self, model_with_ablation):
        bs, net = model_with_ablation
        x = torch.tensor([[1., 2., 3., 4.]])
        scale = apply_state_change(bs, StateChange(
            kind='ablation', target=Selection(layer='fc1'),
            perturbation=Perturbation(kind='scale', scale=0.5)))
        drive = apply_state_change(bs, StateChange(
            kind='stimulation', target=Selection(layer='fc2', indices=[5]),
            perturbation=Perturbation(kind='drive', amount=1.0)))
        both = torch.tensor([[0.5, 1., 1.5, 2., 0., 1., 0., 0.]])
        assert torch.equal(net(x).detach(), both)
        apply_state_change(bs, StateChange(kind='reset', handle_id=scale.handle_id))
        assert torch.equal(net(x).detach(), torch.tensor([[1., 2., 3., 4., 0., 1., 0., 0.]]))
        apply_state_change(bs, StateChange(kind='reset', handle_id=drive.handle_id))
        assert torch.equal(net(x).detach(), torch.tensor([[1., 2., 3., 4., 0., 0., 0., 0.]]))


# ── Layer-resolution edge cases ──────────────────────────────────────

class TestLayerResolution:

    def test_unknown_layer_raises(self, model_with_ablation):
        bs, _ = model_with_ablation
        with pytest.raises(AttributeError, match='Cannot resolve layer path'):
            apply_state_change(bs, StateChange(
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
            extractor={'vision': lambda stimuli, layers=None: stimuli},
            intervention=build_pytorch_ablation_fn(net),
        )
        # Sequential's children are accessed by integer index
        applied = apply_state_change(bs, StateChange(
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

        apply_state_change(bs, StateChange(
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
        apply_state_change(bs, StateChange(
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
            apply_state_change(bs, StateChange(
                kind='ablation',
                target=Selection(layer='fc1', indices=[99]),
                perturbation=Perturbation(kind='zero'),
            ))

    def test_negative_index_raises_at_apply(self, model_with_ablation):
        bs, net = model_with_ablation
        with pytest.raises(ValueError, match="non-negative"):
            apply_state_change(bs, StateChange(
                kind='ablation',
                target=Selection(layer='fc1', indices=[-1]),
                perturbation=Perturbation(kind='zero'),
            ))

    def test_numpy_int_index_accepted(self, model_with_ablation):
        import numpy as np
        bs, net = model_with_ablation  # fc1 has 8 units
        apply_state_change(bs, StateChange(  # np.int64 within range must NOT be rejected
            kind='ablation',
            target=Selection(layer='fc1', indices=[np.int64(2)]),
            perturbation=Perturbation(kind='zero'),
        ))
        bs.reset()

    def test_conv_index_targets_last_axis_not_channels(self):
        # Index perturbation acts on axis -1 (spatial for conv), so index 5 is
        # VALID even though out_channels=4; it must APPLY (not be rejected) and
        # zero spatial column 5. This would fail against c5879b1.
        from brainscore_core.events import PerturbationApplied
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3))  # output (N, 4, H-2, W-2)
        bs = BrainScoreModel('conv', net, {}, extractor={'vision': lambda stimuli, layers=None: stimuli},
                             intervention=build_pytorch_ablation_fn(net))
        applied = apply_state_change(bs, StateChange(
            kind='ablation',
            target=Selection(layer='0', indices=[5]),  # > out_channels, < width
            perturbation=Perturbation(kind='zero'),
        ))
        assert isinstance(applied, PerturbationApplied)  # applied, not rejected
        out = net(torch.ones(1, 3, 10, 10))              # output width 8
        assert torch.allclose(out[..., 5], torch.zeros_like(out[..., 5]))  # spatial col 5 zeroed
        assert not torch.allclose(out[..., 0], torch.zeros_like(out[..., 0]))  # col 0 untouched
        bs.reset()


def test_perturbation_positional_api_preserved():
    # 'amount' (added for kind='drive') must not shift the (kind, scale,
    # replacement) positional order of the public dataclass.
    p = Perturbation('replace', 0.25, 'payload')
    assert (p.kind, p.scale, p.replacement, p.amount) == ('replace', 0.25, 'payload', 0.0)
    assert Perturbation(kind='drive', amount=8.0).amount == 8.0


# ── Input site (forward pre-hook) ────────────────────────────────────

class _GatedMLP(torch.nn.Module):
    """SwiGLU MLP shaped like Qwen/LLaMA: down(silu(gate(x)) * up(x)).
    The down_proj input is computed inline, so only a pre-hook reaches it."""

    def __init__(self, d=6, h=10):
        super().__init__()
        torch.manual_seed(0)
        self.gate_proj = torch.nn.Linear(d, h, bias=False)
        self.up_proj = torch.nn.Linear(d, h, bias=False)
        self.down_proj = torch.nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TestInputSite:

    def _bs(self, net):
        return BrainScoreModel(identifier='gated-mlp-test', model=net, region_layer_map={},
                               intervention=build_pytorch_ablation_fn(net))

    def _steer(self, bs, layer, perturbation, indices=None):
        return apply_state_change(bs, StateChange(kind='steer', perturbation=perturbation,
                                      target=Selection(layer=layer, indices=indices)))

    def test_drive_down_proj_input_adds_weight_column(self):
        net = _GatedMLP()
        bs = self._bs(net)
        x = torch.randn(2, 5, 6)  # (batch, tokens, d)
        baseline = net(x).detach().clone()

        n, delta = 7, 2.5
        self._steer(bs, 'down_proj:input', Perturbation(kind='drive', amount=delta), [n])
        # every token's output moves by exactly delta * W_down[:, n]
        expected = baseline + delta * net.down_proj.weight[:, n].detach()
        assert torch.allclose(net(x).detach(), expected, atol=1e-6)

        bs.reset()
        assert torch.equal(net(x).detach(), baseline)

    def test_whole_layer_vector_drive(self):
        net = _GatedMLP()
        bs = self._bs(net)
        x = torch.randn(2, 5, 6)
        baseline = net(x).detach().clone()
        direction = torch.arange(6, dtype=torch.float64)  # dtype is matched to the layer
        self._steer(bs, 'down_proj', Perturbation(kind='drive', amount=direction))
        assert torch.allclose(net(x).detach(), baseline + direction.float(), atol=1e-6)

    def test_input_site_validates_against_in_features(self):
        net = _GatedMLP()
        with pytest.raises(ValueError, match='out of range'):
            self._steer(self._bs(net), 'down_proj:input', Perturbation(kind='drive', amount=1.0), [10])

    def test_unknown_site_rejected(self):
        net = _GatedMLP()
        with pytest.raises(ValueError, match='site'):
            self._steer(self._bs(net), 'down_proj:middle', Perturbation(kind='drive', amount=1.0))


# ── Array drive amounts never grow the activity ──────────────────────

def _drive(net, amount, layer='0', indices=None):
    return build_pytorch_ablation_fn(net)(StateChange(
        kind='steer', target=Selection(layer=layer, indices=indices),
        perturbation=Perturbation(kind='drive', amount=amount)))


class TestArrayDrive:

    @pytest.mark.parametrize('amount', [torch.arange(3.), torch.arange(3.)[None],
                                        np.arange(3.), torch.tensor(2.), np.float32(2.)])
    def test_broadcastable_amounts_add_in_place(self, amount):
        net = torch.nn.Sequential(torch.nn.Identity())
        x = torch.ones(2, 4, 3)
        _, cleanup = _drive(net, amount)
        try:
            out = net(x)
            assert out.shape == x.shape and out.dtype == x.dtype
            assert torch.equal(out, x + torch.as_tensor(np.asarray(amount), dtype=x.dtype))
        finally:
            cleanup()
        assert torch.equal(net(x), x)

    @pytest.mark.parametrize('shape', [(2, 3), (4,), (1, 1, 3), (2, 1, 3)])
    def test_amounts_that_would_change_the_shape_are_rejected(self, shape):
        net = torch.nn.Sequential(torch.nn.Identity())
        _, cleanup = _drive(net, torch.ones(shape))
        try:
            with pytest.raises(ValueError, match='does not broadcast'):
                net(torch.ones(1, 3))
        finally:
            cleanup()

    def test_numpy_float64_follows_the_activity_dtype(self):
        net = torch.nn.Sequential(torch.nn.Identity())
        direction = np.array([0.1, 0.2, 0.3])
        for dtype in (torch.float32, torch.bfloat16):
            _, cleanup = _drive(net, direction)
            x = torch.ones(2, 3, dtype=dtype)
            out = net(x)
            assert out.dtype == dtype
            assert torch.equal(out, x + torch.from_numpy(direction).to(dtype))
            cleanup()

    def test_caller_edits_after_apply_do_not_change_a_numpy_amount(self):
        net = torch.nn.Sequential(torch.nn.Identity())
        direction = np.ones(3)
        _, cleanup = _drive(net, direction)
        direction[:] = 100.
        assert torch.equal(net(torch.zeros(1, 3)), torch.ones(1, 3))
        cleanup()

    def test_gradients_flow_to_activity_and_to_a_tensor_amount(self):
        net = torch.nn.Sequential(torch.nn.Identity())
        x = torch.ones(2, 3, requires_grad=True)
        _, cleanup = _drive(net, np.arange(3.))
        net(x).sum().backward()
        assert torch.equal(x.grad, torch.ones(2, 3))
        cleanup()
        amount = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        _, cleanup = _drive(net, amount)
        net(torch.ones(2, 3)).sum().backward()
        assert torch.equal(amount.grad, torch.full((3,), 2., dtype=torch.float64))
        cleanup()

    def test_indexed_drive_takes_only_a_scalar(self):
        net = torch.nn.Sequential(torch.nn.Identity())
        with pytest.raises(ValueError, match='scalar amount'):
            _drive(net, np.ones(2), indices=[0, 1])
        assert not net[0]._forward_hooks
        _, cleanup = _drive(net, np.array(2.), indices=[1])  # 0-d is a scalar
        assert torch.equal(net(torch.zeros(1, 3)), torch.tensor([[0., 2., 0.]]))
        cleanup()

    def test_other_amount_types_are_rejected(self):
        with pytest.raises(TypeError, match='drive amount'):
            _drive(torch.nn.Sequential(torch.nn.Identity()), [1., 2., 3.])

    def test_python_scalars_keep_their_arithmetic(self):
        net = torch.nn.Sequential(torch.nn.Identity())
        applied, cleanup = _drive(net, 2)
        x = torch.arange(3)  # an integer activity stays integer with an int amount
        assert torch.equal(net(x[None]), x[None] + 2) and applied.perturbation.amount == 2
        cleanup()


# ── Strict site parser ───────────────────────────────────────────────

from brainscore.model_helpers.selection import split_site


@pytest.mark.parametrize('layer, expected', [
    ('fc1', ('fc1', 'output')), ('model.layers.14.mlp.down_proj', ('model.layers.14.mlp.down_proj', 'output')),
    ('model.layers.14.mlp.down_proj:input', ('model.layers.14.mlp.down_proj', 'input')),
    ('0', ('0', 'output')), ('0:input', ('0', 'input')),
])
def test_split_site_accepts_paths_and_one_input_suffix(layer, expected):
    assert split_site(layer) == expected


@pytest.mark.parametrize('layer', ['', ':input', ':', 'fc1:', 'fc1:output', 'fc1:middle:input',
                                   'fc1:input:input', 'fc1::input', 'fc1:Input'])
def test_split_site_rejects_malformed_addresses(layer):
    with pytest.raises(ValueError):
        split_site(layer)


@pytest.mark.parametrize('layer', [':input', 'fc1:middle:input', 'fc1:input:input'])
def test_perturbation_rejects_malformed_sites_before_hooking(layer):
    net = _TwoLayerNet()
    with pytest.raises(ValueError):
        _drive(net, 1.0, layer=layer)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in net.modules())


def test_index_out_of_range_for_the_selected_axis_fails_on_every_device():
    """Flattened-unit indices on a per-axis selection must raise, not be skipped.

    MPS ignores out-of-range indices silently, so the hook checks bounds itself.
    """
    conv = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1))
    apply_fn = build_pytorch_ablation_fn(conv)
    apply_fn(StateChange(kind='ablation', target=Selection(layer='0', indices=[0, 7], axis=-1),
                         perturbation=Perturbation(kind='zero')))
    with pytest.raises(ValueError, match='out of range for axis'):
        conv(torch.ones(1, 1, 3, 3))
