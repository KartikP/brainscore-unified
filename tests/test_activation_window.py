"""Direct unit tests for ActivationWindow.

The coverage audit flagged ActivationWindow as tested only *indirectly* via the
multi-agent tests. These exercise it directly on a tiny CPU model: hook capture
during a loop, per-call stacking, the `every` stride and `max_captures` cap,
tuple/HF-style output unwrapping, reduce='mean' for ragged sequence lengths,
invalid-layer errors, hook teardown on exit, and save().
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from brainscore.activation_window import ActivationWindow, _first_tensor_out


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class _TupleBlock(nn.Module):
    """outermost forward returns a tuple (tensor, aux) like a transformer block."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x), {'aux': 1}


@pytest.mark.unit
class TestCapture:
    def test_captures_one_per_forward(self):
        m = _Stack()
        with ActivationWindow(m, layers=['blocks.1']) as rec:
            for _ in range(3):
                m(torch.randn(2, 4))
        arrs = rec.by_layer()['blocks.1']
        assert len(arrs) == 3 and arrs[0].shape == (2, 4)

    def test_stack_shape(self):
        m = _Stack()
        with ActivationWindow(m, layers=['blocks.0']) as rec:
            for _ in range(4):
                m(torch.randn(2, 4))
        assert rec.stack('blocks.0').shape == (4, 2, 4)

    def test_every_stride(self):
        m = _Stack()
        with ActivationWindow(m, layers=['blocks.2'], every=2) as rec:
            for _ in range(6):
                m(torch.randn(1, 4))
        assert len(rec.by_layer()['blocks.2']) == 3       # calls 0,2,4

    def test_max_captures_cap(self):
        m = _Stack()
        with ActivationWindow(m, layers=['blocks.0'], max_captures=2) as rec:
            for _ in range(5):
                m(torch.randn(1, 4))
        assert len(rec.captures) == 2

    def test_hooks_removed_on_exit(self):
        m = _Stack()
        with ActivationWindow(m, layers=['blocks.1']):
            pass
        assert len(m.blocks[1]._forward_hooks) == 0


@pytest.mark.unit
class TestOutputHandling:
    def test_first_tensor_out_tuple(self):
        t = torch.randn(2, 4)
        assert _first_tensor_out((t, None)) is t

    def test_first_tensor_out_hf_object(self):
        class O:
            last_hidden_state = torch.randn(1, 3, 4)
        assert _first_tensor_out(O()).shape == (1, 3, 4)

    def test_captures_from_tuple_returning_module(self):
        m = _TupleBlock()
        with ActivationWindow(m, layers=['lin']) as rec:
            m(torch.randn(2, 4))
        assert rec.by_layer()['lin'][0].shape == (2, 4)


@pytest.mark.unit
class TestReduceAndErrors:
    def test_reduce_mean_handles_ragged(self):
        m = _Stack()
        with ActivationWindow(m, layers=['blocks.0']) as rec:
            m(torch.randn(1, 3, 4))      # seq len 3
            m(torch.randn(1, 5, 4))      # seq len 5 -> ragged
        # without reduce: heterogeneous shapes raise
        with pytest.raises(ValueError):
            rec.stack('blocks.0')
        # reduce='mean' pools each call to (4,) -> (2, 4)
        assert rec.stack('blocks.0', reduce='mean').shape == (2, 4)

    def test_invalid_layer_raises(self):
        with pytest.raises(ValueError):
            ActivationWindow(_Stack(), layers=['blocks.99'])

    def test_no_layers_hooks_root(self):
        m = _Stack()
        with ActivationWindow(m) as rec:
            m(torch.randn(2, 4))
        assert len(rec.captures) == 1     # the root module's output


@pytest.mark.unit
class TestAccessors:
    def test_summary(self):
        m = _Stack()
        with ActivationWindow(m, layers=['blocks.0', 'blocks.2']) as rec:
            for _ in range(2):
                m(torch.randn(1, 4))
        s = rec.summary()
        assert s['n_captures'] == 4 and s['by_layer'] == {'blocks.0': 2, 'blocks.2': 2}

    def test_save_writes_manifest(self, tmp_path):
        m = _Stack()
        with ActivationWindow(m, layers=['blocks.0']) as rec:
            for _ in range(3):
                m(torch.randn(1, 4))
        man = rec.save(str(tmp_path))
        assert (tmp_path / 'manifest.json').exists()
        assert 'blocks.0' in man['layers']
