"""Multi-agent rollout + activation capture during interaction.

No model weights: an EchoAgent for routing, a tiny torch-backed agent for the
activation-capture path. Verifies (a) one agent's action becomes the next
agent's observation (the 'connection'), and (b) per-tick hidden activations are
collected from both agents during the shared interaction.
"""
import numpy as np
import torch
from torch import nn

from brainscore_core.model_interface import EnvironmentStep, EnvironmentResponse
from brainscore.harnesses.multi_agent import (
    multi_agent_rollout, DialogueMediator, EchoAgent)
from brainscore.activation_window import ActivationWindow


# ── routing: A's output becomes B's observation ──────────────────────

def test_rollout_alternates_and_routes():
    a, b = EchoAgent('alice'), EchoAgent('bob')
    med = DialogueMediator(task='chat')
    transcript = multi_agent_rollout([a, b], med, n_turns=2)
    assert [e['agent'] for e in transcript] == ['alice', 'bob', 'alice', 'bob']
    assert len(med.history) == 4
    # bob's first utterance must echo alice's first action -> the action was routed
    alice_first = transcript[0]['action']
    bob_first = transcript[1]['action']
    assert f'heard<{alice_first}>' in bob_first


def test_rollout_accepts_dict_of_agents_with_order():
    med = DialogueMediator()
    t = multi_agent_rollout({'a': EchoAgent('a'), 'b': EchoAgent('b')}, med,
                            n_turns=1, order=['b', 'a'])
    assert [e['agent'] for e in t] == ['b', 'a']


def test_first_tick_incoming_is_none():
    med = DialogueMediator()
    t = multi_agent_rollout([EchoAgent('solo')], med, n_turns=1)
    assert 'heard<None>' in t[0]['action']        # nothing came before the first act


# ── activation capture during the interaction (the headline) ─────────

class TinyTorchAgent:
    """A weightless stand-in for a real policy: process() runs a forward pass,
    so an ActivationWindow on its net captures one activation per tick."""
    def __init__(self, identifier, dim=4):
        self.identifier = identifier
        self.net = nn.Sequential(nn.Linear(dim, 6), nn.ReLU(), nn.Linear(6, 3))

    def process(self, step):
        x = torch.ones(1, 4)
        out = self.net(x)
        return EnvironmentResponse(action=out.detach().numpy().ravel().tolist())


def test_activations_captured_for_both_agents_during_rollout():
    a, b = TinyTorchAgent('alice'), TinyTorchAgent('bob')
    med = DialogueMediator(task='negotiate')
    with ActivationWindow(a.net, layers=['0']) as ra, \
         ActivationWindow(b.net, layers=['0']) as rb:
        transcript = multi_agent_rollout([a, b], med, n_turns=3)
    assert len(transcript) == 6
    assert len(ra.captures) == 3 and len(rb.captures) == 3   # one per tick, per agent
    acts_a = ra.stack('0')                                   # (n_calls, 1, 6)
    assert acts_a.shape == (3, 1, 6)
    # time-aligned, same-shape activations from both agents -> ready for inter-agent RSA
    acts_b = rb.stack('0', reduce='mean')
    assert acts_b.shape == (3, 6)


# ── ActivationWindow unit behavior ───────────────────────────────────

def _net():
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))


def test_captures_layer_output():
    net = _net()
    with ActivationWindow(net, layers=['0']) as rec:
        net(torch.ones(1, 4))
    assert len(rec.captures) == 1
    assert tuple(rec.captures[0].shape) == (1, 6)            # first Linear's output


def test_resolves_layer_by_dotted_path():
    net = _net()
    with ActivationWindow(net, layers=['2']) as rec:         # second Linear
        net(torch.ones(1, 4))
    assert tuple(rec.captures[0].shape) == (1, 3)


def test_stride_and_cap():
    net = _net()
    with ActivationWindow(net, layers=['0'], every=2) as rec:
        for _ in range(5):
            net(torch.ones(1, 4))
    assert len(rec.captures) == 3                            # calls 1,3,5
    with ActivationWindow(net, layers=['0'], max_captures=2) as rec2:
        for _ in range(5):
            net(torch.ones(1, 4))
    assert len(rec2.captures) == 2


def test_hooks_removed_on_exit():
    net = _net()
    with ActivationWindow(net, layers=['0']) as rec:
        net(torch.ones(1, 4))
    net(torch.ones(1, 4))                                    # after the block
    assert len(rec.captures) == 1


def test_multi_layer_by_layer_grouping():
    net = _net()
    with ActivationWindow(net, layers=['0', '2']) as rec:
        net(torch.ones(1, 4))
    bl = rec.by_layer()
    assert set(bl) == {'0', '2'}
    assert bl['0'][0].shape == (1, 6) and bl['2'][0].shape == (1, 3)


def test_unknown_layer_raises():
    import pytest
    with pytest.raises(ValueError, match='not found'):
        ActivationWindow(_net(), layers=['nope'])


def test_resolves_wrapper_dot_model():
    class Wrapper:
        def __init__(self, m):
            self._model = m
            self.identifier = 'w'
    net = _net()
    with ActivationWindow(Wrapper(net), layers=['0']) as rec:
        net(torch.ones(1, 4))
    assert len(rec.captures) == 1


def test_first_tensor_out_handles_tuple_and_hf_object():
    from brainscore.activation_window import _first_tensor_out
    t = torch.zeros(2, 3)
    assert _first_tensor_out((t, None)) is t

    class HF:
        last_hidden_state = torch.zeros(1, 5)
    assert _first_tensor_out(HF()).shape == (1, 5)
