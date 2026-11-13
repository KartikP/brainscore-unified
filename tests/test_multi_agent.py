"""Multi-agent conversations through sessions, and activation capture during them.

No model weights: an EchoAgent (a subject with no network) for routing, and
BrainScoreModel subjects with tiny torch policies for recording. Verifies that
(a) one subject's answer on ``behavior:text`` becomes the next subject's
``text`` input, (b) each subject takes part through one session for the whole
conversation, and (c) session tools and recordings work on the run.
"""
import numpy as np
import pytest
import torch
from torch import nn

from brainscore import BrainScoreModel
from brainscore.activation_window import ActivationWindow
from brainscore.harnesses.multi_agent import (
    multi_agent_rollout, DialogueMediator, EchoAgent)
from brainscore.witness import Witness
from brainscore_core.streaming import StreamEvent


def _text_subject(identifier, policy, **kwargs):
    return BrainScoreModel(identifier, policy=policy, policy_in_channels=('text',),
                           policy_out_channels=('behavior:text',), **kwargs)


# ── routing: A's answer becomes B's input ────────────────────────────

def test_rollout_alternates_and_routes():
    a, b = EchoAgent('alice'), EchoAgent('bob')
    med = DialogueMediator(task='chat')
    transcript = multi_agent_rollout([a, b], med, n_turns=2)
    assert [e['subject'] for e in transcript] == ['alice', 'bob', 'alice', 'bob']
    assert len(med.history) == 4
    # bob's first answer echoes alice's first answer: the answer was routed
    assert f"heard<{transcript[0]['action']}>" in transcript[1]['action']


def test_rollout_accepts_dict_of_subjects_with_order():
    t = multi_agent_rollout({'a': EchoAgent('a'), 'b': EchoAgent('b')}, DialogueMediator(),
                            n_turns=1, order=['b', 'a'])
    assert [e['subject'] for e in t] == ['b', 'a']


def test_first_turn_hears_no_text_without_a_start():
    t = multi_agent_rollout([EchoAgent('solo')], DialogueMediator(), n_turns=1)
    assert t[0]['action'] == 'solo: heard<None>'


def test_answers_are_behavior_text_with_the_sender_and_the_next_subject_reads_them_as_text():
    med = DialogueMediator(task='greet', start='greet')
    transcript = multi_agent_rollout([EchoAgent('alice'), EchoAgent('bob')], med, n_turns=2)
    assert all(isinstance(m, StreamEvent) and m.channel == 'behavior:text' for m in med.history)
    assert [m.meta['sender'] for m in med.history] == ['alice', 'bob', 'alice', 'bob']
    assert transcript[0]['action'] == 'alice: heard<greet>'
    assert transcript[1]['action'] == 'bob: heard<alice: heard<greet>>'


def test_brainscoremodel_policy_takes_part_through_participate():
    agent = _text_subject('bsm', lambda row: {'behavior:text': f"ack<{row['text']}>"})
    med = DialogueMediator(task='hi', start='hi')
    transcript = multi_agent_rollout([agent], med, n_turns=1)
    assert transcript[0]['action'] == 'ack<hi>'
    assert med.history[0].meta['sender'] == 'bsm'


def test_one_session_for_the_whole_conversation():
    """begin comes once, so a policy that keeps state keeps it from turn to turn."""
    class Counter:
        def __init__(self):
            self.turns, self.resets = 0, 0

        def reset(self):
            self.resets += 1
            self.turns = 0

        def __call__(self, row):
            self.turns += 1
            return {'behavior:text': f'turn {self.turns}'}
    policy = Counter()
    transcript = multi_agent_rollout([_text_subject('counter', policy)], DialogueMediator(), n_turns=3)
    assert [e['action'] for e in transcript] == ['turn 1', 'turn 2', 'turn 3']
    assert policy.resets == 1


def test_a_subject_that_does_not_declare_the_answer_is_refused_before_any_turn():
    called = []
    agent = BrainScoreModel('motor-only', policy=lambda row: called.append(row) or {'motor': 0.0})
    with pytest.raises(ValueError, match='behavior:text'):
        multi_agent_rollout([EchoAgent('alice'), agent], DialogueMediator(), n_turns=1)
    assert called == []


def test_an_error_in_a_subject_stops_the_conversation():
    def broken(row):
        raise RuntimeError('no words')
    with pytest.raises(RuntimeError, match='no words'):
        multi_agent_rollout([EchoAgent('alice'), _text_subject('bob', broken)],
                            DialogueMediator(), n_turns=2)


def test_a_subject_whose_session_ends_without_an_answer_stops_the_conversation():
    class Silent(EchoAgent):
        def participate(self, session):
            return None
    with pytest.raises(RuntimeError, match='ended before its turn'):
        multi_agent_rollout([EchoAgent('alice'), Silent('bob')], DialogueMediator(), n_turns=1)


def test_an_answer_on_text_is_refused():
    """``text`` is an input; a subject answers on ``behavior:text``."""
    agent = BrainScoreModel('old-form', policy=lambda row: {'text': 'hi'}, policy_in_channels=('text',),
                            policy_out_channels=('text', 'behavior:text'))
    with pytest.raises(ValueError, match='not requested'):
        multi_agent_rollout([agent], DialogueMediator(), n_turns=1)


# ── session tools and recordings on a conversation ───────────────────

def test_a_witness_sees_each_turn_of_a_subject():
    alice, bob = EchoAgent('alice'), EchoAgent('bob')
    with Witness(bob) as witness:
        transcript = multi_agent_rollout([alice, bob], DialogueMediator(start='hello'), n_turns=2)
    assert len(witness.events) == 2                               # one per turn of bob
    for event, entry in zip(witness.events, [e for e in transcript if e['subject'] == 'bob']):
        assert event.mode == 'behavior'
        assert event.did['repr'] == entry['action']
    # what bob saw on its first turn is what alice said
    assert witness.events[0].saw[0]['text']['text'] == transcript[0]['action']


class TinyTorchPolicy:
    """A weightless stand-in for a language model: each turn runs one forward pass."""
    def __init__(self, dim=4):
        self.net = nn.Sequential(nn.Linear(dim, 6), nn.ReLU(), nn.Linear(6, 3))

    def __call__(self, row):
        out = self.net(torch.ones(1, 4))
        return {'behavior:text': str(out.detach().numpy().ravel().round(3).tolist())}


def test_activations_captured_for_both_subjects_during_rollout():
    pa, pb = TinyTorchPolicy(), TinyTorchPolicy()
    a, b = _text_subject('alice', pa), _text_subject('bob', pb)
    with ActivationWindow(pa.net, layers=['0']) as ra, \
         ActivationWindow(pb.net, layers=['0']) as rb:
        transcript = multi_agent_rollout([a, b], DialogueMediator(task='negotiate'), n_turns=3)
    assert len(transcript) == 6
    assert len(ra.captures) == 3 and len(rb.captures) == 3   # one per turn, per subject
    assert ra.stack('0').shape == (3, 1, 6)
    # time-aligned, same-shape activations from both subjects -> ready for inter-subject RSA
    assert rb.stack('0', reduce='mean').shape == (3, 6)


def test_a_region_is_recorded_during_each_answer():
    """``neural:<region>`` requested with the answer is recorded from the pass that made it."""
    def subject(identifier):
        policy = TinyTorchPolicy()
        return _text_subject(identifier, policy, model=policy.net, region_layer_map={'IT': '0'},
                             extractor={'text': lambda stimuli, layers: None})
    a, b = subject('alice'), subject('bob')
    transcript = multi_agent_rollout([a, b], DialogueMediator(start='hello'), n_turns=2,
                                     outputs=('behavior:text', 'neural:IT'))
    assert len(transcript) == 4
    for entry in transcript:
        recorded = entry['neural:IT']
        assert 'neuroid' in recorded.dims and recorded.sizes['neuroid'] == 6
        assert np.isfinite(np.asarray(recorded.values)).all()


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

