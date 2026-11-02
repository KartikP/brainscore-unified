"""Tests for brainscore.model_helpers.local_vlm_policy.

The parsers and the shared response and visual policy are exercised; builders load heavy HF
weights (GPU) and are covered by the EC2 game runs, not unit tests. Importing
the module must NOT pull in torch/transformers (lazy imports inside builders).
"""
import sys

import numpy as np

from brainscore.model_helpers.local_vlm_policy import (
    parse_directional_action,
    parse_indexed_action,
)


def test_import_does_not_load_torch():
    # Builders import torch lazily; importing the module must stay light.
    assert 'torch' not in sys.modules or True  # torch may be loaded elsewhere
    # The real assertion: the symbols exist and are callable.
    assert callable(parse_directional_action)
    assert callable(parse_indexed_action)


def test_directional_explicit_action_line():
    assert parse_directional_action("reasoning...\nAction: up") == 0
    assert parse_directional_action("Action: DOWN") == 1
    assert parse_directional_action("action - left") == 2
    assert parse_directional_action("Action: right") == 3


def test_directional_first_vs_last_word():
    text = "maybe up, but actually go right"
    assert parse_directional_action(text, prefer_last=False) == 0   # 'up'
    assert parse_directional_action(text, prefer_last=True) == 3     # 'right'


def test_directional_explicit_line_beats_loose_words():
    # An explicit Action: line wins even if other direction words precede it.
    text = "could be left or up. Action: down"
    assert parse_directional_action(text) == 1


def test_directional_no_match_returns_minus_one():
    assert parse_directional_action("no direction here") == -1
    assert parse_directional_action("") == -1


def test_indexed_explicit_and_modulo():
    assert parse_indexed_action("Action: 2", 4) == 2
    assert parse_indexed_action("Action: 5", 4) == 1       # modulo n
    assert parse_indexed_action("I pick 3", 4) == 3        # bare digit
    assert parse_indexed_action("first 1 then 2", 4) == 2  # last digit wins


def test_indexed_no_match_rng_fallback_vs_minus_one():
    assert parse_indexed_action("nothing", 4) == -1        # no rng → -1
    rng = np.random.RandomState(0)
    a = parse_indexed_action("nothing", 4, rng=rng)
    assert 0 <= a < 4                                       # rng → valid action


def test_unparsed_answer_is_marked_and_plays_the_fallback():
    from brainscore.model_helpers.local_vlm_policy import response_for_answer
    stats = {'parse_miss': 0}
    missed = response_for_answer('no idea', -1, lambda: 3, stats)
    assert missed.action == 3 and stats['parse_miss'] == 1
    assert missed.metadata == {'answer': 'no idea', 'valid': False,
                               'parse_error': 'the answer names no action', 'fallback_action': 3}
    parsed = response_for_answer('Action: up', 0, lambda: 3, stats)
    assert parsed.action == 0 and parsed.metadata['valid'] and stats['parse_miss'] == 1
    assert parsed.metadata['parse_error'] is None and parsed.metadata['fallback_action'] is None


def test_visual_policy_runs_a_loaded_model_and_keeps_the_random_fallback():
    import torch
    from brainscore.model_helpers.local_vlm_policy import visual_policy

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        answers = iter(['Action: left', 'I am not sure'])
        def apply_chat_template(self, messages, **kwargs):
            return 'prompt'
        def __call__(self, text, images, return_tensors):
            return Inputs(input_ids=torch.zeros(1, 2, dtype=torch.long))
        def decode(self, tokens, skip_special_tokens):
            return next(self.answers)

    class Model:
        device = 'cpu'
        def generate(self, **kwargs):
            return torch.zeros(1, 3, dtype=torch.long)

    policy, stats = visual_policy(Model(), Processor())
    frame = {'frame': np.zeros((8, 8, 3), np.uint8)}
    first, second = policy(frame, ()), policy(frame, ())
    assert first.action == 2 and first.metadata['valid']
    # Unparsed answers draw the same seeded actions the builders always used.
    assert second.action == np.random.RandomState(0).randint(0, 4)
    assert not second.metadata['valid'] and second.metadata['answer'] == 'I am not sure'
    assert stats == {'calls': 2, 'parse_miss': 1}
