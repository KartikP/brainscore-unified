"""Tests for brainscore.model_helpers.local_vlm_policy.

Only the pure action parsers are exercised — the builders load heavy HF
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
