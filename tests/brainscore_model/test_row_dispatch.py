"""Row dispatch.

A closed-loop step is a row (a mapping of channel -> payload). ``process(row)``
routes it to the agent's policy, which returns a mapping of output channel ->
payload. A message between agents is text.
"""
from typing import get_args

import pytest

from brainscore import BrainScoreModel
from brainscore_core.model_interface import InputEvent, OutputEvent


class TestUnions:
    def test_rows_are_in_both_unions(self):
        names = {getattr(arg, '__name__', None) for arg in get_args(InputEvent)}
        assert 'Mapping' in names
        assert 'Mapping' in {getattr(arg, '__name__', None) for arg in get_args(OutputEvent)}


class TestProcessRow:
    def test_routes_to_policy_returning_outputs(self):
        agent = BrainScoreModel(
            'responder',
            policy=lambda row: {'text': f"reply<{row['text']}>"})
        assert agent.process({'text': 'ping'}) == {'text': 'reply<ping>'}

    def test_no_policy_raises(self):
        model = BrainScoreModel('m')
        with pytest.raises(NotImplementedError, match='policy'):
            model.process({'text': 'x'})

    def test_bad_return_type_raises(self):
        agent = BrainScoreModel('a', policy=lambda row: 'raw string')
        with pytest.raises(TypeError, match='mapping of output channel'):
            agent.process({'text': 'x'})
