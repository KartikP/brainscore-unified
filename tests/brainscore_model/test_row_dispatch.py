"""Output-type metric dispatch and row dispatch.

A closed-loop step is a row (a mapping of channel -> payload). ``process(row)``
routes it to the agent's policy, which returns a mapping of output channel ->
payload. A message between agents is text.
"""
from typing import get_args

import pytest

from brainscore import BrainScoreModel
from brainscore_core.model_interface import PerturbationApplied, Selection, Perturbation, InputEvent, OutputEvent, output_event_kind, dispatch_metric
from brainscore_core.streaming import StreamEvent


class TestUnions:
    def test_rows_are_in_both_unions(self):
        names = {getattr(arg, '__name__', None) for arg in get_args(InputEvent)}
        assert 'Mapping' in names
        assert 'Mapping' in {getattr(arg, '__name__', None) for arg in get_args(OutputEvent)}


class TestOutputEventKind:
    def test_stream_events_perturbation(self):
        assert output_event_kind(StreamEvent('motor', [0.0], None)) == 'motor'
        assert output_event_kind(StreamEvent('text', 'x', None)) == 'text'
        pa = PerturbationApplied(
            handle_id='h', target=Selection(layer='L', indices=[0]),
            perturbation=Perturbation(kind='zero'))
        assert output_event_kind(pa) == 'perturbation'

    def test_assembly_kinds_via_duck_typing(self):
        # core can't import brainio's concrete classes; classification is by
        # MRO class name, so locally-named stand-ins resolve correctly.
        class NeuroidAssembly:  # noqa: N801
            pass

        class BehavioralAssembly:  # noqa: N801
            pass

        assert output_event_kind(NeuroidAssembly()) == 'neural'
        assert output_event_kind(BehavioralAssembly()) == 'behavioral'

    def test_unknown(self):
        assert output_event_kind(object()) == 'unknown'


class TestDispatchMetric:
    def test_dispatch_by_kind(self):
        metric_map = {'text': 'M', 'behavioral': 'B'}
        assert dispatch_metric(StreamEvent('text', 'x', None), metric_map) == 'M'

    def test_missing_kind_raises(self):
        with pytest.raises(KeyError, match='neural'):
            dispatch_metric(StreamEvent('motor', [0.0], None), {'neural': 'N'})


class TestProcessRow:
    def test_routes_to_policy_returning_outputs(self):
        agent = BrainScoreModel(
            'responder', None, {}, {}, None,
            policy=lambda row: {'text': f"reply<{row['text']}>"})
        assert agent.process({'text': 'ping'}) == {'text': 'reply<ping>'}

    def test_no_policy_raises(self):
        model = BrainScoreModel('m', None, {}, {}, None)
        with pytest.raises(NotImplementedError, match='policy'):
            model.process({'text': 'x'})

    def test_bad_return_type_raises(self):
        agent = BrainScoreModel('a', None, {}, {}, None, policy=lambda row: 'raw string')
        with pytest.raises(TypeError, match='mapping of output channel'):
            agent.process({'text': 'x'})
