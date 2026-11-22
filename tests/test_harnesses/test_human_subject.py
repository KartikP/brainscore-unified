"""Tests for the v1.5 human subject scaffold.

The human subject realizes the Subject contract on a biological participant via
pluggable hardware I/O. These tests use in-memory mocks for the I/O callables
and confirm the benchmark-side call sequence routes correctly, plus that an
unwired harness fails explicitly.
"""
import pytest

from brainscore_core.contract import TaskContext
from brainscore.harnesses.human import HumanSubject


def test_recording_routes_to_present_and_record():
    calls = []
    h = HumanSubject(
        "p1",
        region_layer_map={"IT": "electrode_array_1"},
        present_fn=lambda s: calls.append(("present", s)),
        record_fn=lambda region, s: (calls.append(("record", region)), "neural_data")[1],
    )
    h.start_recording("IT")
    out = h.process("stimulus_X")
    assert ("present", "stimulus_X") in calls
    assert ("record", "IT") in calls
    assert out == "neural_data"


def test_task_routes_to_present_and_respond():
    calls = []
    h = HumanSubject(
        "p1",
        present_fn=lambda s: calls.append(("present", s)),
        respond_fn=lambda ctx, s: "button_left",
    )
    h.start_task(TaskContext(task_type="probabilities"))
    out = h.process("stim")
    assert ("present", "stim") in calls
    assert out == "button_left"


def test_no_hardware_raises():
    h = HumanSubject("p1", region_layer_map={"IT": "x"})
    h.start_recording("IT")
    with pytest.raises(NotImplementedError, match="no hardware"):
        h.process("stim")


def test_unconfigured_process_raises():
    h = HumanSubject("p1", present_fn=lambda s: None)
    with pytest.raises(RuntimeError, match="answers in a session"):
        h.process("stim")


def test_reset_clears_config():
    h = HumanSubject(
        "p1", region_layer_map={"IT": "x"},
        present_fn=lambda s: None, record_fn=lambda r, s: 1,
    )
    h.start_recording("IT")
    h.reset()
    with pytest.raises(RuntimeError, match="answers in a session"):
        h.process("stim")


def test_an_undeclared_input_never_reaches_the_rig():
    import numpy as np
    from brainscore_core.streaming_helpers import neural_response
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    stimuli = StimulusSet({'stimulus_id': ['a', 'b'], 'image_path': ['a.png', 'b.png'],
                           'sentence': ['one', 'two']})
    stimuli.identifier = 'rig-inputs'
    shown = []

    def record(region, table):
        ids = list(table['stimulus_id'])
        return NeuroidAssembly(np.ones((len(ids), 1)), dims=['presentation', 'neuroid'],
                               coords={'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['n'])})

    rig = HumanSubject("p1", region_layer_map={"IT": "array"}, present_fn=shown.append, record_fn=record)
    neural_response(rig, stimuli, ['neural:IT'])
    assert shown and all('sentence' not in table.columns for table in shown)   # text is not declared


def _choice_rig(shown, **kwargs):
    """A rig whose button box gives a choice between the labels of a ``Choice``."""
    import numpy as np
    from brainscore_core.response_formats import response_record
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

    def respond(task, table):
        ids, labels = list(table['stimulus_id']), list(task.response.labels)
        return BehavioralAssembly(
            np.tile(np.eye(len(labels), dtype=np.float32)[0], (len(ids), 1)), dims=['presentation', 'choice'],
            coords={'stimulus_id': ('presentation', ids), 'choice': labels},
            attrs={'response': response_record('behavior:choice', 'button box')})
    return HumanSubject("p1", present_fn=shown.append, respond_fn=respond, **kwargs)


class _Benchmark:
    identifier = 'rig-formats'
    trial_memory = 'none'

    def __init__(self, outputs):
        self.required_input_channels = frozenset({'vision:image'})
        self.requested_output_channels = frozenset(outputs)


def test_a_rig_answers_a_format_it_states_through_participate():
    from brainscore_core.compatibility import check_compatibility
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    stimuli = StimulusSet({'stimulus_id': ['a', 'b'], 'image_path': ['a.png', 'b.png']})
    stimuli.identifier = 'rig-formats'
    shown = []
    rig = _choice_rig(shown, out_channels={'behavior:choice'})
    assert 'behavior:choice' in rig.out_channels
    check_compatibility(rig, _Benchmark({'behavior:choice'}))
    session = stimulus_session(stimuli, ['behavior:choice'], TaskContext(response=Choice(['cat', 'dog'])))
    rig.participate(session)
    assert len(shown) == 1
    assert [event.channel for event in session.emitted] == ['behavior:choice']


def test_a_format_the_rig_did_not_state_is_not_applicable_before_any_stimulus():
    from brainscore_core.compatibility import CompatibilityError, check_compatibility
    shown = []
    for rig in (_choice_rig(shown), _choice_rig(shown, out_channels={'behavior:choice'})):
        with pytest.raises(CompatibilityError):
            check_compatibility(rig, _Benchmark({'behavior:probabilities'}))
    assert shown == []


def test_a_rig_that_states_no_format_declares_no_behavior():
    # It still answers a direct session on bare ``behavior`` with its respond function.
    assert not any(channel.startswith('behavior') for channel in _choice_rig([]).out_channels)


def test_a_rig_refuses_a_stated_output_that_is_not_a_behavior_channel():
    with pytest.raises(ValueError, match='behavior'):
        _choice_rig([], out_channels={'behavior:guess'})
    with pytest.raises(ValueError, match='behavior'):
        _choice_rig([], out_channels={'neural:IT'})


@pytest.mark.parametrize('other', ['behavior', 'neural:V4'])
def test_a_rig_refuses_an_output_it_does_not_give_before_any_stimulus(other):
    """With a stated format, every other requested output must be a recorded region."""
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    stimuli = StimulusSet({'stimulus_id': ['a'], 'image_path': ['a.png']})
    stimuli.identifier = 'rig-other-output'
    shown = []
    rig = _choice_rig(shown, out_channels={'behavior:choice'}, region_layer_map={'IT': 'array'},
                      record_fn=lambda region, table: shown.append(region))
    session = stimulus_session(stimuli, ['behavior:choice', other], TaskContext(response=Choice(['cat', 'dog'])))
    with pytest.raises(ExecutionNotFeasibleError, match=other):
        rig.participate(session)
    assert shown == []
