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
