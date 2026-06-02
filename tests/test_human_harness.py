"""Tests for the v1.5 human harness scaffold.

The human harness realizes the Subject contract on a biological participant via
pluggable hardware I/O. These tests use in-memory mocks for the I/O callables
and confirm the benchmark-side call sequence routes correctly, plus that an
unwired harness fails explicitly.
"""
import pytest

from brainscore_core.model_interface import Subject, TaskContext
from brainscore.harnesses.human import HumanHarness


def test_is_subject():
    assert isinstance(HumanHarness("participant-01"), Subject)


def test_recording_routes_to_present_and_record():
    calls = []
    h = HumanHarness(
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
    h = HumanHarness(
        "p1",
        present_fn=lambda s: calls.append(("present", s)),
        respond_fn=lambda ctx, s: "button_left",
    )
    h.start_task(TaskContext(task_type="probabilities"))
    out = h.process("stim")
    assert ("present", "stim") in calls
    assert out == "button_left"


def test_no_hardware_raises():
    h = HumanHarness("p1", region_layer_map={"IT": "x"})
    h.start_recording("IT")
    with pytest.raises(NotImplementedError, match="no hardware"):
        h.process("stim")


def test_unconfigured_process_raises():
    h = HumanHarness("p1", present_fn=lambda s: None)
    with pytest.raises(RuntimeError, match="start_recording or start_task"):
        h.process("stim")


def test_reset_clears_config():
    h = HumanHarness(
        "p1", region_layer_map={"IT": "x"},
        present_fn=lambda s: None, record_fn=lambda r, s: 1,
    )
    h.start_recording("IT")
    h.reset()
    with pytest.raises(RuntimeError, match="start_recording or start_task"):
        h.process("stim")
