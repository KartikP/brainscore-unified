"""The minimal core contract passes through each package's scoring entry point."""

import importlib

import numpy as np
import pandas as pd
import pytest

from brainscore_core import Benchmark, Score, Subject
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import neural_response


class NativeSubject(Subject):
    identifier = "native-contract-test"
    in_channels = {"text"}
    out_channels = {"neural:language_network"}
    required_channels = {"text"}

    def interact(self, session):
        while (event := session.next_input()) is not None:
            session.emit(StreamEvent(
                "neural:language_network", np.array([len(event.payload)]),
                event.t_ms, dict(event.meta),
            ))


class NativeBenchmark(Benchmark):
    identifier = "native-contract-benchmark"
    parent = "neural"
    required_input_channels = {"text"}
    requested_output_channels = {"neural:language_network"}
    uses_session = True

    def __call__(self, subject):
        responses = neural_response(subject, pd.DataFrame({
            "stimulus_id": ["a", "b"], "text": ["cat", "horse"],
        }), record="language_network")
        np.testing.assert_array_equal(responses.values, [[3], [5]])
        return Score(float(responses.sum()))


@pytest.mark.parametrize("package", ["brainscore", "brainscore_vision", "brainscore_language"])
def test_native_subject_loads_without_adaptation_and_scores(package, monkeypatch):
    domain = importlib.import_module(package)
    subject = NativeSubject()
    benchmark = NativeBenchmark()
    monkeypatch.setitem(domain.model_registry, subject.identifier, lambda: subject)
    monkeypatch.setitem(domain.benchmark_registry, benchmark.identifier, lambda: benchmark)
    # Plugin discovery is orthogonal to the contract and must not install anything.
    if hasattr(domain, "import_plugin"):
        monkeypatch.setattr(domain, "import_plugin", lambda *args, **kwargs: None)

    assert domain.load_model(subject.identifier) is subject
    score_fn = domain.score if package == "brainscore" else domain._run_score
    result = score_fn(subject.identifier, benchmark.identifier, check_mem=False)
    assert float(result) == 8.0
    assert result.attrs["out_channels"] == ("neural:language_network",)
