from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import threading
from urllib.request import Request as HTTPRequest, urlopen

import numpy as np
import pytest

from umi_lab.workbench import Request, catalog, describe_subject, run_session, run_benchmark, stimulus_table


def test_catalog_reads_registrations_without_loading_weights():
    available = catalog()
    assert "gpt2" in {entry["identifier"] for entry in available["models"]}
    assert "Pereira2018.243sentences-ridge" in {entry["identifier"] for entry in available["benchmarks"]}
    assert len(available["models"]) > 100


def test_registered_model_runs_user_stimuli_and_has_stable_avatar(tmp_path):
    from brainscore import load_model
    subject = load_model("chance-baseline")
    first = describe_subject(subject)
    assert first["avatar_seed"] == describe_subject(load_model("chance-baseline"))["avatar_seed"]
    request = Request(rows=[{"sentence": "A cat sits."}, {"sentence": "The sky is blue."}],
                      labels=["yes", "no"], instruction="Does this describe an animal?")
    result = run_session(subject, request, tmp_path / "run", tmp_path, lambda update: None, lambda: False)
    response = result["report"]["outputs"][0]["response"]
    assert response["shape"] == [2, 2]
    assert response["responses"][0]["scores"] == [.5, .5]
    assert response["responses"][0]["answer"] is None
    assert response["responses"][0]["selection_status"] == "tie"
    assert first["notice"] and "ignores stimulus content" in first["notice"]
    assert (tmp_path / "run/inputs_outputs/events.jsonl").is_file()


def test_user_questions_on_null_control_have_no_preference(tmp_path):
    from brainscore import load_model
    questions = ["Is earth the only known planet with life?", "Is a cat a mammal?", "Is a fish a mammal?", "Say no"]
    request = Request(rows=[{"sentence": sentence} for sentence in questions], labels=["yes", "no"],
                      instruction="Answer the question or follow the instruction in each stimulus. Reply only yes or no.")
    result = run_session(load_model("chance-baseline"), request, tmp_path / "run", tmp_path,
                         lambda update: None, lambda: False)
    responses = result["report"]["outputs"][0]["response"]["responses"]
    assert len(responses) == 4
    assert all(row["answer"] is None and row["selection_status"] == "tie" for row in responses)
    assert all(row["scores"] == [.5, .5] for row in responses)


def test_choice_preview_distinguishes_unique_tied_and_invalid_values():
    import xarray as xr
    from umi_lab.workbench import response_preview
    values = xr.DataArray([[.1, .9], [.5, .5], [np.nan, .5]], dims=["presentation", "choice"],
                          coords={"stimulus_id": ("presentation", ["unique", "tie", "invalid"]),
                                  "choice": ["yes", "no"]})
    rows = response_preview(values)["responses"]
    assert rows[0]["answer"] == "no" and rows[0]["selection_status"] == "selected"
    assert rows[1]["answer"] is None and rows[1]["tied_choices"] == ["yes", "no"]
    assert rows[2]["answer"] is None and rows[2]["selection_status"] == "invalid"
    assert rows[2]["scores"] == [None, .5]


def test_neural_stimulus_session_collects_an_assembly(tmp_path):
    from brainscore import BrainScoreModel
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    def extract(stimuli, layers):
        return NeuroidAssembly(np.array([[len(text), 1] for text in stimuli["sentence"]]),
                               dims=["presentation", "neuroid"],
                               coords={"stimulus_id": ("presentation", list(stimuli["stimulus_id"])),
                                       "neuroid_id": ("neuroid", ["a", "b"]), "layer": ("neuroid", ["units", "units"])})

    subject = BrainScoreModel("length-fixture", extractor={"text": extract}, region_layer_map={"language": "units"})
    request = Request(rows=[{"sentence": "cat"}, {"sentence": "elephant"}], output="neural:language")
    result = run_session(subject, request, tmp_path / "run", tmp_path, lambda update: None, lambda: False)
    assert result["report"]["outputs"][0]["response"]["shape"] == [2, 2]
    (tmp_path / "image.png").write_bytes(b"unused visual stream")
    partial = Request(rows=[{"sentence": "cat", "image_path": "image.png"}], output="neural:language")
    result = run_session(subject, partial, tmp_path / "partial", tmp_path, lambda update: None, lambda: False)
    assert result["report"]["compatibility"]["match"] == "partial"
    assert result["report"]["compatibility"]["dropped_inputs"] == ["vision"]


def test_session_json_uses_real_declared_capability(tmp_path):
    from brainscore.model_helpers.response_trace import build_trace_subject
    subject = build_trace_subject("request-fixture", provider=lambda request: {"text": "actual response"},
                                  parse=str, provenance={"revision": "fixture"})
    request = Request(mode="session", output="response_trace", events=[{
        "channel": "generation_request", "payload": {"messages": [{"role": "user", "content": "hello"}]}, "t_ms": None}])
    result = run_session(subject, request, tmp_path / "run", tmp_path, lambda update: None, lambda: False)
    response = result["report"]["outputs"][0]["response"]["preview"][0]
    assert response["raw_answer"] == "actual response" and response["valid"] is True


def test_text_stimuli_generate_verbatim_responses_through_declared_channel(tmp_path):
    from brainscore.model_helpers.response_trace import build_trace_subject
    seen = []
    def generate(request):
        seen.append(request)
        return {"text": "No."}
    subject = build_trace_subject("text-generation-fixture", provider=generate, parse=str, provenance={})
    request = Request(rows=[{"sentence": "Say no"}, {"sentence": "Is a fish a mammal?"}],
                      output="response_trace", instruction="Reply only yes or no")
    result = run_session(subject, request, tmp_path / "run", tmp_path, lambda update: None, lambda: False)
    responses = result["report"]["outputs"][0]["response"]["preview"]
    assert len(responses) == 2 and all(row["raw_answer"] == "No." for row in responses)
    assert responses[0]["stimulus_text"] == "Say no"
    assert seen[0]["system"] == request.instruction
    assert seen[0]["messages"] == [{"role": "user", "content": "Say no"}]


def test_new_json_catalog_entry_works_without_ui_dispatch_changes(tmp_path):
    from brainscore_core import Subject
    from brainscore_core.extensions import CatalogEntry, register_channel
    from brainscore_core.streaming import StreamEvent
    from brainscore.model_helpers.response_trace import register_trace_channels
    register_trace_channels()
    register_channel(CatalogEntry("lab_json_fixture", "input", "StreamEvent", "JSON object", "test", payload_type=dict))

    class Echo(Subject):
        identifier = "json-extension-fixture"
        in_channels = {"lab_json_fixture"}
        out_channels = {"response_trace"}
        def participate(self, session):
            while (event := session.next_input()) is not None:
                text = str(event.payload["value"])
                session.respond(StreamEvent("response_trace", {"raw": {"text": text}, "answer": text,
                                                                "valid": True, "parse_error": None}, None))

    subject = Echo()
    assert describe_subject(subject)["schemas"]["lab_json_fixture"]["json_input"]
    result = run_session(subject, Request(mode="session", output="response_trace", events=[{
        "channel": "lab_json_fixture", "payload": {"value": "extension response"}}]),
                         tmp_path / "run", tmp_path, lambda update: None, lambda: False)
    assert result["report"]["outputs"][0]["response"]["preview"][0]["answer"] == "extension response"


def test_custom_intervention_runs_through_subject_and_restores(tmp_path):
    import torch
    from brainscore import BrainScoreModel
    from brainscore.perturbation import build_pytorch_ablation_fn
    model = torch.nn.Sequential(torch.nn.Identity())

    def answer(stimulus_row, instruction, label_set):
        output = model(torch.tensor([[1., 0.]]))
        return "yes" if output[0, 0] > 0 else "no"

    subject = BrainScoreModel("intervention-fixture", supported_modalities={"text"}, answer=answer,
                              intervention=build_pytorch_ablation_fn(model))
    request = Request(rows=[{"sentence": "stimulus"}], labels=["yes", "no"], instruction="Answer yes or no",
                      intervention={"target": "0", "indices": [0], "operation": "zero"})
    result = run_session(subject, request, tmp_path / "run", tmp_path, lambda update: None, lambda: False)
    rows = result["report"]["outputs"]
    assert [row["response"]["responses"][0]["answer"] for row in rows] == ["yes", "no", "yes"]
    assert result["report"]["restored_exact"] is True


def test_existing_benchmark_path_preserves_direct_score(monkeypatch, tmp_path):
    import brainscore
    from brainscore_core.metrics import Score
    from brainscore_core import TaskContext
    from brainscore_core.streaming_helpers import behavioral_response
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    table = StimulusSet({"stimulus_id": ["a"], "sentence": ["A stimulus"]})
    table.identifier = "benchmark-fixture"

    class FixtureBenchmark:
        identifier = "local-fixture-benchmark"
        required_input_channels = {"text"}
        requested_output_channels = {"behavior"}
        ceiling = Score(1)
        version = 1

        def __call__(self, subject):
            assembly = behavioral_response(subject, TaskContext("probabilities", label_set=["yes", "no"],
                                                                instruction="Respond", metadata={"stimulus_set": table}))
            return Score(float(assembly.values[0, 0]))

    monkeypatch.setenv("RESULTCACHING_HOME", str(tmp_path / "cache"))
    monkeypatch.setitem(brainscore.benchmark_registry, FixtureBenchmark.identifier, FixtureBenchmark)
    direct = brainscore.score(brainscore.load_model("chance-baseline"), FixtureBenchmark())
    result = run_benchmark(brainscore.load_model("chance-baseline"), Request(mode="benchmark", benchmark=FixtureBenchmark.identifier),
                           tmp_path / "run", lambda update: None, lambda: False)
    assert result["report"]["score"]["sample"] == [float(direct)] == [.5]


@pytest.mark.parametrize("bad", [{"mode": "unknown"}, {"events": [1]}, {"intervention": {"target": "x", "indices": [-1]}},
                                  {"intervention": {"target": "x", "operation": "invented"}},
                                  {"mode": "benchmark", "benchmark": "x", "intervention": {"target": "x"}}])
def test_invalid_experiment_requests_fail_before_execution(bad):
    with pytest.raises(ValueError):
        Request.parse(bad)


def test_media_rows_only_accept_uploaded_assets(tmp_path):
    with pytest.raises(ValueError, match="uploaded"):
        stimulus_table([{"image_path": "/etc/passwd"}], tmp_path)
    (tmp_path / "asset.png").write_bytes(b"fixture")
    assert stimulus_table([{"image_path": "asset.png"}], tmp_path).stimulus_paths


def test_restoration_comparison_checks_values_beyond_the_preview():
    from umi_lab.workbench import same_output
    left = np.ones(200000, dtype=np.float32)
    right = left.copy()
    assert same_output(left, right)
    right[-1] = 2
    assert not same_output(left, right)


def test_incompatible_benchmark_is_na_never_zero(monkeypatch, tmp_path):
    import brainscore
    class Incompatible:
        identifier = "lab-neural-fixture"
        required_input_channels = {"text"}
        requested_output_channels = {"neural:IT"}
        def __call__(self, subject):
            raise AssertionError("Incompatible pair must not run")
    monkeypatch.setenv("RESULTCACHING_HOME", str(tmp_path / "cache"))
    monkeypatch.setitem(brainscore.benchmark_registry, Incompatible.identifier, Incompatible)
    result = run_benchmark(brainscore.load_model("chance-baseline"), Request(mode="benchmark", benchmark=Incompatible.identifier),
                           tmp_path / "run", lambda update: None, lambda: False)
    assert result["report"]["status"] == "N/A"
    assert "score" not in result["report"]


def test_worker_loads_registered_subject_and_recovers_from_failed_load(tmp_path):
    from umi_lab.benchmark import wait
    from umi_lab.server import LabController
    import time
    controller = LabController(tmp_path / "runs")
    try:
        wait(controller, ready=True)
        controller.load({"identifier": "chance-baseline"})
        ready = wait(controller, ready=True)
        assert ready["profile"]["calibration"] is False
        controller.submit_request({"mode": "stimuli", "rows": [{"sentence": "A cat"}],
                                   "labels": ["yes", "no"], "instruction": "Respond"})
        assert wait(controller)["report"]["outputs"][0]["response"]["shape"] == [1, 2]
        controller.load({"identifier": "__calibration__"})
        assert wait(controller, ready=True)["profile"]["calibration"] is True
        controller.load({"identifier": "__missing_lab_model__"})
        deadline = time.monotonic() + 10
        while controller.snapshot()["phase"] != "load_failed" and time.monotonic() < deadline:
            time.sleep(.01)
        assert controller.snapshot()["phase"] == "load_failed"
        controller.load({"identifier": "chance-baseline"})
        assert wait(controller, ready=True)["profile"]["identifier"] == "chance-baseline"
    finally:
        controller.close()
