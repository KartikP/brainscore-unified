from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from umi_lab.demo import create_profile
from umi_lab.experiment import Cancelled, Plan, build_experiment, execute, selections
from umi_lab.worker import bounded_message


@pytest.fixture
def profile():
    return create_profile()


def test_tools_change_behavior_and_cleanup_restores_exact_outputs(profile, tmp_path):
    from brainscore.experiments import compare_outputs
    plan = Plan(activity=True, ablation=True)
    updates = []
    direct = execute(profile, plan, tmp_path / "direct")
    observed = execute(profile, plan, tmp_path / "observed", updates.append)
    assert direct["report"] == observed["report"]
    assert compare_outputs(tmp_path / "direct", tmp_path / "observed")["equal"]
    assert [row["accuracy"] for row in observed["report"]["conditions"]] == [1, .5, 1, 1]
    assert observed["report"]["restored_exact"] is True
    assert len(updates) == 4
    assert all(set(update) == {"phase", "condition", "repetition"} for update in updates)
    assert (tmp_path / "observed/inputs_outputs/events.jsonl").is_file()
    assert execute(profile, Plan(), tmp_path / "again")["report"]["conditions"][0]["accuracy"] == 1


def test_failure_and_cancellation_do_not_leave_interventions(profile, tmp_path):
    def cancelled():
        return True

    with pytest.raises(Cancelled):
        execute(profile, Plan(ablation=True), tmp_path / "cancel", lambda event: None, cancelled)
    assert json.loads((tmp_path / "cancel/experiment.json").read_text())["status"] == "failed"

    def fail_during_ablation(event):
        if event.get("condition") == "ablation":
            raise RuntimeError("fixture failure")

    with pytest.raises(RuntimeError, match="fixture failure"):
        execute(profile, Plan(ablation=True), tmp_path / "failed", fail_during_ablation)
    restored = execute(profile, Plan(), tmp_path / "restored")
    assert restored["report"]["conditions"][0]["accuracy"] == 1


def test_recording_failure_during_forward_removes_ablation(profile, tmp_path):
    import numpy as np
    from brainscore.experiments import TorchInstrumentation

    class BrokenRecorder(TorchInstrumentation):
        def record(self, targets, receive):
            def fail_on_zero_signal(target, value):
                if np.all(value["array"][..., :2] == 0):
                    raise RuntimeError("recording failed during ablation")
                receive(target, value)
            return super().record(targets, fail_on_zero_signal)

    broken = replace(profile, instrumentation=BrokenRecorder(profile.instrumentation.targets))
    with pytest.raises(RuntimeError, match="recording failed during ablation"):
        execute(broken, Plan(activity=True, ablation=True), tmp_path / "failed")
    assert execute(profile, Plan(), tmp_path / "restored")["report"]["conditions"][0]["accuracy"] == 1


@pytest.mark.parametrize("bad", [{"fraction": 101}, {"fraction": True}, {"activity": "yes"},
                                  {"repeats": 0}, {"seed": -1}, {"factory": "unsafe"}, []])
def test_invalid_plans_fail_before_compute(bad):
    with pytest.raises((ValueError, TypeError)):
        Plan.parse(bad)


def test_targets_are_explicit_and_controls_are_reproducible(profile, tmp_path):
    selected = selections(profile, Plan(ablation=True))
    assert selected == selections(profile, Plan(ablation=True))
    assert selected["selected"] == [0, 1]
    assert len(selected["control"]) == 2 and not selected["overlap"]
    assert selections(profile, Plan(fraction=100))["overlap"] == 8
    with pytest.raises(ValueError, match="selected target"):
        build_experiment(profile, Plan(target="missing", ablation=True), tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_incompatible_channels_fail_before_execution(profile, tmp_path):
    incompatible = replace(profile, output_channels=("motor",))
    with pytest.raises(ValueError, match="Incompatible"):
        build_experiment(incompatible, Plan(), tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_browser_messages_cannot_contain_large_arrays():
    with pytest.raises(ValueError, match="64 KiB"):
        bounded_message({"tensor": "x" * 65536})
    with pytest.raises(ValueError):
        bounded_message({"invalid": float("nan")})


def test_server_process_does_not_import_model_libraries():
    backend = Path(__file__).resolve().parents[1] / "backend"
    result = subprocess.run([sys.executable, "-c",
                             "import sys; sys.path.insert(0, sys.argv[1]); import umi_lab.server; "
                             "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
                             "assert 'brainscore' not in sys.modules", str(backend)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_http_runs_in_spawned_worker_and_rejects_untrusted_requests(tmp_path):
    from umi_lab.benchmark import wait
    from umi_lab.server import LabController, make_server

    controller = LabController(tmp_path / "runs")
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<title>Lab test</title>")
    server = make_server(controller, frontend, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        wait(controller, ready=True)
        with urlopen(url + "/api/state") as response:
            state = json.load(response)
        assert state["profile"]["identifier"] == "comparison-circuit-v1"
        body = json.dumps({"ablation": True, "activity": True}).encode()
        with pytest.raises(HTTPError) as denied:
            urlopen(Request(url + "/api/runs", body, {"Content-Type": "application/json"}))
        assert denied.value.code == 403
        headers = {"Content-Type": "application/json", "X-Lab-Token": state["token"]}
        with pytest.raises(HTTPError) as cross_origin:
            urlopen(Request(url + "/api/runs", body, {**headers, "Origin": "http://untrusted.example"}))
        assert cross_origin.value.code == 403
        with urlopen(Request(url + "/api/runs", body, headers)) as response:
            run_id = json.load(response)["id"]
        result = wait(controller)
        assert result["report"]["restored_exact"] is True
        with urlopen(url + f"/api/runs/{run_id}/plan.json") as response:
            assert json.load(response)["ablation"] is True
        with urlopen(url + f"/api/runs/{run_id}/artifacts.zip") as response:
            assert response.read(2) == b"PK"
        with pytest.raises(HTTPError) as traversal:
            urlopen(url + "/../backend/umi_lab/server.py")
        assert traversal.value.code == 404
        with urlopen(Request(url + "/api/uploads", b"fixture-image", {
                "Content-Type": "application/octet-stream", "X-Lab-Token": state["token"],
                "X-Lab-Filename": "stimulus.png"})) as response:
            asset = json.load(response)["asset"]
        assert (controller.root / "uploads" / asset).read_bytes() == b"fixture-image"
    finally:
        server.shutdown()
        server.server_close()
        controller.close()
    assert not controller.process.is_alive()
