"""Measure matched experiments, with a persistent model and live HTTP polling."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import tempfile
import threading
import time
from urllib.request import urlopen

from .experiment import Plan
from .server import ACTIVE, LabController, make_server
from .worker import DEFAULT_FACTORY


def wait(controller: LabController, *, ready: bool = False, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = controller.snapshot()
        if ready:
            if state["phase"] == "ready":
                return state
            if state["phase"] == "failed":
                raise RuntimeError(state.get("error"))
        elif state["run"] and state["run"]["phase"] not in ACTIVE:
            if state["run"]["phase"] != "complete":
                raise RuntimeError(state["run"].get("error", "Experiment failed"))
            return state["run"]
        time.sleep(0.001)
    raise TimeoutError("Experiment exceeded benchmark timeout")


def benchmark(factory: str, pairs: int, repeats: int, root: Path) -> dict:
    controller = LabController(root, factory)
    try:
        server = make_server(controller, Path(__file__).resolve().parents[2] / "frontend/dist", 0)
    except BaseException:
        controller.close()
        raise
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    polling = threading.Event()
    finished = threading.Event()

    def poll() -> None:
        while not finished.wait(0.05):
            if polling.is_set():
                with urlopen(f"http://127.0.0.1:{server.server_port}/api/state", timeout=2) as response:
                    response.read()

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    try:
        ready = wait(controller, ready=True)
        target = ready["profile"]["targets"][0]["name"]
        scenarios = {}
        for activity in (False, True):
            plan = Plan(target=target, ablation=True, activity=activity, repeats=repeats)
            # Warm both paths. Model loading and first-use imports are excluded.
            for telemetry in (False, True):
                controller.submit(asdict(plan), telemetry=telemetry)
                wait(controller)
            samples = {"direct": [], "gui": []}
            reports = []
            output_hashes = []
            for pair in range(pairs):
                # Alternate execution order to reduce warm-up and drift bias.
                for telemetry in ((False, True) if pair % 2 == 0 else (True, False)):
                    polling.set() if telemetry else polling.clear()
                    controller.submit(asdict(plan), telemetry=telemetry)
                    result = wait(controller)
                    samples["gui" if telemetry else "direct"].append(result["elapsed_s"])
                    reports.append(result["report"])
                    output_hashes.append(next(artifact["sha256"] for artifact in result["artifacts"]
                                              if artifact["path"] == "result.json"))
            direct = statistics.median(samples["direct"])
            gui = statistics.median(samples["gui"])
            overhead = 100 * (gui / direct - 1)
            scenarios["activity_recording" if activity else "behavior_recording"] = {
                "direct_median_s": direct, "gui_median_s": gui, "overhead_percent": overhead,
                "samples_s": samples, "summaries_identical": all(report == reports[0] for report in reports),
                "encoded_outputs_identical": len(set(output_hashes)) == 1,
                "within_budget": overhead <= 25,
            }
        return {"factory": factory, "pairs": pairs, "repeats_per_condition": repeats,
                "budget_percent": 25, "http_poll_interval_s": 0.05,
                "scope": "Warm experiment runtime, identical tools/artifacts; GUI telemetry and live HTTP polling added. Excludes model startup, browser rendering, and report download.",
                "scenarios": scenarios, "passed": all(value["within_budget"] and value["summaries_identical"]
                                                        and value["encoded_outputs_identical"] for value in scenarios.values())}
    finally:
        finished.set()
        poller.join(timeout=3)
        server.shutdown()
        server.server_close()
        controller.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", default=DEFAULT_FACTORY)
    parser.add_argument("--pairs", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--report", type=Path, default=Path("performance-result.json"))
    args = parser.parse_args()
    if args.pairs < 3 or not 1 <= args.repeats <= 100:
        parser.error("Use at least 3 pairs and 1 to 100 repetitions")
    with tempfile.TemporaryDirectory() as temporary:
        result = benchmark(args.factory, args.pairs, args.repeats, Path(temporary))
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
