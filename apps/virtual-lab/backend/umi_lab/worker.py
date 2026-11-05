"""Keep model execution and weights in one persistent, spawned process."""
from __future__ import annotations

from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
from queue import Full
import signal
import shutil
import traceback

from .experiment import Cancelled, Plan, execute
from .profile import LabProfile

DEFAULT_FACTORY = "umi_lab.demo:create_profile"
MAX_MESSAGE = 64 * 1024


def error_message(error: Exception) -> str:
    message = str(error)
    return f"{type(error).__name__}: {message}" if message else f"{type(error).__name__}: the plugin supplied no additional error message"


def load_profile(factory: str) -> LabProfile:
    module, name = factory.split(":", 1)
    profile = getattr(importlib.import_module(module), name)()
    if not isinstance(profile, LabProfile):
        raise TypeError("Factory must return a LabProfile")
    return profile


def bounded_message(value: dict) -> dict:
    # Payloads crossing the process boundary are bounded metadata only.
    encoded = json.dumps(value, allow_nan=False)
    if len(encoded.encode()) > MAX_MESSAGE:
        raise ValueError("Lab reports must fit in 64 KiB; save large data as artifacts")
    return value


def run_worker(factory: str, root: str, commands: object, messages: object,
               stop: object) -> None:
    # The server owns Ctrl-C and requests cooperative cleanup. A terminal signal
    # must not independently interrupt the worker inside a model forward call.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # The local server never changes the user's cache configuration. The worker
    # uses app-local storage only when the user has not selected a cache.
    os.environ.setdefault("RESULTCACHING_HOME", str(Path(root).parent / ".lab-cache"))

    def send(value: dict, terminal: bool = False) -> None:
        value = bounded_message(value)
        if terminal:
            messages.put(value, timeout=5)
        else:
            try:
                messages.put_nowait(value)
            except Full:
                pass

    try:
        profile = load_profile(factory)
        send({"phase": "ready", "profile": profile.describe()}, terminal=True)
        from .workbench import catalog
        available = catalog()
        for key in ("models", "benchmarks", "stimulus_sets"):
            for start in range(0, len(available[key]), 100):
                send({"catalog": key, "items": available[key][start:start + 100]}, terminal=True)
        send({"catalog_ready": True}, terminal=True)
    except Exception as error:
        traceback.print_exc()
        send({"phase": "failed", "error": error_message(error)}, terminal=True)
        return
    while True:
        request = commands.get()
        if request is None:
            return
        if request.get("kind") == "load":
            from .workbench import describe_subject, release_model
            try:
                if profile is not None:
                    profile.subject.reset()
                profile = None
                release_model()
                if request["identifier"] == "__calibration__":
                    profile = load_profile(factory)
                    description = profile.describe()
                else:
                    from brainscore import load_model
                    from .profile import LabProfile
                    subject = load_model(request["identifier"])
                    description = describe_subject(subject)
                    profile = LabProfile(subject, subject.identifier, description["description"],
                                         description["task"], (), (), None, None)
                    del subject
                send({"phase": "ready", "profile": description, "run": None, "error": None}, terminal=True)
            except Exception as error:
                traceback.print_exc()
                send({"phase": "load_failed", "profile": None, "error": error_message(error), "run": None}, terminal=True)
            continue
        run_id = request["id"]
        directory = Path(root) / run_id
        try:
            directory.mkdir()
            if profile is None:
                raise ValueError("Load a subject first")

            def progress(value: dict) -> None:
                send({"id": run_id, **value})

            if request.get("kind") == "workbench":
                from .workbench import Request, run_session, run_benchmark
                experiment_request = Request.parse(request["request"])
                (directory / "plan.json").write_text(json.dumps({
                    "kind": "workbench", "subject": profile.subject.identifier,
                    "model_source": "profile" if profile.make_session else "registered",
                    "request": asdict(experiment_request)}, indent=2) + "\n")
                if experiment_request.mode == "benchmark":
                    result = run_benchmark(profile.subject, experiment_request, directory / "experiment", progress, stop.is_set)
                else:
                    inputs = directory / "inputs"
                    inputs.mkdir()
                    for row in experiment_request.rows + experiment_request.fitting_rows:
                        for column in ("image_path", "audio_path", "video_path"):
                            if column in row:
                                name = str(row[column])
                                source = (Path(root) / "uploads" / name).resolve()
                                if Path(name).name != name or not source.is_relative_to((Path(root) / "uploads").resolve()) or not source.is_file():
                                    raise ValueError("Unknown uploaded input file")
                                shutil.copyfile(source, inputs / name)
                    result = run_session(profile.subject, experiment_request, directory / "experiment", inputs, progress, stop.is_set)
            else:
                plan = Plan.parse(request["plan"])
                if profile.make_session is None:
                    raise ValueError("Use benchmark or stimulus mode for a registered subject")
                (directory / "plan.json").write_text(json.dumps(asdict(plan), indent=2) + "\n")
                result = execute(profile, plan, directory / "experiment",
                                 progress if request.get("telemetry", True) else None, stop.is_set)
            message = bounded_message({"id": run_id, "phase": "complete", **result})
            (directory / "summary.json").write_text(json.dumps(message, indent=2, allow_nan=False) + "\n")
            send(message, terminal=True)
        except Cancelled as error:
            send({"id": run_id, "phase": "cancelled", "error": str(error)}, terminal=True)
        except Exception as error:
            traceback.print_exc()
            send({"id": run_id, "phase": "failed", "error": error_message(error)}, terminal=True)
