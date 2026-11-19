"""Registry-backed models, existing benchmarks, and arbitrary stimulus sessions."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

from .experiment import Cancelled, progress_tool
from .profile import LabProfile


def catalog() -> dict:
    """List registrations without importing every plugin or loading weights."""
    import brainscore

    models = {name: "unified" for name in brainscore.model_registry}
    benchmarks = {name: "unified" for name in brainscore.benchmark_registry}
    stimuli = {name: "unified" for name in brainscore.stimulus_set_registry}
    for package in ("brainscore_vision", "brainscore_language"):
        spec = importlib.util.find_spec(package)
        if spec is None:
            continue
        for root in spec.submodule_search_locations or ():
            for folder, registry, destination in (("models", "model_registry", models),
                                                   ("benchmarks", "benchmark_registry", benchmarks),
                                                   ("data", "stimulus_set_registry", stimuli)):
                for file in sorted((Path(root) / folder).glob("*/__init__.py")):
                    try:
                        tree = ast.parse(file.read_text())
                    except (SyntaxError, UnicodeError, OSError):
                        continue
                    for node in ast.walk(tree):
                        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                                and node.value.id == registry and isinstance(node.slice, ast.Constant)
                                and isinstance(node.slice.value, str)):
                            destination.setdefault(node.slice.value, package.removeprefix("brainscore_"))
    return {"models": [{"identifier": name, "source": models[name]} for name in sorted(models)],
            "benchmarks": [{"identifier": name, "source": benchmarks[name]} for name in sorted(benchmarks)],
            "stimulus_sets": sorted(stimuli),
            "note": "Literal registrations plus the unified registry; identifiers registered dynamically can also be entered directly."}


def describe_subject(subject: Any) -> dict:
    from brainscore_core import io_catalog

    channels = sorted(set(subject.in_channels) | set(subject.out_channels))
    schemas = {}
    for channel in channels:
        if io_catalog.has(channel):
            entry = io_catalog.get(channel)
            schemas[channel] = {"direction": entry.direction, "payload": entry.payload_contract,
                                "addressing": entry.addressing, "json_input": json_channel(channel)}
    # This is documentation for a known registration, not capability discovery
    # or routing. The null's implementation explicitly returns uniform choices.
    null_control = subject.identifier == "chance-baseline"
    return {"identifier": subject.identifier, "title": subject.identifier,
            "description": "Uniform-probability null control." if null_control else "Registered UMI subject. Select a benchmark or supply your own stimuli.",
            "notice": "This null control ignores stimulus content and returns equal choice values. Load an instruction-following model to answer questions." if null_control else "",
            "task": "Choose an experiment", "in_channels": sorted(subject.in_channels),
            "out_channels": sorted(subject.out_channels), "required_channels": sorted(subject.required_channels),
            "traits": sorted(subject.traits),
            "targets": [], "provenance": {"source": "registered model"}, "schemas": schemas,
            "avatar_seed": hashlib.sha256(subject.identifier.encode()).hexdigest()[:16],
            "calibration": False}


def json_channel(channel: str) -> bool:
    """Choose a JSON editor from a catalog declaration, not a subject class."""
    from brainscore_core import io_catalog
    if not io_catalog.has(channel):
        return False
    entry = io_catalog.get(channel)
    declared = entry.payload_type
    types = declared if isinstance(declared, tuple) else (declared,)
    if any(item in (str, dict, list, int, float, bool, type(None)) for item in types):
        return True
    # Older extension entries documented their JSON schema before payload_type.
    return declared is None and entry.payload_contract.lower().lstrip().startswith(("dict", "str", "text", "list", "json"))


def release_model() -> None:
    """Release allocator caches after the runtime has dropped its old subject."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@dataclass
class Request:
    mode: str = "stimuli"
    benchmark: str = ""
    rows: list[dict] = field(default_factory=list)
    fitting_rows: list[dict] = field(default_factory=list)
    stimulus_set: str = ""
    output: str = "behavior"
    instruction: str = ""
    labels: list[str] = field(default_factory=list)
    prefer_path: str = "auto"
    events: list[dict] = field(default_factory=list)
    intervention: dict | None = None
    record: bool = True
    repeats: int = 1
    protocol: dict | None = None

    @classmethod
    def parse(cls, value: dict) -> Request:
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown workbench settings")
        request = cls(**value)
        if request.mode not in ("benchmark", "stimuli", "session", "protocol"):
            raise ValueError("Choose benchmark, stimuli, or session mode")
        if request.mode == "protocol":
            from .protocol import ProtocolSpec
            ProtocolSpec.parse(request.protocol)
        for name in ("benchmark", "stimulus_set", "output", "instruction", "prefer_path"):
            if not isinstance(getattr(request, name), str):
                raise ValueError(f"{name} must be text")
        if request.mode == "benchmark" and not request.benchmark.strip():
            raise ValueError("Enter a registered benchmark identifier")
        if type(request.record) is not bool or type(request.repeats) is not int or not 1 <= request.repeats <= 100:
            raise ValueError("Use a recording checkbox and 1 to 100 repetitions")
        if request.prefer_path not in ("auto", "generation", "readout"):
            raise ValueError("Behavioral path must be auto, generation, or readout")
        if not isinstance(request.labels, list) or not all(isinstance(label, str) for label in request.labels):
            raise ValueError("Labels must be a list of strings")
        for rows in (request.rows, request.fitting_rows, request.events):
            if not isinstance(rows, list) or len(rows) > 10000 or not all(isinstance(row, dict) for row in rows):
                raise ValueError("Provide at most 10000 structured rows/events")
        if request.mode == "benchmark" and request.intervention:
            raise ValueError("Use a custom session or an intervention benchmark for perturbations; standard benchmark runs preserve their protocol")
        if request.intervention is not None:
            change = request.intervention
            if not isinstance(change, dict) or set(change) - {"target", "indices", "operation", "value"}:
                raise ValueError("Invalid intervention fields")
            if not isinstance(change.get("target"), str) or not change["target"]:
                raise ValueError("An intervention requires an explicit address")
            if change.get("operation", "zero") not in ("zero", "scale", "drive"):
                raise ValueError("Intervention operation must be zero, scale, or drive")
            indices = change.get("indices")
            if indices is not None and (not isinstance(indices, list) or not indices
                                        or any(type(index) is not int or index < 0 for index in indices)):
                raise ValueError("Indices must be nonnegative integers, or null for the whole target")
            amount = change.get("value", 0)
            if type(amount) not in (int, float) or not math.isfinite(amount):
                raise ValueError("Intervention value must be finite")
        return request


def stimulus_table(rows: list[dict], uploads: Path) -> Any:
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    if not rows:
        raise ValueError("Add stimuli, upload a CSV, or choose a registered stimulus set")
    copied, paths = [], {}
    for index, row in enumerate(rows):
        row = dict(row)
        row.setdefault("stimulus_id", f"stimulus-{index}")
        for column in ("image_path", "audio_path", "video_path"):
            if column not in row:
                continue
            asset = str(row[column])
            path = (uploads / asset).resolve()
            if Path(asset).name != asset or not path.is_relative_to(uploads.resolve()) or not path.is_file():
                raise ValueError("Media rows must reference a file uploaded through this lab")
            row[column] = str(path)
            if column == "image_path":
                paths[row["stimulus_id"]] = path
        copied.append(row)
    table = StimulusSet(copied)
    if table["stimulus_id"].duplicated().any():
        raise ValueError("Stimulus IDs must be unique")
    table.identifier = "lab-" + hashlib.sha256(json.dumps(copied, sort_keys=True).encode()).hexdigest()[:24]
    table.stimulus_paths = paths
    return table


def table_inputs(table: Any) -> tuple:
    """The input channels a stimulus table presents: read from the session's own events."""
    from brainscore_core.streaming import CONTROL_CHANNEL
    from brainscore_core.streaming_helpers import stimulus_session

    session = stimulus_session(table)
    channels = set()
    while (event := session.next_input()) is not None:
        if event.channel != CONTROL_CHANNEL:
            channels.add(event.channel)
    return tuple(sorted(channels))


def response_preview(value: Any) -> dict:
    """Summarize outputs with bounded previews; full arrays stay in artifacts."""
    import numpy as np
    import xarray as xr
    from brainscore_core.streaming import StreamEvent

    if isinstance(value, xr.DataArray):
        from .previews import activation_preview
        rows = []
        if "choice" in value.dims and "presentation" in value.dims and value.ndim == 2:
            choices = [str(choice) for choice in value["choice"].values]
            for index in range(min(20, value.sizes["presentation"])):
                scores = np.asarray(value.isel(presentation=index).values)
                valid = bool(scores.size and np.isfinite(scores).all())
                winners = np.flatnonzero(scores == scores.max()) if valid else np.array([], dtype=int)
                status = "selected" if len(winners) == 1 else "tie" if len(winners) > 1 else "invalid"
                rows.append({"stimulus": str(value["stimulus_id"].values[index]),
                             "stimulus_text": str(value["sentence"].values[index])[:500] if "sentence" in value.coords else None,
                             "answer": choices[int(winners[0])] if status == "selected" else None,
                             "selection_status": status,
                             "tied_choices": [choices[int(winner)] for winner in winners] if status == "tie" else [],
                             "scores": [float(score) if np.isfinite(score) else None for score in scores[:10]]})
        sample = np.asarray(value.values).flat[:16]
        attrs = {}
        for name, item in list(value.attrs.items())[:20]:
            if isinstance(item, (str, bool, int, float)):
                attrs[name] = item if not isinstance(item, str) else item[:1000]
            elif isinstance(item, xr.DataArray) and item.size == 1:
                number = float(item)
                attrs[name] = number if math.isfinite(number) else None
        preview = activation_preview(value)
        return {"type": type(value).__name__, "shape": list(value.shape), "dims": list(value.dims), "attrs": attrs,
                **({"activation_preview": preview} if preview is not None else {}),
                "sample": [float(item) if np.isfinite(item) else None for item in sample]
                          if sample.dtype.kind in "fiu" else [str(item) for item in sample],
                "responses": rows,
                "choice_labels": [str(choice) for choice in value["choice"].values[:10]] if "choice" in value.dims else []}
    if isinstance(value, StreamEvent):
        payload = value.payload
        if isinstance(payload, dict) and "valid" in payload:
            raw = payload.get("raw", {})
            return {"channel": value.channel, "valid": payload["valid"],
                    "stimulus": value.meta.get("stimulus_id"), "stimulus_text": value.meta.get("sentence"),
                    "parse_error": payload.get("parse_error"),
                    "raw_answer": str(raw.get("text", ""))[:2000],
                    "answer": str(payload.get("answer"))[:2000]}
        return {"channel": value.channel, "payload": response_preview(payload)}
    if isinstance(value, list):
        return {"type": "events", "count": len(value), "preview": [response_preview(item) for item in value[:8]]}
    if isinstance(value, dict):
        return {str(key): response_preview(item) for key, item in list(value.items())[:8]}
    return {"type": type(value).__name__, "preview": str(value)[:2000]}


def same_output(left: Any, right: Any) -> bool:
    """Compare restored outputs with bounded temporary memory, including labels."""
    import numpy as np
    import xarray as xr

    if isinstance(left, xr.DataArray):
        return (isinstance(right, xr.DataArray) and left.dims == right.dims
                and set(left.coords) == set(right.coords) and same_output(left.attrs, right.attrs)
                and same_output(left.values, right.values)
                and all(left[name].dims == right[name].dims
                        and same_output(left[name].values, right[name].values) for name in left.coords))
    if isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.shape != right.shape or left.dtype != right.dtype:
            return False
        for start in range(0, left.size, 65536):
            a, b = left.flat[start:start + 65536], right.flat[start:start + 65536]
            if left.dtype.hasobject:
                if not np.array_equal(a, b):
                    return False
            elif a.tobytes() != b.tobytes():
                return False
        return True
    if type(left) is not type(right):
        return False
    if is_dataclass(left):
        return all(same_output(getattr(left, item.name), getattr(right, item.name)) for item in fields(left))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_output(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_output(a, b) for a, b in zip(left, right))
    return bool(left == right)


def intervention_tool(subject: Any, change: dict) -> Any:
    from brainscore.experiments import Intervene
    from brainscore_core.events import Perturbation, Selection, StateChange

    operation = change.get("operation", "zero")
    channel = "lesion" if operation == "zero" else "stimulation"
    if channel not in subject.in_channels or "perturbation" not in subject.out_channels:
        raise ValueError(f"Subject does not declare {channel} input and perturbation output")
    state_change = StateChange(kind="ablation" if operation == "zero" else "stimulation",
                               target=Selection(layer=change["target"], indices=change.get("indices")),
                               perturbation=Perturbation(kind=operation, scale=change.get("value", 0),
                                                         amount=change.get("value", 0)))
    return Intervene(state_change, conditions=["intervention"], name="subject_intervention")


def run_session(subject: Any, request: Request, directory: Path, uploads: Path,
                send: Callable, cancelled: Callable) -> dict:
    from brainscore import load_stimulus_set
    from brainscore.experiments import Experiment, SessionProtocol, RecordInputsOutputs
    from brainscore_core import TaskContext
    from brainscore_core.streaming import Session, StreamEvent
    from brainscore_core.streaming_helpers import stimulus_session

    if request.output not in subject.out_channels:
        raise ValueError(f"Subject does not declare output {request.output}")
    if request.mode == "session":
        if not request.events:
            raise ValueError("Add at least one input event")
        events = []
        for row in request.events:
            if set(row) - {"channel", "payload", "t_ms", "meta"} or "channel" not in row or "payload" not in row:
                raise ValueError("Each event needs channel and payload; optional t_ms and meta")
            if row["channel"] not in subject.in_channels:
                raise ValueError(f"Subject does not declare input {row['channel']}")
            if row.get("t_ms") is not None and (type(row["t_ms"]) not in (int, float) or not math.isfinite(row["t_ms"])):
                raise ValueError("Event time must be finite milliseconds or null")
            # Data-only channels such as generation/log-probability requests are
            # JSON-native. Dataclass/environment schemas need their own harness.
            if not json_channel(row["channel"]):
                raise ValueError("This channel needs a typed payload harness; use its benchmark or a trusted Python profile")
            events.append(StreamEvent(row["channel"], row["payload"], row.get("t_ms"), row.get("meta", {})))
        inputs = tuple(sorted({event.channel for event in events}))

        @contextmanager
        def factory(trial: Any) -> Any:
            yield Session(events, requested_output_channels=(request.output,))
    else:
        table = load_stimulus_set(request.stimulus_set) if request.stimulus_set else stimulus_table(request.rows, uploads)
        inputs = table_inputs(table)
        if request.output == "behavior":
            fitting = stimulus_table(request.fitting_rows, uploads) if request.fitting_rows else None
            task = TaskContext(task_type="probabilities", instruction=request.instruction or None,
                               label_set=request.labels or None, fitting_stimuli=fitting,
                               prefer_path=request.prefer_path, metadata={"stimulus_set": table})

            @contextmanager
            def factory(trial: Any) -> Any:
                yield stimulus_session(table, ('behavior',), task)
        elif request.output.startswith("neural:"):
            @contextmanager
            def factory(trial: Any) -> Any:
                yield stimulus_session(table, (request.output,))
        elif request.output == "response_trace" and "generation_request" in subject.in_channels:
            if "sentence" not in table.columns or set(inputs) != {"text"}:
                raise ValueError("Text generation needs a sentence column; other payloads require their declared request channel")
            events = [StreamEvent("generation_request", {
                "system": request.instruction or None,
                "messages": [{"role": "user", "content": str(row["sentence"])}],
                "temperature": 0, "max_tokens": 64,
            }, None, {"stimulus_id": str(row["stimulus_id"]), "sentence": str(row["sentence"])})
                      for _, row in table.iterrows()]
            inputs = ("generation_request",)

            @contextmanager
            def factory(trial: Any) -> Any:
                yield Session(events, requested_output_channels=(request.output,))
        else:
            raise ValueError("For generation or other request channels, use Session JSON mode")
    if not (set(inputs) & set(subject.in_channels)) or set(subject.required_channels) - set(inputs):
        raise ValueError("The supplied inputs do not satisfy the subject's declared required channels")
    tools = [RecordInputsOutputs()] if request.record else []
    tools.append(progress_tool(send, cancelled, activity=(
        "Viewing stimuli" if request.output.startswith("neural:") else
        "Processing inputs and generating responses" if request.output == "response_trace"
        else "Processing stimuli and computing responses")))
    if request.intervention:
        tools.append(intervention_tool(subject, request.intervention))
    conditions = ("baseline", "intervention", "restored") if request.intervention else ("baseline",)
    experiment = Experiment(subject=subject, protocol=SessionProtocol(
        "umi-workbench-session-v1", factory, input_channels=inputs, output_channels=(request.output,),
        conditions=conditions, repetitions=tuple(range(request.repeats))), tools=tools,
        output_dir=directory, metadata={"workbench_request": asdict(request)})
    compatibility = experiment.validate()["protocol"]["config"].get("compatibility")
    started = time.perf_counter()
    result = experiment.run()
    elapsed = time.perf_counter() - started
    report = {"kind": "session", "subject": subject.identifier, "output": request.output,
              "compatibility": compatibility, "subject_traits": sorted(subject.traits),
              "outputs": [{"condition": condition, "response": response_preview(result.value[index * request.repeats])}
                          for index, condition in enumerate(conditions)],
              "restored_exact": same_output(result.value[0], result.value[-1]) if request.intervention else None,
              "note": "Model responses from a custom session; these are not benchmark scores."}
    return {"report": report, "elapsed_s": elapsed, "artifacts": result.manifest["artifacts"],
            "directory": str(directory)}


def run_benchmark(subject: Any, request: Request, directory: Path,
                  send: Callable, cancelled: Callable) -> dict:
    from brainscore import load_benchmark, score
    from brainscore.experiments import Experiment
    from brainscore_core.compatibility import CompatibilityError

    send({"phase": "running", "condition": "loading_benchmark", "activity": "Loading benchmark"})
    benchmark = load_benchmark(request.benchmark)

    class BenchmarkProtocol:
        identifier = request.benchmark

        def describe(self) -> dict:
            return {"benchmark": request.benchmark, "reset_policy": "before and after the whole benchmark"}

        def validate(self, candidate: Any) -> None:
            pass  # score performs compatibility then memory pre-flight itself.

        def run(self, candidate: Any, context: Any) -> Any:
            if cancelled():
                raise Cancelled("Stopped before benchmark evaluation")
            try:
                candidate.reset()
                with context.repetition("benchmark"):
                    send({"phase": "running", "condition": "benchmark", "activity": "Running benchmark"})
                    return score(candidate, benchmark)
            finally:
                candidate.reset()

    started = time.perf_counter()
    try:
        result = Experiment(subject=subject, protocol=BenchmarkProtocol(), output_dir=directory).run()
    except CompatibilityError as error:
        return {"report": {"kind": "benchmark", "status": "N/A", "reason": str(error),
                           "benchmark": request.benchmark, "subject": subject.identifier},
                "elapsed_s": time.perf_counter() - started, "artifacts": [], "directory": str(directory)}
    value = result.value
    report = {"kind": "benchmark", "status": "complete", "benchmark": request.benchmark,
              "subject": subject.identifier, "score": response_preview(value),
              "subject_traits": sorted(value.attrs.get("subject_traits", ())),
              "trial_memory": value.attrs.get("trial_memory", "none"),
              "note": "Executed by brainscore.score with its compatibility and memory checks; the benchmark protocol is unchanged."}
    return {"report": report, "elapsed_s": time.perf_counter() - started,
            "artifacts": result.manifest["artifacts"], "directory": str(directory)}
