"""Data-only configuration of the documented TaskProtocol experiment workflow."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Callable


def fields(value: Any, allowed: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f"Unknown or invalid {name} settings")
    return value


def text(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > 8192 or (not empty and not value.strip()):
        raise ValueError(f"{name} must be {'optional ' if empty else 'nonempty '}text, at most 8192 characters")
    return value.strip()


def scale(value: Any) -> dict:
    value = fields(value, {"low", "high", "step"}, "response scale")
    low, high, step = value.get("low", 1), value.get("high", 7), value.get("step", 1)
    for number in (low, high, step):
        if number is not None and (type(number) not in (int, float) or not math.isfinite(number)):
            raise ValueError("Scale bounds and step must be finite numbers")
    if low is None or high is None or low >= high or (step is not None and step <= 0):
        raise ValueError("A scale needs low < high and a positive step (or null)")
    return {"low": low, "high": high, "step": step}


@dataclass(frozen=True)
class ProtocolSpec:
    identifier: str
    task: dict
    measurements: dict
    conditions: list[dict]
    repetitions: int
    required_traits: list[str]

    @classmethod
    def parse(cls, raw: Any) -> ProtocolSpec:
        value = fields(raw, {"identifier", "task", "measurements", "conditions", "repetitions", "required_traits"}, "protocol")
        identifier = text(value.get("identifier", "my-protocol"), "Protocol name")
        if len(identifier) > 200:
            raise ValueError("Protocol names must be at most 200 characters")
        task = dict(fields(value.get("task", {}), {"kind", "instruction", "labels", "probabilities", "confidence", "scale", "path", "readout_fit", "present", "trial_memory"}, "task"))
        kind = task.get("kind", "passive")
        if kind not in ("passive", "choice", "rating", "text", "legacy"):
            raise ValueError("Choose plain viewing, choice, rating, text or legacy choices")
        task["kind"] = kind
        task["instruction"] = text(task.get("instruction", ""), "Task instruction", empty=True)
        labels = task.get("labels", [])
        if not isinstance(labels, list) or len(labels) > 100 or any(not isinstance(label, str) or not label.strip() or len(label) > 200 for label in labels):
            raise ValueError("Provide at most 100 nonempty response labels")
        labels = [label.strip() for label in labels]
        if kind in ("choice", "legacy") and (len(labels) < 2 or len({label.casefold() for label in labels}) != len(labels)):
            raise ValueError("Choice tasks need at least two distinct response labels")
        task["labels"] = labels
        if type(task.get("probabilities", False)) is not bool:
            raise ValueError("Probabilities must be a checkbox")
        task["probabilities"] = task.get("probabilities", False)
        if task.get("confidence") is not None:
            if kind != "choice":
                raise ValueError("Reported confidence is an option on a choice task")
            task["confidence"] = scale(task["confidence"])
        if kind == "rating":
            task["scale"] = scale(task.get("scale", {}))
        if task.get("path", "auto") not in ("auto", "generation", "likelihood", "readout"):
            raise ValueError("Response path must be auto, generation, likelihood or readout")
        task["path"] = task.get("path", "auto")
        if task.get("readout_fit", "current") not in ("current", "intact"):
            raise ValueError("Fit the readout on the current or intact network")
        task["readout_fit"] = task.get("readout_fit", "current")
        if task.get("trial_memory", "none") not in ("none", "history"):
            raise ValueError("Trial memory must be none (each trial starts fresh) or history (the subject keeps what it took in and answered on earlier trials)")
        task["trial_memory"] = task.get("trial_memory", "none")
        present = task.get("present", [])
        if not isinstance(present, list) or len(present) > 32 or any(not isinstance(part, str) or not part.strip() or len(part) > 200 for part in present):
            raise ValueError("Trial part order must be a list of named parts")
        task["present"] = [part.strip() for part in present]
        measurements = dict(fields(value.get("measurements", {}), {"behavior", "regions", "replay"}, "recording"))
        for name in ("behavior", "replay"):
            if type(measurements.get(name, False)) is not bool:
                raise ValueError(f"{name} must be a checkbox")
            measurements[name] = measurements.get(name, False)
        regions = measurements.get("regions", [])
        if not isinstance(regions, list) or len(regions) > 16 or any(not isinstance(region, str) or not region.strip() or len(region) > 200 for region in regions):
            raise ValueError("Choose at most 16 declared neural regions")
        measurements["regions"] = list(dict.fromkeys(regions))
        if kind == "passive" and measurements["behavior"]:
            raise ValueError("Plain viewing has no behavioral answer; remove behavior recording or choose a response")
        if not measurements["behavior"] and not measurements["regions"]:
            raise ValueError("Choose behavior or at least one neural recording")
        conditions = value.get("conditions", [{"name": "baseline"}])
        if not isinstance(conditions, list) or not 1 <= len(conditions) <= 12:
            raise ValueError("Provide 1 to 12 conditions")
        parsed, names = [], set()
        for item in conditions:
            item = fields(item, {"name", "intervention"}, "condition")
            name = text(item.get("name"), "Condition name")
            if len(name) > 100 or name in names:
                raise ValueError("Condition names must be distinct and at most 100 characters")
            names.add(name)
            change = item.get("intervention")
            if change is not None:
                change = dict(fields(change, {"target", "indices", "axis", "operation", "value"}, "intervention"))
                change["target"] = text(change.get("target"), "Intervention address")
                if len(change["target"]) > 512:
                    raise ValueError("Intervention addresses must be at most 512 characters")
                if change.get("operation", "zero") not in ("zero", "scale", "drive"):
                    raise ValueError("Intervention operation must be zero, scale or drive")
                change["operation"] = change.get("operation", "zero")
                indices = change.get("indices")
                if indices is not None and (not isinstance(indices, list) or not indices or len(indices) > 10000 or any(type(index) is not int or index < 0 for index in indices)):
                    raise ValueError("Unit indices must be a nonempty list of nonnegative integers, or null for the whole target")
                axis = change.get("axis", -1)
                if axis is not None and type(axis) is not int:
                    raise ValueError("Unit axis must be an integer or null for flattened units")
                change["axis"] = axis
                amount = change.get("value", 0)
                if type(amount) not in (int, float) or not math.isfinite(amount):
                    raise ValueError("Intervention strength must be finite")
                change["value"] = amount
            parsed.append({"name": name, "intervention": change})
        repetitions = value.get("repetitions", 1)
        if type(repetitions) is not int or not 1 <= repetitions <= 100:
            raise ValueError("Provide 1 to 100 independent repetitions")
        traits = value.get("required_traits", [])
        if not isinstance(traits, list) or len(traits) > 16 or any(not isinstance(trait, str) or not trait or len(trait) > 200 for trait in traits):
            raise ValueError("Required traits must be a list of declared trait names")
        return cls(identifier, task, measurements, parsed, repetitions, traits)


def build_protocol(subject: Any, request: Any, uploads: Path, directory: Path | None = None,
                   send: Callable | None = None, cancelled: Callable = lambda: False) -> tuple[Any, ProtocolSpec, list[str]]:
    from brainscore import load_stimulus_set
    from brainscore.experiments import Experiment, TaskProtocol, RecordBehavior, RecordNeural, RecordInputsOutputs, Intervene
    from brainscore_core import TaskContext, Selection, Perturbation, StateChange
    from brainscore_core.responses import Choice, Scale, Text, trial_prompt
    from .workbench import stimulus_table
    from .experiment import progress_tool

    spec = ProtocolSpec.parse(request.protocol)
    stimuli = load_stimulus_set(request.stimulus_set) if request.stimulus_set else stimulus_table(request.rows, uploads)
    fitting_rows = [{**row, "stimulus_id": row.get("stimulus_id", f"fitting-{index}")}
                    for index, row in enumerate(request.fitting_rows)]
    fitting = stimulus_table(fitting_rows, uploads) if fitting_rows else None
    settings = spec.task
    kind = settings["kind"]
    response = None
    if kind == "choice":
        confidence = Scale(**settings["confidence"]) if settings.get("confidence") else None
        response = Choice(settings["labels"], probabilities=settings["probabilities"], confidence=confidence)
    elif kind == "rating":
        response = Scale(**settings["scale"])
    elif kind == "text":
        response = Text()
    if kind != "passive" and settings["path"] == "readout":
        if fitting is None or "label" not in fitting.columns:
            raise ValueError("A fitted readout needs a separate fitting table with a label column")
        if any(value is None or not str(value).strip() for value in fitting["label"]):
            raise ValueError("Fill the label for every fitting example")
        if kind in ("choice", "legacy"):
            fitted_labels = set(map(str, fitting["label"]))
            if len(fitted_labels) < 2 or fitted_labels - set(settings["labels"]):
                raise ValueError("Choice fitting examples need at least two classes drawn from the response labels")
    task = TaskContext(task_type="passive" if kind == "passive" else "probabilities" if kind == "legacy" else None,
                       instruction=settings["instruction"] or None, response=response,
                       label_set=(settings["labels"] or None) if kind != "passive" else None, fitting_stimuli=fitting,
                       prefer_path=settings["path"], readout_fit=settings["readout_fit"],
                       present=tuple(settings["present"]) or None, trial_memory=settings["trial_memory"])
    if kind != "passive":
        for _, row in stimuli.iterrows():
            trial_prompt(task, row)  # check explicit multimodal ordering without a forward pass
    notes = []
    if fitting is not None:
        overlap = set(map(str, fitting["stimulus_id"])) & set(map(str, stimuli["stimulus_id"]))
        if overlap:
            notes.append(f"Fitting and test tables share {len(overlap)} stimulus IDs; this is not a held-out evaluation.")
    tools = []
    if spec.measurements["regions"]:
        tools.append(RecordNeural(spec.measurements["regions"]))
    if spec.measurements["behavior"]:
        tools.append(RecordBehavior())
    if spec.measurements["replay"]:
        tools.append(RecordInputsOutputs())
    if send is not None:
        activity = ("Viewing stimuli" if kind == "passive" else
                    "Viewing stimuli and generating responses" if settings["path"] != "readout" and kind == "text"
                    else "Processing stimuli and computing responses")
        tools.append(progress_tool(send, cancelled, activity=activity))
    for index, condition in enumerate(spec.conditions):
        change = condition["intervention"]
        if change is None:
            continue
        operation = change["operation"]
        expected = "lesion" if operation == "zero" else "stimulation"
        if expected not in subject.in_channels or "perturbation" not in subject.out_channels:
            raise ValueError(f"This subject does not declare {expected} input and perturbation output")
        state_change = StateChange(kind="ablation" if operation == "zero" else "stimulation",
            target=Selection(layer=change["target"], indices=change.get("indices"), axis=change["axis"]),
            perturbation=Perturbation(kind=operation, scale=change["value"], amount=change["value"]))
        tools.append(Intervene(state_change, conditions=[condition["name"]], name=f"condition_{index}"))
    protocol = TaskProtocol(stimuli, task, identifier=spec.identifier,
                            conditions=[condition["name"] for condition in spec.conditions],
                            repetitions=tuple(range(spec.repetitions)), required_traits=spec.required_traits)
    experiment = Experiment(subject=subject, protocol=protocol, tools=tools, output_dir=directory)
    return experiment, spec, notes


def validate_protocol(subject: Any, request: Any, uploads: Path) -> dict:
    experiment, spec, notes = build_protocol(subject, request, uploads)
    plan = experiment.validate()
    config = plan["protocol"]["config"]
    return {"identifier": spec.identifier, "subject": subject.identifier,
            "subject_traits": sorted(subject.traits), "trial_memory": spec.task["trial_memory"],
            "stimuli": len(experiment.protocol.stimuli), "conditions": [condition["name"] for condition in spec.conditions],
            "repetitions": spec.repetitions, "channels": config["output_channels"],
            "compatibility": config.get("compatibility"), "tools": [tool["name"] for tool in plan["tools"]],
            "reset_policy": config["reset_policy"], "notes": notes,
            "presentations": len(experiment.protocol.stimuli) * len(spec.conditions) * spec.repetitions}


def run_protocol(subject: Any, request: Any, directory: Path, uploads: Path,
                 send: Callable, cancelled: Callable) -> dict:
    experiment, spec, notes = build_protocol(subject, request, uploads, directory, send, cancelled)
    plan = experiment.validate()
    started = time.perf_counter()
    result = experiment.run()
    elapsed = time.perf_counter() - started
    config = plan["protocol"]["config"]
    return {"report": {"kind": "protocol", "subject": subject.identifier, "identifier": spec.identifier,
                       "subject_traits": sorted(subject.traits), "trial_memory": spec.task["trial_memory"],
                       "conditions": [condition["name"] for condition in spec.conditions],
                       "repetitions": spec.repetitions, "stimuli": len(experiment.protocol.stimuli),
                       "channels": config["output_channels"], "compatibility": config.get("compatibility"),
                       "notes": notes, "restored_exact": None,
                       "note": "Each stimulus was presented once per condition and repetition. Recordings describe that same presentation; these are not benchmark scores."},
            "elapsed_s": elapsed, "artifacts": result.manifest["artifacts"], "directory": str(directory)}
