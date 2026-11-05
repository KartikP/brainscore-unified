"""Build the same experiment for both command-line and browser execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import time
from pathlib import Path
from typing import Callable

from .profile import LabProfile

CONDITIONS = ("baseline", "ablation", "random_control", "restored")


class Cancelled(Exception):
    """Cooperative stop at the next repetition boundary."""


@dataclass(frozen=True)
class Plan:
    target: str = "population"
    fraction: int = 25
    activity: bool = False
    behavior: bool = True
    ablation: bool = False
    seed: int = 1729
    repeats: int = 1

    @classmethod
    def parse(cls, value: dict) -> Plan:
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Use only the documented experiment settings")
        plan = cls(**value)
        if not isinstance(plan.target, str) or not 0 < len(plan.target) <= 200:
            raise ValueError("Select a target")
        if type(plan.fraction) is not int or plan.fraction not in (25, 50, 75, 100):
            raise ValueError("Ablation fraction must be 25, 50, 75, or 100")
        if any(type(flag) is not bool for flag in (plan.activity, plan.behavior, plan.ablation)):
            raise ValueError("Tool settings must be booleans")
        if type(plan.seed) is not int or not 0 <= plan.seed <= 2**32 - 1:
            raise ValueError("Seed must be an unsigned 32-bit integer")
        if type(plan.repeats) is not int or not 1 <= plan.repeats <= 100:
            raise ValueError("Repetitions must be between 1 and 100")
        return plan


def selections(profile: LabProfile, plan: Plan) -> dict:
    target = next((target for target in profile.targets if target.name == plan.target), None)
    if target is None:
        if plan.activity or plan.ablation:
            raise ValueError("This profile does not expose the selected target")
        return {"selected": [], "control": [], "overlap": 0, "layer": None}
    if target.units < 1:
        raise ValueError("Targets must declare a positive unit count")
    count = max(1, target.units * plan.fraction // 100)
    chosen = list(range(count))
    remaining = list(range(count, target.units))
    # Prefer a disjoint, size-matched control. Report any unavoidable overlap.
    pool = remaining if len(remaining) >= count else list(range(target.units))
    control = sorted(random.Random(plan.seed).sample(pool, count))
    return {"selected": chosen, "control": control,
            "overlap": len(set(chosen) & set(control)), "layer": target.layer}


def build_experiment(profile: LabProfile, plan: Plan, directory: Path,
                     observer: object | None = None) -> object:
    from brainscore.experiments import (
        Ablate, Experiment, RecordActivity, RecordInputsOutputs, SessionProtocol,
    )
    from brainscore_core.events import Selection

    picked = selections(profile, plan)
    if (plan.activity or plan.ablation) and profile.instrumentation is None:
        raise ValueError("The profile has no instrumentation provider")
    # No synthetic model-kind checks or changes to the Subject interface.
    inputs = set(profile.input_channels)
    missing = set(profile.output_channels) - set(profile.subject.out_channels)
    required = set(profile.subject.required_channels) - inputs
    if not (inputs & set(profile.subject.in_channels)) or missing or required:
        raise ValueError(f"Incompatible declarations: outputs={sorted(missing)}, required={sorted(required)}")
    if inputs - set(profile.subject.in_channels):
        # The installed SessionProtocol currently requires a full input match.
        # Refuse explicitly rather than claim a partial run that it cannot make.
        raise ValueError("This runner version requires a full input match; use a compatible profile")

    tools = []
    if plan.behavior or plan.activity:
        tools.append(RecordInputsOutputs(max_bytes=256 * 1024 * 1024))
    if observer is not None:
        tools.append(observer)
    if plan.ablation:
        tools.append(Ablate([Selection(layer=picked["layer"], indices=picked["selected"])],
                           conditions=["ablation"]))
        control = Ablate([Selection(layer=picked["layer"], indices=picked["control"])],
                        conditions=["random_control"])
        control.name = "random_control"
        tools.append(control)
    if plan.activity:
        tools.append(RecordActivity([picked["layer"]]))
    return Experiment(
        subject=profile.subject,
        protocol=SessionProtocol(
            identifier="virtual-lab-comparison-v1", session_factory=profile.make_session,
            conditions=CONDITIONS if plan.ablation else ("baseline",),
            repetitions=tuple(range(plan.repeats)), input_channels=profile.input_channels,
            output_channels=profile.output_channels,
        ),
        tools=tools, instrumentation=profile.instrumentation,
        output_dir=directory,
        metadata={"lab_plan": asdict(plan), "selection": picked,
                  "provenance": profile.provenance or {}},
    )


def progress_tool(send: Callable[[dict], None], cancelled: Callable[[], bool]) -> object:
    from brainscore.experiments import Tool

    class Progress(Tool):
        name = "lab_progress"

        def __init__(self) -> None:
            self.last_condition = None
            self.last_sent = 0.0

        def on_event(self, event: dict) -> None:
            kind = event["kind"]
            if kind == "repetition_start":
                if cancelled():
                    raise Cancelled("Stopped between repetitions; the current repetition was allowed to finish")
                now = time.monotonic()
                if event["condition"] != self.last_condition or now - self.last_sent >= 0.5:
                    self.last_condition, self.last_sent = event["condition"], now
                    send({"phase": "running", "condition": event["condition"],
                          "repetition": event["repetition_id"]})

    return Progress()


def execute(profile: LabProfile, plan: Plan, directory: Path,
            send: Callable[[dict], None] | None = None,
            cancelled: Callable[[], bool] = lambda: False) -> dict:
    observer = progress_tool(send, cancelled) if send is not None else None
    started = time.perf_counter()
    result = build_experiment(profile, plan, directory, observer).run()
    elapsed = time.perf_counter() - started
    report = profile.summarize(result)
    return {"report": report, "elapsed_s": elapsed,
            "selection": selections(profile, plan),
            "artifacts": result.manifest["artifacts"], "directory": str(directory)}
