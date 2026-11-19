"""Check installation metadata without importing model libraries or weights."""
from __future__ import annotations

from importlib import metadata, util
import sys

from .frontend import frontend_directory


def setup_report() -> tuple[str, int]:
    from packaging.requirements import Requirement

    lines = [f"Python: {sys.executable}"]
    problems = 0
    if sys.version_info[:2] != (3, 11):
        lines.append("ERROR: UMI v2 requires Python 3.11")
        problems += 1
    try:
        lines.append(f"Frontend: {frontend_directory()}")
    except ValueError as error:
        lines.append(f"ERROR: {error}")
        problems += 1
    for module in ("brainscore", "brainscore_core", "brainscore_vision", "brainscore_language", "torch", "transformers"):
        if util.find_spec(module) is None:
            lines.append(f"ERROR: Missing module {module}")
            problems += 1
    try:
        version = metadata.version("brainscore")
        lines.append(f"Brain-Score: {version}")
        if version != "0.3.0rc1":
            lines.append("ERROR: This lab requires brainscore==0.3.0rc1")
            problems += 1
    except metadata.PackageNotFoundError:
        lines.append("ERROR: Install the coordinated UMI v2 environment before the lab")
        problems += 1
    checked = set()
    for package in ("brainscore", "brainscore_core", "brainscore_vision", "brainscore_language"):
        try:
            requirements = metadata.requires(package) or ()
        except metadata.PackageNotFoundError:
            lines.append(f"ERROR: Missing UMI distribution {package}")
            problems += 1
            continue
        for text in requirements:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            key = (requirement.name.lower().replace("_", "-"), str(requirement.specifier), requirement.url)
            if key in checked:
                continue
            checked.add(key)
            try:
                installed = metadata.version(requirement.name)
                valid = requirement.specifier.contains(installed, prereleases=True)
            except metadata.PackageNotFoundError:
                installed, valid = "absent", False
            if not valid:
                lines.append(f"ERROR: {requirement.name} is {installed}; requires {requirement.specifier}")
                problems += 1
    if problems:
        lines.append("Activate or repair the pinned UMI environment, then run brainscore-lab --check again.")
    else:
        lines.append("Setup metadata and frontend checks passed. No model was loaded.")
    lines.append("This check does not test binary imports, weights or datasets. Use python -m brainscore.doctor for import checks.")
    return "\n".join(lines), problems
