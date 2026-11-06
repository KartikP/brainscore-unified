"""Optional app-local dependency paths, without changing editable installs."""
import json
import os
from pathlib import Path
import sys


def configure_runtime(core: Path | None = None) -> None:
    configuration = Path(__file__).resolve().parents[2] / ".lab-runtime.json"
    if configuration.is_file():
        settings = json.loads(configuration.read_text())
        for path in reversed(settings.get("python_paths", [])):
            resolved = str((configuration.parent / path).resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
        for name, value in settings.get("environment", {}).items():
            os.environ.setdefault(name, str((configuration.parent / value).resolve()))
    if core:
        sys.path.insert(0, str(core.resolve()))
