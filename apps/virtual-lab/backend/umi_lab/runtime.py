"""Optional app-local dependency paths, without changing editable installs."""
import json
import os
from pathlib import Path
import sys


def configure_runtime(core: Path | None = None) -> None:
    from .frontend import source_root
    root = source_root()
    configuration = root / ".lab-runtime.json" if root is not None else None
    if configuration is not None and configuration.is_file():
        settings = json.loads(configuration.read_text())
        for path in reversed(settings.get("python_paths", [])):
            resolved = str((configuration.parent / path).resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
        for name, value in settings.get("environment", {}).items():
            os.environ.setdefault(name, str((configuration.parent / value).resolve()))
    if core:
        sys.path.insert(0, str(core.resolve()))
