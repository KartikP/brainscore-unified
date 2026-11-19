"""Locate and verify the frontend shipped with the lab package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def source_root() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    return root if (root / "lab.py").is_file() and (root / "frontend/src").is_dir() else None


def verify_files(root: Path, hashes: dict) -> None:
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("The frontend manifest has no file hashes")
    for name, expected in hashes.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Invalid frontend manifest path")
        filename = root.joinpath(*relative.parts).resolve()
        if not filename.is_relative_to(root.resolve()) or not filename.is_file():
            raise ValueError(f"Frontend file is missing: {name}")
        # Git may convert text files to CRLF on a Windows source checkout.
        if hashlib.sha256(filename.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError(f"Frontend file differs from the prebuilt bundle: {name}")


def frontend_directory() -> Path:
    directory = Path(__file__).resolve().parent / "static"
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("version") != 1 or not {"index.html", "main.js", "style.css"}.issubset(manifest.get("assets", {})):
            raise ValueError("Invalid frontend manifest")
        verify_files(directory, manifest["assets"])
        root = source_root()
        if root is not None:
            sources = manifest["sources"]
            actual = {str(path.relative_to(root / "frontend")).replace("\\", "/")
                      for path in (root / "frontend/src").glob("*.ts")}
            recorded = {name for name in sources if name.startswith("src/")}
            if actual != recorded:
                raise ValueError("Frontend source files changed after the bundle was built")
            verify_files(root / "frontend", sources)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"The prebuilt frontend is unavailable or stale: {error}. "
                         "Install a complete lab package. Developers: run npm ci and npm run bundle in frontend/.") from error
    return directory
