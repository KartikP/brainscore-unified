"""The installed lab starts without Node and rejects stale frontend assets."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from umi_lab import frontend


def bundle(tmp_path, monkeypatch):
    package = tmp_path / "backend/umi_lab"
    directory = package / "static"
    directory.mkdir(parents=True)
    hashes = {}
    for name in ("index.html", "main.js", "style.css"):
        content = f"test {name}\n".encode()
        (directory / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    manifest = {"version": 1, "assets": hashes, "sources": {}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(frontend, "__file__", str(package / "frontend.py"))
    return directory, manifest


def test_complete_bundle_works_without_source_or_node(tmp_path, monkeypatch):
    directory, _ = bundle(tmp_path, monkeypatch)
    assert frontend.frontend_directory() == directory
    assert frontend.source_root() is None
    (directory / "main.js").write_text("different")
    with pytest.raises(ValueError, match="differs"):
        frontend.frontend_directory()


def test_source_changes_require_a_new_bundle(tmp_path, monkeypatch):
    directory, manifest = bundle(tmp_path, monkeypatch)
    (tmp_path / "lab.py").touch()
    source = tmp_path / "frontend/src"
    source.mkdir(parents=True)
    (source / "main.ts").write_text("let value = 1;")
    manifest["sources"] = {"src/main.ts": hashlib.sha256((source / "main.ts").read_bytes()).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    assert frontend.frontend_directory() == directory
    (source / "extra.ts").touch()
    with pytest.raises(ValueError, match="source files changed"):
        frontend.frontend_directory()
    (source / "extra.ts").unlink()
    (source / "main.ts").write_text("let value = 2;")
    with pytest.raises(ValueError, match="differs"):
        frontend.frontend_directory()


def test_git_line_endings_do_not_invalidate_a_bundle(tmp_path, monkeypatch):
    directory, _ = bundle(tmp_path, monkeypatch)
    for name in ("index.html", "main.js", "style.css"):
        filename = directory / name
        filename.write_bytes(filename.read_bytes().replace(b"\n", b"\r\n"))
    assert frontend.frontend_directory() == directory


def test_invalid_bundle_paths_and_missing_files_fail_explicitly(tmp_path, monkeypatch):
    directory, manifest = bundle(tmp_path, monkeypatch)
    manifest["assets"]["../escape.js"] = "0" * 64
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest path"):
        frontend.frontend_directory()
    del manifest["assets"]["../escape.js"]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "main.js").unlink()
    with pytest.raises(ValueError, match="missing"):
        frontend.frontend_directory()


def test_setup_metadata_checks_versions_and_markers_without_imports(tmp_path, monkeypatch):
    from umi_lab import checks
    monkeypatch.setattr(checks, "frontend_directory", lambda: tmp_path)
    monkeypatch.setattr(checks.util, "find_spec", lambda name: object())
    versions = {"brainscore": "0.3.0rc1", "scikit-learn": "1.7.2"}
    monkeypatch.setattr(checks.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(checks.metadata, "requires", lambda name: [
        "scikit-learn>=1.7,<1.8", 'optional-plugin; extra == "test"'])
    text, problems = checks.setup_report()
    assert problems == 0 and "No model was loaded" in text
    versions["scikit-learn"] = "1.5.2"
    text, problems = checks.setup_report()
    assert problems == 1 and "requires <1.8,>=1.7" in text
    monkeypatch.setattr(checks.util, "find_spec", lambda name: None if name == "torch" else object())
    text, problems = checks.setup_report()
    assert problems == 2 and "Missing module torch" in text


def test_check_command_propagates_failure_without_starting_a_worker(monkeypatch, capsys):
    from umi_lab import __main__, checks
    monkeypatch.setattr(checks, "setup_report", lambda: ("fixture error", 1))
    assert __main__.main(["--check"]) == 1
    assert "fixture error" in capsys.readouterr().out


def test_check_includes_peer_dependencies(tmp_path, monkeypatch):
    from umi_lab import checks
    monkeypatch.setattr(checks, "frontend_directory", lambda: tmp_path)
    monkeypatch.setattr(checks.util, "find_spec", lambda name: object())
    def version(name):
        if name == "datasets":
            raise checks.metadata.PackageNotFoundError(name)
        return "0.3.0rc1"
    monkeypatch.setattr(checks.metadata, "version", version)
    monkeypatch.setattr(checks.metadata, "requires", lambda name: ["datasets>=2.14"] if name == "brainscore_language" else [])
    text, problems = checks.setup_report()
    assert problems == 1 and "datasets is absent" in text


@pytest.mark.parametrize("no_browser", [False, True])
def test_launcher_uses_packaged_assets_and_closes_its_worker(tmp_path, monkeypatch, no_browser):
    import threading
    import webbrowser
    from umi_lab import __main__, checks, server
    calls = []
    monkeypatch.setattr(checks, "setup_report", lambda: ("ok", 0))
    monkeypatch.setattr(frontend, "frontend_directory", lambda: tmp_path)
    class Controller:
        def __init__(self, *args):
            calls.append("worker")
        def close(self):
            calls.append("worker closed")
    class Server:
        server_port = 45678
        def serve_forever(self):
            raise KeyboardInterrupt
        def server_close(self):
            calls.append("server closed")
    def make_server(controller, directory, port):
        assert directory == tmp_path and port == 0
        return Server()
    monkeypatch.setattr(server, "LabController", Controller)
    monkeypatch.setattr(server, "make_server", make_server)
    monkeypatch.setattr(webbrowser, "open_new_tab", calls.append)
    monkeypatch.setattr(threading, "Thread", lambda target, **kwargs: SimpleNamespace(start=target))
    args = ["--port", "0", "--output", str(tmp_path / "runs")]
    __main__.main(args + (["--no-browser"] if no_browser else []))
    assert ("http://127.0.0.1:45678" in calls) is not no_browser
    assert calls[-2:] == ["server closed", "worker closed"]
