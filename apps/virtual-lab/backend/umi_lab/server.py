"""Loopback-only static/API server; no model or tensor imports in this process."""
from __future__ import annotations

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import mimetypes
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import secrets
import shutil
import tempfile
import threading
import time
from urllib.parse import urlsplit, unquote, parse_qs

from .experiment import Plan
from .worker import DEFAULT_FACTORY, run_worker

ACTIVE = {"queued", "running", "trial_finished", "cancelling"}


class LabController:
    def __init__(self, root: Path, factory: str = DEFAULT_FACTORY) -> None:
        self.factory = factory
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.state = {"phase": "loading", "profile": None, "run": None}
        self.validation_fingerprint: str | None = None
        self.catalog = {"models": [], "benchmarks": [], "stimulus_sets": []}
        self.runs: dict[str, Path] = {}
        context = mp.get_context("spawn")
        self.commands = context.Queue(maxsize=1)
        self.messages = context.Queue(maxsize=8)
        self.stop = context.Event()
        self.closed = threading.Event()
        self.process = context.Process(target=run_worker,
                                       args=(factory, str(self.root), self.commands,
                                             self.messages, self.stop), daemon=True)
        self.process.start()
        self.listener = threading.Thread(target=self._listen, daemon=True)
        self.listener.start()

    def _listen(self) -> None:
        while not self.closed.is_set():
            try:
                message = self.messages.get(timeout=0.2)
            except Empty:
                if not self.process.is_alive():
                    with self.lock:
                        if self.state["phase"] != "failed":
                            self.state["phase"] = "failed"
                            self.state["error"] = "Model worker exited; restart the lab to reload the subject"
                    return
                continue
            with self.lock:
                if "catalog" in message:
                    self.catalog[message["catalog"]].extend(message["items"])
                elif "id" not in message:
                    self.state.update(message)
                elif self.state["run"] and message["id"] == self.state["run"]["id"]:
                    cancelling = self.state["run"]["phase"] == "cancelling"
                    if cancelling and message["phase"] in ACTIVE:
                        message["phase"] = "cancelling"
                    self.state["run"].update(message)

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def submit(self, data: dict, *, telemetry: bool = True) -> str:
        plan = Plan.parse(data)
        with self.lock:
            if self.state["phase"] != "ready":
                raise ValueError("Wait for the subject to finish loading")
            if self.state["run"] and self.state["run"]["phase"] in ACTIVE:
                raise ValueError("An experiment is already running")
            identifier = secrets.token_hex(16)
            self.stop.clear()
            self.runs[identifier] = self.root / identifier
            self.state["run"] = {"id": identifier, "phase": "queued", "plan": asdict(plan),
                                  "started_at": time.time()}
            self.commands.put_nowait({"id": identifier, "plan": asdict(plan), "telemetry": telemetry})
            return identifier

    def load(self, data: dict) -> None:
        if not isinstance(data, dict) or set(data) != {"identifier"}:
            raise ValueError("Select a registered model identifier")
        identifier = data["identifier"]
        if not isinstance(identifier, str) or not 0 < len(identifier) <= 300:
            raise ValueError("Enter a model identifier")
        with self.lock:
            if self.state["phase"] not in ("ready", "load_failed") or (self.state["run"] and self.state["run"]["phase"] in ACTIVE):
                raise ValueError("Wait for the current operation before changing models")
            self.state.update(phase="loading", error=None, validation=None)
            self.validation_fingerprint = None
            self.commands.put_nowait({"kind": "load", "identifier": identifier})

    def submit_request(self, data: dict) -> str:
        from .workbench import Request
        request = Request.parse(data)
        with self.lock:
            if self.state["phase"] != "ready":
                raise ValueError("Load a subject first")
            if self.state["run"] and self.state["run"]["phase"] in ACTIVE:
                raise ValueError("An experiment is already running")
            if request.mode == "protocol":
                validation = self.state.get("validation") or {}
                if validation.get("phase") != "valid" or self.validation_fingerprint != self._fingerprint(request):
                    raise ValueError("Validate this exact protocol with the current subject before running")
            identifier = secrets.token_hex(16)
            self.stop.clear()
            self.runs[identifier] = self.root / identifier
            self.state["run"] = {"id": identifier, "phase": "queued", "request": asdict(request),
                                  "started_at": time.time()}
            self.commands.put_nowait({"kind": "workbench", "id": identifier, "request": asdict(request)})
            return identifier

    def _fingerprint(self, request: object) -> str:
        payload = {"subject": self.state["profile"]["identifier"], "request": asdict(request)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def validate_request(self, data: dict) -> str:
        from .workbench import Request
        request = Request.parse(data)
        if request.mode != "protocol":
            raise ValueError("The protocol builder validates TaskProtocol configurations")
        with self.lock:
            if self.state["phase"] != "ready" or (self.state["run"] and self.state["run"]["phase"] in ACTIVE):
                raise ValueError("Wait until the subject is ready and the current run has finished")
            identifier = secrets.token_hex(16)
            self.validation_fingerprint = self._fingerprint(request)
            self.state.update(phase="validating", validation={"id": identifier, "phase": "pending"})
            self.commands.put_nowait({"kind": "validate", "validation_id": identifier, "request": asdict(request)})
            return identifier

    def cancel(self) -> None:
        with self.lock:
            if self.state["run"] and self.state["run"]["phase"] in ACTIVE:
                self.stop.set()
                self.state["run"]["phase"] = "cancelling"

    def close(self) -> None:
        self.stop.set()
        try:
            self.commands.put_nowait(None)
        except Exception:
            pass
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.closed.set()
        self.listener.join(timeout=1)
        self.commands.close()
        self.messages.close()


def make_server(controller: LabController, frontend: Path, port: int = 8765) -> ThreadingHTTPServer:
    frontend = frontend.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def _trusted(self, mutation: bool = False, upload: bool = False) -> bool:
            host = self.headers.get("Host", "")
            expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            allowed = host in expected and (not origin or origin == f"http://{host}")
            if mutation:
                allowed = (allowed and self.headers.get("Content-Type") == ("application/octet-stream" if upload else "application/json")
                           and secrets.compare_digest(self.headers.get("X-Lab-Token", ""), controller.token))
            if not allowed:
                self._json({"error": "Request must originate from this local lab"}, 403)
            return allowed

        def _headers(self, mime: str, length: int, status: int = 200, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            for name, value in (extra or {}).items():
                self.send_header(name, str(value))
            self.end_headers()

        def _json(self, data: dict, status: int = 200) -> None:
            body = json.dumps(data, allow_nan=False).encode()
            self._headers("application/json", len(body), status)
            self.wfile.write(body)

        def do_GET(self) -> None:
            if not self._trusted():
                return
            path = urlsplit(self.path).path
            if path == "/api/state":
                self._json({**controller.snapshot(), "token": controller.token})
                return
            if path == "/api/catalog":
                with controller.lock:
                    available = dict(controller.catalog)
                self._json(available)
                return
            if path.startswith("/api/runs/"):
                parts = path.strip("/").split("/")
                if len(parts) != 4 or parts[2] not in controller.runs:
                    self._json({"error": "Unknown run"}, 404)
                    return
                directory = controller.runs[parts[2]]
                if parts[3] in ("replay", "media"):
                    state = controller.snapshot()["run"]
                    if state and state["id"] == parts[2] and state["phase"] in ACTIVE:
                        self._json({"error": "Wait for the recording to finish"}, 409)
                        return
                    try:
                        from .replay import replay_manifest, replay_media, byte_range
                        query = parse_qs(urlsplit(self.path).query)
                        allowed = {"condition", "repetition"} if parts[3] == "replay" else {"stimulus", "kind"}
                        if set(query) - allowed or any(len(values) != 1 for values in query.values()):
                            raise ValueError("Unknown or repeated replay settings")
                        settings = {key: values[0] if key == "kind" else int(values[0]) for key, values in query.items()}
                        if parts[3] == "replay":
                            self._json(replay_manifest(directory, **settings))
                        else:
                            filename, mime = replay_media(directory, **settings)
                            size = filename.stat().st_size
                            try:
                                start, stop, status = byte_range(self.headers.get("Range"), size)
                            except ValueError:
                                self._headers(mime, 0, 416, {"Content-Range": f"bytes */{size}"})
                                return
                            headers = {"Accept-Ranges": "bytes"}
                            if status == 206:
                                headers["Content-Range"] = f"bytes {start}-{stop}/{size}"
                            self._headers(mime, stop - start + 1, status, headers)
                            with filename.open("rb") as stream:
                                stream.seek(start)
                                remaining = stop - start + 1
                                while remaining:
                                    chunk = stream.read(min(65536, remaining))
                                    if not chunk:
                                        break
                                    self.wfile.write(chunk)
                                    remaining -= len(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    except (ValueError, OSError, KeyError, IndexError, TypeError) as error:
                        self._json({"error": str(error)}, 400)
                    return
                if parts[3] == "observations":
                    try:
                        from .observations import observation_page
                        query = parse_qs(urlsplit(self.path).query)
                        if set(query) - {"index", "channel"} or any(len(values) != 1 for values in query.values()):
                            raise ValueError("Choose one recorded repetition index")
                        index = int(query.get("index", ["0"])[0])
                        channel = query.get("channel", [None])[0]
                        self._json({"outputs": observation_page(directory / "experiment", index=index, channel=channel)})
                    except (ValueError, OSError, KeyError, IndexError, TypeError) as error:
                        self._json({"error": str(error)}, 400)
                    return
                if parts[3] in ("activation", "temporal"):
                    state = controller.snapshot()["run"]
                    if state and state["id"] == parts[2] and state["phase"] in ACTIVE:
                        self._json({"error": "Wait for the recording to finish"}, 409)
                        return
                    try:
                        from .activation_data import activation_page, temporal_page
                        query = parse_qs(urlsplit(self.path).query)
                        temporal = parts[3] == "temporal"
                        allowed = {"index", "presentation", "offset", "limit", "channel"}
                        allowed |= {"time_offset", "time_limit", "neurons", "labels"} if temporal else {"stats", "labels", "time_bin"}
                        if set(query) - allowed or any(len(values) != 1 for values in query.values()):
                            raise ValueError("Unknown or repeated activation-view settings")
                        settings = {key: int(values[0]) for key, values in query.items() if key not in ("channel", "neurons")}
                        if "channel" in query:
                            settings["channel"] = query["channel"][0]
                        if "neurons" in query:
                            settings["neurons"] = [int(value) for value in query["neurons"][0].split(",")]
                        if any(settings.get(key, 0) not in (0, 1) for key in ("stats", "labels")):
                            raise ValueError("Stats and labels flags must be 0 or 1")
                        for key in ("stats", "labels"):
                            if key in settings:
                                settings[key] = bool(settings[key])
                        reader = temporal_page if temporal else activation_page
                        self._json(reader(directory / "experiment", **settings))
                    except (ValueError, OSError, KeyError, IndexError, TypeError) as error:
                        self._json({"error": str(error)}, 400)
                    return
                if parts[3] == "experiment.py":
                    state = controller.snapshot()["run"]
                    if state and state["id"] == parts[2] and state["phase"] in ACTIVE:
                        self._json({"error": "Wait until the experiment stops before exporting"}, 409)
                        return
                    try:
                        from .python_export import script_for_run
                        body = script_for_run(directory).encode('utf-8')
                    except (ValueError, OSError, KeyError, TypeError) as error:
                        self._json({"error": str(error)}, 400)
                        return
                    self._headers("text/x-python; charset=utf-8", len(body))
                    self.wfile.write(body)
                    return
                if parts[3] in ("plan.json", "summary.json"):
                    file = directory / parts[3]
                    if file.is_file():
                        self._headers("application/json", file.stat().st_size)
                        with file.open("rb") as stream:
                            shutil.copyfileobj(stream, self.wfile)
                        return
                elif parts[3] == "artifacts.zip":
                    state = controller.snapshot()["run"]
                    if state and state["id"] == parts[2] and state["phase"] in ACTIVE:
                        self._json({"error": "Wait until the experiment stops before exporting"}, 409)
                        return
                    with tempfile.TemporaryDirectory() as temporary:
                        from .python_export import write_archive
                        archive = Path(temporary) / "artifacts.zip"
                        try:
                            write_archive(directory, archive)
                        except (ValueError, OSError, KeyError, TypeError) as error:
                            self._json({"error": str(error)}, 400)
                            return
                        self._headers("application/zip", archive.stat().st_size)
                        with archive.open("rb") as stream:
                            shutil.copyfileobj(stream, self.wfile)
                    return
                self._json({"error": "Artifact not available"}, 404)
                return
            relative = path.lstrip("/") or "index.html"
            file = (frontend / relative).resolve()
            if not file.is_relative_to(frontend) or not file.is_file():
                self._json({"error": "Not found"}, 404)
                return
            self._headers(mimetypes.guess_type(file.name)[0] or "application/octet-stream", file.stat().st_size)
            with file.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile)

        def do_POST(self) -> None:
            if not self._trusted(mutation=True, upload=self.path == "/api/uploads"):
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path == "/api/uploads":
                    if not 0 < length <= 128 * 1024 * 1024:
                        raise ValueError("Upload limit is 128 MiB per file")
                    name = Path(unquote(self.headers.get("X-Lab-Filename", ""))).name
                    suffix = Path(name).suffix.lower()
                    if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".wav", ".mp3", ".flac", ".ogg", ".mp4", ".mov", ".webm"):
                        raise ValueError("Upload an image, audio, or video file")
                    uploads = controller.root / "uploads"
                    uploads.mkdir(exist_ok=True)
                    identifier = secrets.token_hex(16) + suffix
                    destination = uploads / identifier
                    remaining = length
                    with destination.open("xb") as stream:
                        while remaining:
                            chunk = self.rfile.read(min(remaining, 1024 * 1024))
                            if not chunk:
                                destination.unlink()
                                raise ValueError("Incomplete upload")
                            stream.write(chunk)
                            remaining -= len(chunk)
                    self._json({"asset": identifier, "name": name}, 201)
                    return
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("JSON request limit is 1 MiB")
                data = json.loads(self.rfile.read(length))
                if self.path == "/api/export-python":
                    from .python_export import python_source
                    source = python_source(data, factory=controller.factory)
                    self._json({"filename": "experiment.py", "source": source})
                elif self.path == "/api/runs":
                    self._json({"id": controller.submit(data)}, 202)
                elif self.path == "/api/experiments":
                    self._json({"id": controller.submit_request(data)}, 202)
                elif self.path == "/api/validate":
                    self._json({"id": controller.validate_request(data)}, 202)
                elif self.path == "/api/model":
                    controller.load(data)
                    self._json({"status": "loading"}, 202)
                elif self.path == "/api/cancel":
                    controller.cancel()
                    self._json({"status": "stopping after the current trial"})
                else:
                    self._json({"error": "Not found"}, 404)
            except (ValueError, TypeError) as error:
                self._json({"error": str(error)}, 400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
