"""Launch the optional lab, or run its exported plan without a browser."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .worker import DEFAULT_FACTORY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--factory", default=DEFAULT_FACTORY, help="Trusted Python module:function returning LabProfile")
    parser.add_argument("--core", type=Path, help="Use a matching core source checkout for this app only")
    parser.add_argument("--output", type=Path, default=Path(".lab-runs"))
    parser.add_argument("--run", type=Path, help="Run an exported plan directly, without the GUI")
    args = parser.parse_args()
    from .runtime import configure_runtime
    configure_runtime(args.core)
    if args.run:
        from .experiment import Plan, execute
        from .worker import load_profile
        saved = json.loads(args.run.read_text())
        args.output.mkdir(parents=True, exist_ok=True)
        directory = args.output / f"direct-{time.time_ns()}"
        if saved.get("kind") == "workbench":
            from .workbench import Request, run_session, run_benchmark
            from brainscore import load_model
            subject = (load_model(saved["subject"]) if saved["model_source"] == "registered"
                       else load_profile(args.factory).subject)
            request = Request.parse(saved["request"])
            if request.mode == "benchmark":
                result = run_benchmark(subject, request, directory, lambda event: None, lambda: False)
            elif request.mode == "protocol":
                from .protocol import run_protocol
                result = run_protocol(subject, request, directory, args.run.parent / "inputs", lambda event: None, lambda: False)
            else:
                result = run_session(subject, request, directory, args.run.parent / "inputs", lambda event: None, lambda: False)
        else:
            plan = Plan.parse(saved)
            result = execute(load_profile(args.factory), plan, directory)
        print(json.dumps(result, indent=2))
        return
    from .server import LabController, make_server
    frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if not (frontend / "index.html").is_file():
        parser.error("Build the frontend first: cd apps/virtual-lab/frontend && npm ci && npm run build")
    controller = LabController(args.output, args.factory)
    try:
        server = make_server(controller, frontend, args.port)
        print(f"Virtual lab: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
