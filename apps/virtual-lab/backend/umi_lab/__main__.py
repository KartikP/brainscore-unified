"""Launch the optional lab, or run its exported plan without a browser."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .worker import DEFAULT_FACTORY


def main(argv: list[str] | None = None) -> int | None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--factory", default=DEFAULT_FACTORY, help="Trusted Python module:function returning LabProfile")
    parser.add_argument("--core", type=Path, help="Use a matching core source checkout for this app only")
    parser.add_argument("--output", type=Path, default=Path(".lab-runs"))
    parser.add_argument("--run", type=Path, help="Run an exported plan directly, without the GUI")
    parser.add_argument("--check", action="store_true", help="Check the frontend and UMI package metadata without loading a model")
    parser.add_argument("--no-browser", action="store_true", help="Print the local URL without opening a browser")
    args = parser.parse_args(argv)
    from .runtime import configure_runtime
    configure_runtime(args.core)
    if args.check:
        from .checks import setup_report
        text, problems = setup_report()
        print(text)
        return 1 if problems else 0
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
    from .checks import setup_report
    text, problems = setup_report()
    if problems:
        parser.error(text)
    from .frontend import frontend_directory
    from .server import LabController, make_server
    frontend = frontend_directory()
    controller = LabController(args.output, args.factory)
    try:
        try:
            server = make_server(controller, frontend, args.port)
        except OSError as error:
            parser.error(f"Cannot start the lab on port {args.port}: {error}. Select another port with --port.")
        print(f"Virtual lab: http://127.0.0.1:{server.server_port}", flush=True)
        print(f"Experiment files: {args.output.resolve()}", flush=True)
        if not args.no_browser:
            import threading
            import webbrowser

            def open_browser() -> None:
                try:
                    webbrowser.open_new_tab(f"http://127.0.0.1:{server.server_port}")
                except Exception:
                    print("The browser did not open. Open the URL above manually.", flush=True)

            threading.Thread(target=open_browser, daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
