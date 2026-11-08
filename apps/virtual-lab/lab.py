"""Run from a source checkout: python lab.py [benchmark | server options]."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from umi_lab.runtime import configure_runtime
configure_runtime()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        sys.argv.pop(1)
        from umi_lab.benchmark import main
    else:
        from umi_lab.__main__ import main
    raise SystemExit(main())
