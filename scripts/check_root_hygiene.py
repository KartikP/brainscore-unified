"""Check that workspace-root secrets have not been reintroduced."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_FILE_PATTERNS: tuple[str, ...] = (
    "*.pem",
    "OPENROUTER.API",
)
IGNORED_DIR_NAMES = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
})


@dataclass(frozen=True)
class HygieneFinding:
    path: Path
    pattern: str


def _matching_pattern(filename: str, patterns: Sequence[str]) -> str | None:
    for pattern in patterns:
        if fnmatchcase(filename, pattern):
            return pattern
    return None


def check_root_hygiene(
        root: str | Path = DEFAULT_WORKSPACE_ROOT,
        patterns: Sequence[str] = FORBIDDEN_FILE_PATTERNS,
        ignored_dir_names: Iterable[str] = IGNORED_DIR_NAMES,
) -> list[HygieneFinding]:
    root_path = Path(root).resolve()
    ignored = set(ignored_dir_names)
    findings: list[HygieneFinding] = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [name for name in dirnames if name not in ignored]
        for filename in filenames:
            pattern = _matching_pattern(filename, patterns)
            if pattern is not None:
                findings.append(HygieneFinding(Path(dirpath) / filename, pattern))

    return sorted(findings, key=lambda finding: str(finding.path))


def format_findings(
        findings: Iterable[HygieneFinding],
        root: str | Path = DEFAULT_WORKSPACE_ROOT,
) -> str:
    root_path = Path(root).resolve()
    lines: list[str] = []
    for finding in findings:
        path = finding.path.resolve()
        try:
            display_path = path.relative_to(root_path)
        except ValueError:
            display_path = path
        lines.append(f"- {display_path} (matched {finding.pattern})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail if private-key/API-secret files exist under the workspace root.")
    parser.add_argument(
        "root",
        nargs="?",
        default=DEFAULT_WORKSPACE_ROOT,
        type=Path,
        help="Workspace root to scan. Defaults to the parent of the unified repo.",
    )
    args = parser.parse_args(argv)

    findings = check_root_hygiene(args.root)
    if findings:
        print("Secret-like files are not allowed under the workspace root:", file=sys.stderr)
        print(format_findings(findings, args.root), file=sys.stderr)
        print(
            "Move them to ~/.ssh, ~/.config/brainscore, or an environment variable.",
            file=sys.stderr,
        )
        return 1

    print(f"Workspace root hygiene OK: {Path(args.root).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
