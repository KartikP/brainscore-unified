"""Tests for the shared workspace root hygiene check."""

from pathlib import Path
import sys

import pytest

# check_root_hygiene is a maintainer tool that scans the workspace root for stray
# secrets; it is deliberately not part of the distributed package. Skip rather than
# fail collection wherever it is absent.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

_mod = pytest.importorskip(
    "check_root_hygiene",
    reason="root-hygiene checker is a maintainer tool, not part of the package")
check_root_hygiene = _mod.check_root_hygiene
format_findings = _mod.format_findings


def test_detects_forbidden_secret_filenames(tmp_path):
    (tmp_path / "quest.pem").write_text("redacted", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "OPENROUTER.API").write_text("redacted", encoding="utf-8")

    findings = check_root_hygiene(tmp_path)

    assert {finding.path.name for finding in findings} == {
        "quest.pem",
        "OPENROUTER.API",
    }


def test_ignores_vcs_and_cache_dirs(tmp_path):
    for ignored_dir in [".git", ".pytest_cache", "__pycache__"]:
        path = tmp_path / ignored_dir
        path.mkdir()
        (path / "ignored.pem").write_text("redacted", encoding="utf-8")

    assert check_root_hygiene(tmp_path) == []


def test_workspace_root_has_no_forbidden_secret_files():
    workspace_root = Path(__file__).resolve().parents[2]
    findings = check_root_hygiene(workspace_root)

    assert findings == [], format_findings(findings, workspace_root)
