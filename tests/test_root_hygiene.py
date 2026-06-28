"""Tests for the shared workspace root hygiene check."""

from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_root_hygiene import check_root_hygiene, format_findings  # noqa: E402


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
