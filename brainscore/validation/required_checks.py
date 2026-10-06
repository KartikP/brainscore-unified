"""Pytest plugin: report selected checks and fail on unexpected skips."""
import json
from pathlib import Path

import pytest

from .optional_checks import OPTIONAL_SKIPS


def pytest_addoption(parser):
    parser.addoption('--umi-check-report', help='Write the required-check evidence JSON')
    parser.addoption('--umi-repository', default='', help='Repository profile name')


def pytest_configure(config):
    output = config.getoption('--umi-check-report')
    if output:
        config.pluginmanager.register(
            RequiredChecks(output, config.getoption('--umi-repository')),
            'umi-required-checks',
        )


class RequiredChecks:
    def __init__(self, output, repository):
        self.output = Path(output)
        self.allowed = OPTIONAL_SKIPS.get(repository, {})
        self.selected = []
        self.deselected = []
        self.passed = []
        self.failed = []
        self.skipped = []

    def pytest_collection_finish(self, session):
        self.selected = [item.nodeid for item in session.items]

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def _record_skip(self, report):
        reason = str(report.longrepr)
        expected = self.allowed.get(report.nodeid)
        self.skipped.append({
            'nodeid': report.nodeid,
            'reason': reason,
            'optional': expected is not None and expected in reason,
        })

    def pytest_collectreport(self, report):
        if report.skipped:
            self._record_skip(report)
        elif report.failed:
            self.failed.append(report.nodeid)

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self._record_skip(report)
        elif report.failed:
            self.failed.append(report.nodeid)
        elif report.when == 'call' and report.passed:
            self.passed.append(report.nodeid)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        unexpected = [item for item in self.skipped if not item['optional']]
        if exitstatus == 0 and (unexpected or not self.passed or self.failed):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps({
            'selected': self.selected,
            'deselected': self.deselected,
            'passed': self.passed,
            'failed': self.failed,
            'skipped': self.skipped,
            'unexpected_skips': unexpected,
            'returncode': int(session.exitstatus),
        }, indent=2) + '\n')

    def pytest_terminal_summary(self, terminalreporter):
        unexpected = [item for item in self.skipped if not item['optional']]
        terminalreporter.section('UMI required-check accounting')
        terminalreporter.write_line(
            f'{len(self.passed)} passed; {len(self.skipped) - len(unexpected)} '
            f'explicit optional skips; {len(unexpected)} unexpected skips')
        for item in unexpected:
            terminalreporter.write_line(f"FAIL: {item['nodeid']}: {item['reason']}")
