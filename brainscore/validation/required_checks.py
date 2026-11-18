"""Pytest plugin: report selected checks and fail on unexpected skips."""
from importlib import metadata
import json
from pathlib import Path

import pytest

from .optional_checks import OPTIONAL_SKIPS

# Benchmark plugins keep their tests in their own folders. An installed run takes
# them from the installed package, not from a copy: a copy would import as source.
PLUGIN_TESTS = 'brainscore/benchmarks'


def installed_argument(argument):
    """The installed copy of a benchmark plugin test path; any other argument unchanged."""
    if argument != PLUGIN_TESTS and not argument.startswith(PLUGIN_TESTS + '/'):
        return argument
    return str(metadata.distribution('brainscore').locate_file(argument))


def plugin_nodeid(path, nodeid):
    """Name a benchmark plugin test by its file below ``brainscore/benchmarks/``.

    An installed run collects the plugin tests from site-packages, outside the
    rootdir, so pytest names them relative to a different folder. The file path
    and the test suffix give the same name in every run.
    """
    parts = Path(path).parts
    starts = [i for i in range(len(parts) - 1) if parts[i:i + 2] == ('brainscore', 'benchmarks')]
    if not starts:
        return nodeid
    _, sep, rest = nodeid.partition('::')
    return '/'.join(parts[starts[-1]:]) + sep + rest


@pytest.hookimpl(tryfirst=True)
def pytest_collectstart(collector):
    # Rename the module before its tests are built, so that every test ID, also
    # those an xdist worker sends to the controller, starts with the canonical path.
    if isinstance(collector, pytest.Module):
        collector._nodeid = plugin_nodeid(collector.path, collector.nodeid)


def pytest_addoption(parser):
    parser.addoption('--umi-check-report', help='Write the required-check evidence JSON')
    parser.addoption('--umi-repository', default='', help='Repository profile name')


def pytest_configure(config):
    output = config.getoption('--umi-check-report')
    # pytest-xdist workers report to the controller; only the controller records.
    if output and not hasattr(config, 'workerinput'):
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

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        # With pytest-xdist the controller does not collect; every worker sends
        # the same list.
        self.selected = list(ids)

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
