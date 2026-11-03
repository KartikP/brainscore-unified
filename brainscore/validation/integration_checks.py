"""Require the reviewed test selection and retain every test phase."""
import json
import os
from pathlib import Path
import socket

import pytest


def pytest_addoption(parser) -> None:
    parser.addoption('--umi-integration-profile', required=False)
    parser.addoption('--umi-integration-report', required=False)


def pytest_configure(config) -> None:
    path = config.getoption('--umi-integration-profile')
    if path:
        config.pluginmanager.register(IntegrationChecks(config, Path(path)), 'integration_checks')


class IntegrationChecks:
    def __init__(self, config, profile: Path) -> None:
        self.config = config
        self.expected = json.loads(profile.read_text())['expected_collection']
        self.selected = []
        self.phases = {}
        self.collection_errors = []
        self.patches = pytest.MonkeyPatch()

    def pytest_sessionstart(self, session) -> None:
        # A worker-level outbound rule is still required for subprocesses.
        def refuse_network(*args, **kwargs):
            raise RuntimeError('Integration tests require pre-staged assets; network access refused')

        self.patches.setattr(socket.socket, 'connect', refuse_network)
        self.patches.setattr(socket.socket, 'connect_ex', refuse_network)
        if os.environ.get('UMI_INTEGRATION_DEVICE') == 'cpu':
            import torch
            # Module-scoped model fixtures run before function-scoped CPU fixtures.
            self.patches.setattr(torch.cuda, 'is_available', lambda: False)
            self.patches.setattr(torch.backends.mps, 'is_available', lambda: False)

    def pytest_collectreport(self, report) -> None:
        if report.failed or report.skipped:
            self.collection_errors.append(report.nodeid)

    def pytest_collection_finish(self, session) -> None:
        self.selected = [item.nodeid for item in session.items]
        if not self.expected or sorted(self.selected) != sorted(self.expected):
            raise pytest.UsageError('Integration selection differs from the reviewed test IDs')

    def pytest_runtest_logreport(self, report) -> None:
        self.phases.setdefault(report.nodeid, {})[report.when] = report.outcome

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus) -> None:
        collect_only = self.config.option.collectonly
        complete = bool(self.expected) and sorted(self.selected) == sorted(self.expected)
        complete = complete and not self.collection_errors
        if not collect_only:
            complete = complete and set(self.phases) == set(self.expected) and all(
                phases == {'setup': 'passed', 'call': 'passed', 'teardown': 'passed'}
                for phases in self.phases.values()
            )
        if not complete and not session.exitstatus:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        report = {
            'complete': complete,
            'collect_only': collect_only,
            'selected': self.selected,
            'phases': self.phases,
            'collection_errors': self.collection_errors,
            'models': getattr(self.config, '_umi_model_placements', []),
            'returncode': int(session.exitstatus),
        }
        Path(self.config.getoption('--umi-integration-report')).write_text(
            json.dumps(report, indent=2) + '\n')
        self.patches.undo()

def record_placement(config, model, *, identifier: str, dtype: str) -> None:
    """Check the actual extracted module, not just the worker's GPU inventory."""
    device = os.environ.get('UMI_INTEGRATION_DEVICE')
    if not device:
        return
    parameters = list(model.parameters())
    devices = sorted({parameter.device.type for parameter in parameters})
    dtypes = sorted({str(parameter.dtype) for parameter in parameters})
    if devices != [device] or dtypes != [dtype]:
        raise AssertionError(f'{identifier}: expected {device}/{dtype}, got {devices}/{dtypes}')
    config._umi_model_placements = getattr(config, '_umi_model_placements', [])
    config._umi_model_placements.append({
        'identifier': identifier, 'devices': devices, 'dtypes': dtypes,
    })
