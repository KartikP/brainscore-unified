"""Tier markers of the unified suite: one pytest plugin for ``tests`` and ``brainscore/benchmarks``.

Both conftest files name this module in ``pytest_plugins``; pytest registers it once by name, so
a joint collection marks each test once, and a plugin folder that runs alone gets the same markers.
"""
import pytest

_TIERS = ('unit', 'integration', 'slow')

# Files that start subprocesses, bind sockets or measure time. Under pytest-xdist
# (--dist loadgroup) they all run on one worker, one after another, so they never
# race each other for ports, git state or CPU.
SERIAL_FILES = {
    'test_record_validation.py', 'test_libero_transport.py', 'test_run_candidate.py',
    'test_openpi_remote.py', 'test_integration_runner.py', 'test_check_pins.py',
    'test_environment_sessions.py', 'test_required_checks.py',
    'test_mirage_umi.py', 'test_mirage_overhead.py',
}
# Qualification and documented-example tests. The per-PR tier skips them (see
# brainscore.validation.workspace); push to v2 and a manual full run include them.
QUALIFICATION_FILES = {'test_external_qualification.py', 'test_mirage_umi.py', 'test_mirage_overhead.py'}
QUALIFICATION_TESTS = {
    'test_contrastive_steering_tool.py::test_documented_steering_example_runs',
    'test_leace_tool.py::test_documented_leace_example_runs',
    'test_linear_probe_tool.py::test_documented_cpu_example_runs_without_downloads',
    'test_extension_templates.py::test_documented_code_blocks_run',
    'test_notebook_validation.py::test_execution_retains_failure_without_changing_source',
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = item.path.name
        if path in SERIAL_FILES:
            item.add_marker(pytest.mark.serial)
            item.add_marker(pytest.mark.xdist_group('serial'))
        if path in QUALIFICATION_FILES or any(
                item.nodeid.split('/')[-1].split('[')[0] == test for test in QUALIFICATION_TESTS):
            item.add_marker(pytest.mark.qualification)
        if not any(item.get_closest_marker(t) for t in _TIERS):
            item.add_marker(pytest.mark.unit)
