"""Shared pytest config for the unified test suite.

Tier taxonomy (markers registered in pyproject.toml):
  * unit         — fast, offline, isolated (no real model weights, no network, no S3)
  * integration  — end-to-end across components (e.g. process() -> benchmark -> score),
                   still offline (synthetic models/fixtures), runnable in CI
  * slow         — needs real model downloads / heavy data / GPU / AWS; run on demand
  * private_access — needs S3 / private resources (often combined with slow)

Default behavior: any test WITHOUT an explicit tier marker is treated as `unit`,
so `-m unit` and `-m "not slow"` are meaningful without hand-marking every file.
Files that need a real model or heavy data set `pytestmark = pytest.mark.slow`
(e.g. test_text_wrapper, test_roar_benchmark, test_vlm_vision_wrapper); e2e
glue tests set `@pytest.mark.integration`.

Run tiers:
  pytest -m unit                  # fast, every commit
  pytest -m "unit or integration" # CI tier (offline, no big models)
  pytest -m slow                  # on demand (AWS creds + model downloads)
"""
import pytest

pytest_plugins = ['brainscore.validation.cpu_profile']

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
# brainscore.validation.workspace); push to v2 and the nightly run include them.
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
