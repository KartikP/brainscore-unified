"""Run the offline suites against source checkouts or installed candidate wheels.

python -m brainscore.validation.workspace --root /path/to/sibling/repos --out report.json
Use --mode installed --wheels /path/to/wheels for installed-package checks.
Neither mode replaces scientific or first-use qualification.
"""
import argparse
import base64
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

# Keep offline selections explicit: mixed files contain real-model/data tests.
SUITES = {
    'core': [
        'tests/test_api_cohesion.py',
        'tests/test_lab_vocabulary.py',
        'tests/test_external_extensions.py',
        'tests/test_capabilities.py',
        'tests/test_model_interface.py',
        'tests/test_constructor_ergonomics.py',
        'tests/test_channel_compatibility.py',
        'tests/test_device_agnostic_env.py',
        'tests/test_streaming.py',
        'tests/test_streaming_helpers.py',
        'tests/test_environment_contract.py',
        'tests/test_streaming_behavior.py',
        'tests/test_payload_validation.py',
        'tests/test_io_catalog.py',
        'tests/test_io_catalog_preflight.py',
        'tests/test_channel_registry.py',
        'tests/test_contract_drift.py',
        'tests/test_output_event.py',
        'tests/test_native_subject.py',
        'tests/test_subject_channels.py',
        'tests/test_subject_rename.py',
        'tests/test_model_interface_public_api.py',
        'tests/test_plugin_management/test_plugin_command_paths.py',
        'tests/test_compatibility.py',
        'tests/test_state_change.py',
        'tests/test_extraction_cache.py',
        'tests/test_runtime_dependencies.py',
        'tests/test_preflight.py',
        'tests/test_leaderboard_eligibility.py',
        'tests/test_assembly_builder.py',
        'tests/test_behavioral_readout.py',
        'tests/test_composite_selector.py',
        'tests/test_dispatch_cache.py',
        'tests/test_env_check.py',
        'tests/test_environment_step.py',
        'tests/test_execution_plan.py',
        'tests/test_hf_compat.py',
        'tests/test_memory.py',
        'tests/test_message.py',
        'tests/test_metrics/test___init__.py',
        'tests/test_multimodal.py',
        'tests/test_nulls.py',
        'tests/test_numeric.py',
        'tests/test_out_channels.py',
        'tests/test_plugin_management/test_benchmarks.py',
        'tests/test_plugin_management/test_parse_plugin_changes.py',
        'tests/test_score_metadata.py',
        'tests/test_score_status.py',
        'tests/test_streaming_perception.py',
        'tests/test_streaming_realtime.py',
        'tests/test_streaming_windowed.py',
        'tests/test_synchronize_modalities.py',
        'tests/test_temporal.py',
        'tests/test_unit_selection.py',
    ],
    'vision': [
        'tests/test_package_contents.py',
        'tests/test_unified_adapter.py',
        'tests/test_preflight.py',
        'tests/test_subject_conformance.py',
        'tests/test_hmax_runtime.py',
        'tests/test_activation_cache_key.py',
        'tests/test_activation_cache_integrity.py',
        'tests/test_activation_assembly_builder.py',
        'tests/test_layer_mapping_explorer.py::TestImports',
        'tests/test_layer_mapping_explorer.py::TestSuggestMapping',
        'tests/test_layer_mapping_explorer.py::TestSaveOutputs',
        'tests/test_metric_helpers/test_temporal.py',
        'tests/test_metric_helpers/test_transformations.py',
        'tests/test_metric_helpers/test_xarray_utils.py',
        'tests/test_model_helpers/brain_transformation/test_temporal.py',
        'tests/test_model_helpers/temporal/brain_transformation/test_temporal_ops.py',
        'tests/test_model_helpers/temporal/test_utils.py',
        'tests/test_model_helpers/test_generic_plugin_tests.py',
        'tests/test_plugin_management/test_memory_precheck.py',
        'tests/test_plugin_management/test_memory_wrappers.py',
        'tests/test_plugin_management/test_vision_domain_plugin.py::TestVisionDomainPluginUnit',
        'tests/test_pytorch_wrapper.py',
        'tests/test_utils.py',
    ],
    'language': [
        'tests/test_unified_adapter.py',
        'tests/test_preflight.py',
        'tests/test_subject_conformance.py',
        'tests/test_kv_cache.py',
        'tests/test_kv_cache_slicing.py',
        'tests/test_pereira2018_registry.py',
        'tests/test_legacy_score.py',
        'tests/test_model_helpers/test_preprocessing.py',
        'tests/test_submission/test_metadata_only_workflow.py',
    ],
    'unified': [
        'tests',
        '-m',
        'unit or integration',
    ],
}


def stage_tests(checkout: Path, destination: Path) -> None:
    """Copy test inputs without making repository packages importable."""
    for name in ('tests', 'examples', 'notebooks', 'templates', 'install', '.github'):
        source = checkout / name
        if source.is_dir():
            shutil.copytree(source, destination / name,
                            ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '*.pyc'))
    for name in ('pyproject.toml', 'pytest.ini', 'setup.cfg'):
        source = checkout / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    if (destination / 'examples/libero/evaluate.py').is_file():
        _stage_installed_file(checkout, destination, EVALUATOR_HELPER)


# The Python 3.8 LIBERO evaluator loads this standard-library helper by path from
# ../../ without importing brainscore, so the stage needs the installed copy there.
EVALUATOR_HELPER = 'brainscore/experiments/backends/socket_deadline.py'


def _stage_installed_file(checkout: Path, destination: Path, name: str) -> None:
    """Copy one file of the installed brainscore distribution, verified against its RECORD.

    No __init__.py is staged, so the installed regular package still wins imports.
    """
    distribution = metadata.distribution('brainscore')
    entry = next((path for path in distribution.files or () if str(path) == name), None)
    if entry is None or entry.hash is None or entry.hash.mode != 'sha256':
        raise ValueError(f'Installed brainscore does not record {name}')
    source = Path(distribution.locate_file(entry)).resolve()
    if source.is_relative_to(Path(checkout).resolve()):
        raise ValueError(f'{name} resolves to the source checkout, not an installed package')
    data = source.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
    if digest != entry.hash.value:
        raise ValueError(f'Installed {name} differs from its RECORD')
    target = destination / name
    target.parent.mkdir(parents=True)
    target.write_bytes(data)


def qualify(root, out, *, mode='source', wheels=None):
    root, out = Path(root).resolve(), Path(out).resolve()
    if mode not in ('source', 'installed'):
        raise ValueError(f'Unknown test mode: {mode}')
    if mode == 'installed' and wheels is None:
        raise ValueError('Installed checks require --wheels')
    wheels = Path(wheels).resolve() if wheels is not None else None
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(root/repo) for repo in SUITES),
        RESULTCACHING_DISABLE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        BS_INSTALL_DEPENDENCIES='no')
    if mode == 'installed':
        for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS'):
            env.pop(key, None)
        for key in list(env):
            if key.startswith('AWS_'):
                del env[key]
        env.update(AWS_EC2_METADATA_DISABLED='true', AWS_SHARED_CREDENTIALS_FILE=os.devnull,
                   AWS_CONFIG_FILE=os.devnull)
    report = {'tier': f'{mode}-offline', 'python': sys.version,
              'complete': False, 'repositories': {}}
    out.write_text(json.dumps(report, indent=2)+'\n')
    for repo, suite in SUITES.items():
        checkout = root/repo
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
        diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=checkout)
        untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'],
                                           cwd=checkout, text=True).splitlines()
        digest = hashlib.sha256(diff)
        for name in sorted(untracked):
            digest.update(name.encode())
            digest.update((checkout/name).read_bytes())
        log = out.with_name(f'{out.stem}-{repo}.log')
        checks = out.with_name(f'{out.stem}-{repo}-checks.json')
        checks.unlink(missing_ok=True)
        command = [sys.executable, '-m', 'pytest', *suite, '-q', '-ra',
                   '-p', 'no:cacheprovider', '-p', 'brainscore.validation.required_checks',
                   '-p', 'brainscore.validation.cpu_profile',
                   '--umi-check-report', str(checks), '--umi-repository', repo]
        identity = out.with_name(f'{out.stem}-{repo}-installed.json')
        identity.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix=f'umi-{repo}-') as temporary, log.open('w') as handle:
            cwd = checkout
            run_env = env.copy()
            if mode == 'installed':
                cwd = Path(temporary)
                stage_tests(checkout, cwd)
                home = cwd / 'home'
                home.mkdir()
                run_env.update(HOME=str(home), RESULTCACHING_HOME=str(home / 'results'),
                               BRAINIO_HOME=str(home / 'brainio'), HF_HOME=str(home / 'huggingface'),
                               HF_HUB_CACHE=str(home / 'huggingface/hub'),
                               HUGGINGFACE_HUB_CACHE=str(home / 'huggingface/hub'),
                               TRANSFORMERS_CACHE=str(home / 'huggingface/transformers'),
                               TORCH_HOME=str(home / 'torch'),
                               XDG_CACHE_HOME=str(home / 'cache'), MPLCONFIGDIR=str(home / 'matplotlib'))
                command = [sys.executable, '-I', '-m', 'brainscore.validation.installed',
                           '--stage', str(cwd), '--wheels', str(wheels),
                           '--identity', str(identity), *command[3:]]
            try:
                run = subprocess.run(command, cwd=cwd, env=run_env, stdout=handle,
                                     stderr=subprocess.STDOUT, timeout=1800)
                returncode = run.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                handle.write('\nFAIL: required suite exceeded the 30-minute limit\n')
        evidence = json.loads(checks.read_text()) if checks.exists() else None
        # A zero process exit without a completed evidence report is not success.
        if returncode == 0 and (not evidence or evidence['returncode'] or not evidence['passed']):
            returncode = 1
        installed = json.loads(identity.read_text()) if identity.exists() else None
        if mode == 'installed' and returncode == 0 and (
                not installed or not installed['complete'] or installed['returncode']
                or installed['import_violations']):
            returncode = 1
        selected_files = {node.split('::')[0] for node in (evidence or {}).get('selected', [])}
        inventory = {str(path.relative_to(checkout)) for path in (checkout/'tests').rglob('test_*.py')}
        report['repositories'][repo] = {'base_commit': commit, 'working_changes_sha256': digest.hexdigest(),
            'dirty': bool(diff or untracked), 'returncode': returncode, 'log': str(log),
            'command': command, 'checks': evidence,
            'installed': installed,
            'unselected_test_files': sorted(inventory - selected_files)}
        out.write_text(json.dumps(report, indent=2)+'\n')
        print(f'{repo}: {"PASS" if returncode == 0 else "FAIL"} ({log})', flush=True)
    report['complete'] = True
    out.write_text(json.dumps(report, indent=2)+'\n')
    return int(any(r['returncode'] for r in report['repositories'].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=('source', 'installed'), default='source')
    parser.add_argument('--wheels', help='Directory containing the four candidate wheels')
    args = parser.parse_args()
    return qualify(args.root, args.out, mode=args.mode, wheels=args.wheels)


if __name__ == '__main__':
    raise SystemExit(main())
