"""Run the offline contract suites in separate processes for all four repos.

python -m brainscore.validation.workspace --root /path/to/sibling/repos --out report.json
This is a source qualification tier. It is not scientific parity or wheel CI.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SUITES = {
    'core': ['tests/test_external_extensions.py', 'tests/test_capabilities.py',
             'tests/test_model_interface.py', 'tests/test_constructor_ergonomics.py',
             'tests/test_channel_compatibility.py', 'tests/test_device_agnostic_env.py',
             'tests/test_streaming.py', 'tests/test_streaming_helpers.py',
             'tests/test_streaming_behavior.py', 'tests/test_payload_validation.py',
             'tests/test_io_catalog.py', 'tests/test_io_catalog_preflight.py',
             'tests/test_channel_registry.py', 'tests/test_contract_drift.py',
             'tests/test_output_event.py', 'tests/test_native_subject.py',
             'tests/test_subject_channels.py', 'tests/test_subject_rename.py',
             'tests/test_model_interface_public_api.py',
             'tests/test_plugin_management/test_plugin_command_paths.py',
             'tests/test_compatibility.py', 'tests/test_state_change.py',
             'tests/test_extraction_cache.py', 'tests/test_preflight.py'],
    'vision': ['tests/test_unified_adapter.py', 'tests/test_preflight.py',
               'tests/test_subject_conformance.py', 'tests/test_hmax_runtime.py',
               'tests/test_activation_cache_key.py', 'tests/test_activation_cache_integrity.py'],
    'language': ['tests/test_unified_adapter.py', 'tests/test_preflight.py',
                 'tests/test_subject_conformance.py', 'tests/test_kv_cache.py',
                 'tests/test_kv_cache_slicing.py', 'tests/test_pereira2018_registry.py',
                 'tests/test_legacy_score.py'],
    'unified': ['tests', '-m', 'unit or integration'],
}


def qualify(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(root/repo) for repo in SUITES),
        RESULTCACHING_DISABLE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        BS_INSTALL_DEPENDENCIES='no')
    report = {'tier': 'source-offline', 'python': sys.version,
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
                   '--umi-check-report', str(checks), '--umi-repository', repo]
        with log.open('w') as handle:
            try:
                run = subprocess.run(command, cwd=checkout, env=env, stdout=handle,
                                     stderr=subprocess.STDOUT, timeout=1800)
                returncode = run.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                handle.write('\nFAIL: required suite exceeded the 30-minute limit\n')
        evidence = json.loads(checks.read_text()) if checks.exists() else None
        # A zero process exit without a completed evidence report is not success.
        if returncode == 0 and (not evidence or evidence['returncode'] or not evidence['passed']):
            returncode = 1
        selected_files = {node.split('::')[0] for node in (evidence or {}).get('selected', [])}
        inventory = {str(path.relative_to(checkout)) for path in (checkout/'tests').rglob('test_*.py')}
        report['repositories'][repo] = {'base_commit': commit, 'working_changes_sha256': digest.hexdigest(),
            'dirty': bool(diff or untracked), 'returncode': returncode, 'log': str(log),
            'command': command, 'checks': evidence,
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
    args = parser.parse_args()
    return qualify(args.root, args.out)


if __name__ == '__main__':
    raise SystemExit(main())
