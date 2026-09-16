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
             'tests/test_channel_registry.py', 'tests/test_contract_drift.py'],
    'vision': ['tests/test_unified_adapter.py', 'tests/test_preflight.py',
               'tests/test_subject_conformance.py', 'tests/test_hmax_runtime.py'],
    'language': ['tests/test_unified_adapter.py', 'tests/test_preflight.py', 'tests/test_subject_conformance.py'],
    'unified': ['tests', '-m', 'unit or integration'],
}


def qualify(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(root/repo) for repo in SUITES),
        RESULTCACHING_DISABLE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        BS_INSTALL_DEPENDENCIES='no')
    report = {'tier': 'source-offline', 'python': sys.version, 'repositories': {}}
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
        with log.open('w') as handle:
            run = subprocess.run([sys.executable, '-m', 'pytest', *suite, '-q', '-p', 'no:cacheprovider'],
                                 cwd=checkout, env=env, stdout=handle, stderr=subprocess.STDOUT)
        report['repositories'][repo] = {'base_commit': commit, 'working_changes_sha256': digest.hexdigest(),
            'dirty': bool(diff or untracked), 'returncode': run.returncode, 'log': str(log)}
        out.write_text(json.dumps(report, indent=2)+'\n')
        print(f'{repo}: {"PASS" if run.returncode == 0 else "FAIL"} ({log})', flush=True)
    return int(any(r['returncode'] for r in report['repositories'].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    return qualify(args.root, args.out)


if __name__ == '__main__':
    raise SystemExit(main())
