"""Check built wheels before installing or publishing them.

This checks files and notices, not legal permission to redistribute them.
"""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile


REQUIRED = {
    'brainscore_core': ['*.dist-info/LICENSE', 'brainscore_core/plugin_management/test_plugin.sh'],
    'brainscore_vision': ['*.dist-info/LICENSE',
        'brainscore_vision/models/voneresnet_50_non_stochastic/vonenet/LICENSE'],
    'brainscore_language': ['*.dist-info/LICENSE',
        'brainscore_language/benchmarks/pereira2018/ceiling_packaging.py'],
    'brainscore': ['brainscore/models/vjepa_v1/backbone/LICENSE',
        'brainscore/models/vjepa_v1/backbone/NOTICE.md',
        'brainscore/benchmarks/rajalingham2018_2afc/*.csv'],
}
PERSONAL_PATH = re.compile(r'/(?:Users|home|Volumes|braintree)/')


def inspect_wheel(path: Path, package: str) -> dict:
    """Reject namespace collisions, missing resources, and personal notebook paths."""
    errors = []
    with ZipFile(path) as archive:
        names = archive.namelist()
        for pattern in REQUIRED[package]:
            if not any(fnmatch.fnmatchcase(name, pattern) for name in names):
                errors.append(f'Missing required resource: {pattern}')
        for name in names:
            parts = PurePosixPath(name).parts
            if not parts:
                continue
            if parts[0] != package and not parts[0].endswith('.dist-info'):
                errors.append(f'Unexpected top-level package: {name}')
            if '..' in parts or name.startswith('/') or '__pycache__' in parts or name.endswith('.pyc'):
                errors.append(f'Unexpected build file: {name}')
            if name.startswith('brainscore_core/supported_data_standards/brainio/tests/'):
                errors.append(f'Internal BrainIO test shipped: {name}')
            if name.endswith('.ipynb'):
                notebook = json.loads(archive.read(name))
                for index, cell in enumerate(notebook.get('cells', [])):
                    if cell.get('cell_type') != 'code':
                        continue
                    source = ''.join(cell.get('source', []))
                    # Ignore comments, which may describe historical locations.
                    code = '\n'.join(line for line in source.splitlines()
                                     if not line.lstrip().startswith('#'))
                    if PERSONAL_PATH.search(code):
                        errors.append(f'Personal path in {name}, cell {index}')
                    if cell.get('outputs') or cell.get('execution_count') is not None:
                        errors.append(f'Saved execution output in {name}, cell {index}')
    return {'wheel': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'files': len(names), 'errors': errors}


def main() -> int:
    """Write an inspectable report and fail if any coordinated wheel is invalid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheels', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for package in REQUIRED:
        wheels = list(args.wheels.glob(f'{package}-*.whl'))
        report[package] = (inspect_wheel(wheels[0], package) if len(wheels) == 1
                           else {'errors': [f'Expected one wheel; found {len(wheels)}']})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    return int(any(item['errors'] for item in report.values()))


if __name__ == '__main__':
    raise SystemExit(main())
