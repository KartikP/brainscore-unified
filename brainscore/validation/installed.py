"""Verify candidate wheels and run copied tests against installed packages."""
import argparse
import base64
import csv
from email.parser import BytesParser
import hashlib
import importlib
from importlib import metadata
import io
import json
from pathlib import Path
import sys
import sysconfig
import zipfile

import pytest


PACKAGES = ('brainscore_core', 'brainscore_vision', 'brainscore_language', 'brainscore')


def _sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_wheels(directory: Path) -> dict:
    """Require installed package files to match the four supplied wheels."""
    evidence = {}
    for package in PACKAGES:
        wheels = list(directory.glob(f'{package}-*.whl'))
        if len(wheels) != 1:
            raise ValueError(f'Expected one {package} wheel, found {len(wheels)}')
        wheel = wheels[0]
        distribution = metadata.distribution(package)
        package_root = Path(distribution.locate_file(package)).resolve()
        sites = {Path(sysconfig.get_path(name)).resolve() for name in ('purelib', 'platlib')}
        if not any(package_root.is_relative_to(site) for site in sites):
            raise ValueError(f'{package} is outside this interpreter\'s site-packages')
        direct_url = json.loads(distribution.read_text('direct_url.json') or '{}')
        if direct_url.get('dir_info', {}).get('editable'):
            raise ValueError(f'{package} is installed editable')
        module = importlib.import_module(package)
        if Path(module.__file__).resolve() != package_root / '__init__.py':
            raise ValueError(f'{package} imported outside its installed distribution')
        checked = 0
        with zipfile.ZipFile(wheel) as archive:
            records = [name for name in archive.namelist() if name.endswith('.dist-info/RECORD')]
            if len(records) != 1:
                raise ValueError(f'Invalid wheel RECORD: {wheel}')
            wheel_metadata = BytesParser().parsebytes(
                archive.read(records[0].removesuffix('RECORD') + 'METADATA'))
            if wheel_metadata['Version'] != distribution.version:
                raise ValueError(f'Installed version differs from candidate wheel: {package}')
            for name, fingerprint, _ in csv.reader(io.StringIO(archive.read(records[0]).decode())):
                if not name.startswith(package + '/'):
                    continue
                if not fingerprint or not fingerprint.startswith('sha256='):
                    raise ValueError(f'Missing SHA-256 for {name}')
                path = Path(distribution.locate_file(name)).resolve()
                if not path.is_relative_to(package_root):
                    raise ValueError(f'Package file escapes distribution: {name}')
                expected = fingerprint.split('=', 1)[1]
                actual = base64.urlsafe_b64encode(bytes.fromhex(_sha256(path))).decode().rstrip('=')
                if actual != expected:
                    raise ValueError(f'Installed file differs from candidate wheel: {name}')
                checked += 1
        if not checked:
            raise ValueError(f'Wheel contains no files for {package}')
        evidence[package] = {
            'version': distribution.version,
            'import_root': str(package_root),
            'wheel': str(wheel.resolve()),
            'wheel_sha256': _sha256(wheel),
            'verified_files': checked,
        }
    return evidence


class InstalledImports:
    """Reject source imports introduced during collection or test execution."""

    def __init__(self, packages: dict) -> None:
        self.roots = {name: Path(info['import_root']) for name, info in packages.items()}
        self.checked = set()
        self.violations = set()

    def check(self) -> None:
        for name, module in list(sys.modules.items()):
            package = name.split('.')[0]
            if package not in self.roots:
                continue
            filename = getattr(module, '__file__', None)
            if not isinstance(filename, str) or (name, filename) in self.checked:
                continue
            self.checked.add((name, filename))
            if not Path(filename).resolve().is_relative_to(self.roots[package]):
                self.violations.add(f'{name}: {filename}')

    def pytest_collection_finish(self, session) -> None:
        self.check()

    def pytest_runtest_logreport(self, report) -> None:
        self.check()

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionfinish(self, session, exitstatus) -> None:
        self.check()
        if self.violations:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--wheels', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    args, tests = parser.parse_known_args()
    report = {'complete': False, 'packages': {}, 'import_violations': []}
    args.identity.parent.mkdir(parents=True, exist_ok=True)
    try:
        report['packages'] = verify_wheels(args.wheels)
        guard = InstalledImports(report['packages'])
        # Only copied tests and fixtures are here, never the package source.
        # This also lets peer tests import helpers from their own tests package.
        sys.path.insert(0, str(args.stage))
        code = int(pytest.main(['--import-mode=importlib', *tests], plugins=[guard]))
        report['import_violations'] = sorted(guard.violations)
        report['complete'] = True
        report['returncode'] = code
        return code
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['returncode'] = 1
        print(f"Installed-package check failed: {report['error']}", flush=True)
        return 1
    finally:
        args.identity.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
