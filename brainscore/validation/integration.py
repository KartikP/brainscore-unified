"""Run one reviewed integration profile against installed candidate wheels.

This command prepares local evidence. It does not provision workers or download
assets. Run it in a disposable Linux environment with outbound traffic disabled.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import psutil

from .revisions import REPOSITORIES, verify
from .workspace import stage_tests


CACHE_PATHS = {
    'huggingface': 'HF_HOME', 'brainio': 'BRAINIO_HOME', 'torch': 'TORCH_HOME',
    'model_tools': 'MT_HOME',
}


def validate_pca_assets(directory: Path) -> None:
    """Check the ImageNet input used to fit legacy vision PCA, without downloading."""
    import h5py

    path = directory / 'imagenet2012.hdf5'
    if not path.is_file():
        raise ValueError(f'Missing PCA input: {path}. Stage your authorized ImageNet HDF5 file here.')
    try:
        with h5py.File(path, 'r') as handle:
            images = handle['val/images']
            if not isinstance(images, h5py.Dataset) or images.ndim == 0 or len(images) < 50_000:
                raise ValueError('Expected the 50,000-image validation set at val/images')
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f'Invalid PCA input {path}: {error}') from error


def fingerprint(directory: Path) -> tuple[str, int]:
    """Hash file names and contents, including the targets of HF snapshot links."""
    if not directory.is_dir():
        raise ValueError(f'Asset directory is missing: {directory}')
    digest = hashlib.sha256()
    size = 0
    files = sorted(path for path in directory.rglob('*') if path.is_file())
    if any(path.is_symlink() and (not path.exists() or path.is_dir())
           for path in directory.rglob('*')):
        raise ValueError('Asset cache contains a broken link or a directory link')
    if not files:
        raise ValueError(f'Asset directory is empty: {directory}')
    for path in files:
        name = str(path.relative_to(directory)).encode()
        digest.update(len(name).to_bytes(8, 'big'))
        digest.update(name)
        with path.open('rb') as handle:
            digest.update(hashlib.file_digest(handle, 'sha256').digest())
        size += path.stat().st_size
    return digest.hexdigest(), size


def prepare_assets(record: dict, profile: dict, home: Path) -> dict:
    """Validate prepared cache builds, then copy them into disposable storage."""
    if record['profile'] != profile['id']:
        raise ValueError('Asset manifest is for a different profile')
    assets = record['assets']
    if set(assets) != set(profile['required_caches']):
        raise ValueError('Asset manifest must contain exactly the required caches')
    total = 0
    for name, asset in assets.items():
        if name not in CACHE_PATHS or not asset.get('build'):
            raise ValueError('Each supported cache needs a checkpoint or dataset build description')
        if name == 'model_tools':
            validate_pca_assets(Path(asset['path']))
        digest, size = fingerprint(Path(asset['path']))
        if digest != asset['sha256']:
            raise ValueError(f'Asset content differs from its pin: {name}')
        total += size
    if shutil.disk_usage(home).free < total + 2 * 1024**3:
        raise ValueError('Insufficient disk space to copy assets with 2 GiB working headroom')
    for name, asset in assets.items():
        destination = home / name
        shutil.copytree(asset['path'], destination)
        if fingerprint(destination)[0] != asset['sha256']:
            raise ValueError(f'Asset changed while copying: {name}')
    return {'assets': assets, 'copied_bytes': total}


def child_environment(home: Path, device: str) -> dict:
    """Use private caches and exclude inherited cloud credentials and test flags."""
    # Allowlist avoids hidden cache overrides, credentials and PYTEST_ADDOPTS.
    env = {key: os.environ[key] for key in (
        'PATH', 'LANG', 'LC_ALL', 'SYSTEMROOT', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
    ) if key in os.environ}
    env.update(HOME=str(home), RESULTCACHING_HOME=str(home / 'results'),
               RESULTCACHING_DISABLE='1', XDG_CACHE_HOME=str(home / 'cache'),
               MPLCONFIGDIR=str(home / 'matplotlib'), TMPDIR=str(home / 'tmp'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', BS_INSTALL_DEPENDENCIES='no',
               AWS_EC2_METADATA_DISABLED='true', AWS_SHARED_CREDENTIALS_FILE=os.devnull,
               AWS_CONFIG_FILE=os.devnull, UMI_INTEGRATION_DEVICE=device,
               UMI_TEST_CPU_ONLY='1' if device == 'cpu' else '0',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    env.update({variable: str(home / name) for name, variable in CACHE_PATHS.items()})
    env['HF_HUB_CACHE'] = str(home / 'huggingface' / 'hub')
    env['HUGGINGFACE_HUB_CACHE'] = env['HF_HUB_CACHE']
    return env


def bounded_run(command: list[str], *, cwd: Path, env: dict, log: Path,
                seconds: float, memory_bytes: int) -> dict:
    """Stop the whole process group on deadline, memory excess or interruption."""
    start = time.monotonic()
    peak = 0
    reason = None
    with log.open('w') as output:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            parent = psutil.Process(process.pid)
            while process.poll() is None:
                rss = 0
                try:
                    children = [parent, *parent.children(recursive=True)]
                except psutil.NoSuchProcess:
                    children = []
                for child in children:
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                if time.monotonic() - start >= seconds:
                    reason = 'timeout'
                    break
                if rss > memory_bytes:
                    reason = 'memory_limit'
                    break
                time.sleep(0.1)
        finally:
            # Also clean up descendants that outlive an otherwise successful parent.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return {'returncode': process.returncode if reason is None else 1,
            'stop_reason': reason, 'seconds': time.monotonic() - start,
            'peak_process_tree_rss_bytes': peak, 'command': command}


def read_evidence(path: Path) -> dict:
    report = json.loads(path.read_text())
    if not report.get('complete') or report.get('returncode') != 0:
        raise ValueError(f'Incomplete or failed evidence: {path.name}')
    return report


def qualify(*, root: Path, revisions: Path, wheels: Path, manifest: Path,
            assets: Path, profile_id: str, output: Path) -> int:
    """Fail closed and retain evidence even when preparation or tests fail."""
    output.mkdir(parents=True, exist_ok=False)
    report = {'profile': profile_id, 'complete': False, 'returncode': 1, 'phases': {},
              'environment': {'platform': platform.platform(), 'python': sys.version}}
    try:
        if sys.platform != 'linux' or sys.version_info[:2] != (3, 11):
            raise ValueError('These qualification profiles require Linux and Python 3.11')
        plan = json.loads(manifest.read_text())
        profile = next(item for item in plan['profiles'] if item['id'] == profile_id)
        if profile.get('blocked_reason'):
            raise ValueError(profile['blocked_reason'])
        expected = profile['expected_collection']
        if not expected or len(set(expected)) != len(expected) or len(expected) != profile['expected_count']:
            raise ValueError('Expected test IDs must be nonempty, unique and match expected_count')
        record = json.loads(revisions.read_text())
        verify(record, root, list(REPOSITORIES))
        for name in REPOSITORIES:
            dirty = subprocess.check_output(
                ['git', '-C', str(root / name), 'status', '--porcelain'], text=True)
            if dirty.strip():
                raise ValueError(f'Qualification requires a clean checkout: {name}')
        report['revisions'] = record
        report['profile_definition'] = profile
        report['dependencies'] = dict(sorted(
            (dist.metadata['Name'], dist.version) for dist in metadata.distributions()
            if dist.metadata['Name']))
        device = profile['device']
        import torch
        report['device'] = {'requested': device, 'torch': torch.__version__, 'cuda': torch.version.cuda}
        if device == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA profile requires an available NVIDIA GPU')
            free, total = torch.cuda.mem_get_info()
            report['device'].update(name=torch.cuda.get_device_name(), free_bytes=free, total_bytes=total)
            if total < profile['proposed_limits']['gpu_memory_gib'] * 1024**3 * 0.95:
                raise ValueError('GPU is smaller than the proposed profile requirement')
        lock = output / 'profile.json'
        lock.write_text(json.dumps(profile, indent=2) + '\n')
        limits = profile['proposed_limits']
        report['available_memory_bytes'] = psutil.virtual_memory().available
        if report['available_memory_bytes'] < limits['memory_gib'] * 1024**3:
            raise ValueError('Available host memory is below the profile memory allowance')
        with tempfile.TemporaryDirectory(prefix='umi-integration-') as temporary:
            stage = Path(temporary)
            stage_tests(root / profile['repository'], stage)
            home = stage / 'home'
            (home / 'tmp').mkdir(parents=True)
            report['prepared_assets'] = prepare_assets(json.loads(assets.read_text()), profile, home)
            env = child_environment(home, device)
            deadline = time.monotonic() + limits['timeout_minutes'] * 60
            for phase in ('collection', 'execution'):
                evidence = output / f'{phase}.json'
                identity = output / f'{phase}-installed.json'
                command = [sys.executable, '-I', '-m', 'brainscore.validation.installed',
                           '--stage', str(stage), '--wheels', str(wheels), '--identity', str(identity),
                           *profile['pytest_args'], '-q', '-ra', '-p', 'no:cacheprovider',
                           '-p', 'pytest_check.plugin', '-p', 'pytest_mock',
                           '-p', 'brainscore.validation.integration_checks',
                           '--umi-integration-profile', str(lock),
                           '--umi-integration-report', str(evidence)]
                if phase == 'collection':
                    command.append('--collect-only')
                else:
                    command.extend(['--junitxml', str(output / 'tests.xml')])
                result = bounded_run(command, cwd=stage, env=env, log=output / f'{phase}.log',
                                     seconds=max(0, deadline - time.monotonic()),
                                     memory_bytes=limits['memory_gib'] * 1024**3)
                report['phases'][phase] = result
                if result['returncode']:
                    raise ValueError(f'{phase} failed; see retained log')
                checked = read_evidence(evidence)
                installed = read_evidence(identity)
                if installed['import_violations'] or sorted(checked['selected']) != sorted(expected):
                    raise ValueError('Installed identity or collected selection differs')
                if checked['collect_only'] != (phase == 'collection'):
                    raise ValueError('Test report belongs to the wrong phase')
                if phase == 'execution' and (
                    set(checked['phases']) != set(expected) or any(
                        states != {'setup': 'passed', 'call': 'passed', 'teardown': 'passed'}
                        for states in checked['phases'].values())
                ):
                    raise ValueError('Not every selected test completed successfully')
                if phase == 'execution' and profile_id in ('native-text', 'native-vlm') and not checked['models']:
                    raise ValueError('Native model placement evidence is missing')
            if not (output / 'tests.xml').is_file():
                raise ValueError('JUnit report missing')
        report.update(complete=True, returncode=0)
    except (Exception, KeyboardInterrupt) as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report['returncode']


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'revisions', 'wheels', 'manifest', 'assets', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    return qualify(**{name: getattr(args, name).resolve() for name in (
        'root', 'revisions', 'wheels', 'manifest', 'assets', 'output')}, profile_id=args.profile)


if __name__ == '__main__':
    raise SystemExit(main())
