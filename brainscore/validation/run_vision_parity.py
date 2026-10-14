"""Plan or run the four vision parity cases in separate CPU processes.

No inference runs without --execute. Each case retains the standard parity
checks and its own report. A successful vision selection is not a completed
eight-case release qualification.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from .benchmark_parity import RELEASE_CASES, CPU_FP32_POLICY


VISION_CASES = tuple(case.unified for case in RELEASE_CASES if case.domain == 'vision')
GIB = 2 ** 30


def _positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Use a finite positive value')
    return number


def _stop(process):
    # Stop the whole scoring process group, including worker processes.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run_case(command, *, directory, timeout_s, max_rss_bytes, min_available_bytes):
    started = time.monotonic()
    peak = 0
    reason = None
    env = {key: value for key, value in os.environ.items() if not key.startswith('AWS_')}
    env.update(
        AWS_EC2_METADATA_DISABLED='true', AWS_SHARED_CREDENTIALS_FILE=os.devnull,
        AWS_CONFIG_FILE=os.devnull, RESULTCACHING_DISABLE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', BS_INSTALL_DEPENDENCIES='no',
    )
    with (directory / 'scoring.log').open('w') as log:
        process = subprocess.Popen(
            command, cwd=directory, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)]
                              if p.is_running())
                    peak = max(peak, rss)
                    if time.monotonic() - started > timeout_s:
                        reason = 'time limit'
                    elif rss > max_rss_bytes:
                        reason = 'process memory limit'
                    elif psutil.virtual_memory().available < min_available_bytes:
                        reason = 'system memory reserve'
                except psutil.NoSuchProcess:
                    if process.poll() is not None:
                        break
                    reason = 'process monitoring interrupted'
                except (psutil.Error, OSError) as error:
                    reason = f'resource monitoring failed: {error}'
                if reason:
                    _stop(process)
                    break
                time.sleep(1)
        finally:
            if process.poll() is None:
                _stop(process)
    return {'returncode': process.returncode, 'stop_reason': reason,
            'elapsed_s': time.monotonic() - started, 'peak_process_rss_bytes': peak}


def _completed(path, case):
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text())
    except (ValueError, OSError):
        return False
    if not isinstance(report, dict):
        return False
    observations = report.get('reports', {})
    if not isinstance(observations, dict):
        return False
    observation = observations.get(case, {})
    environment = report.get('environment', {})
    if not isinstance(observation, dict) or not isinstance(environment, dict):
        return False
    return (report.get('selection_complete') is True
            and report.get('release_complete') is False
            and report.get('reference_only') is False
            and environment.get('requested_device') == 'cpu'
            and report.get('policy') == CPU_FP32_POLICY
            and not report.get('failures')
            and set(report.get('reports', {})) == {case}
            and set(observation.get('routes', {})) == {'legacy', 'adapter', 'native'})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resnet18', required=True)
    parser.add_argument('--gpt2', required=True)
    parser.add_argument('--out', required=True, help='new directory for reports, logs, and scratch files')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--min-free-memory-gib', type=_positive, default=16.)
    parser.add_argument('--max-rss-gib', type=_positive, default=14.)
    parser.add_argument('--reserve-memory-gib', type=_positive, default=1.)
    parser.add_argument('--timeout-per-case-s', type=_positive, default=7200.)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.reserve_memory_gib >= args.min_free_memory_gib:
        parser.error('The memory reserve must be below the launch threshold')
    if os.name != 'posix':
        parser.error('This launcher requires POSIX process-group termination')
    out = Path(args.out).expanduser().resolve()
    commands = {}
    for index, case in enumerate(VISION_CASES):
        folder = out / f'{index + 1}-{case}'
        commands[case] = [
            sys.executable, '-m', 'brainscore.validation.run_parity',
            '--resnet18', str(Path(args.resnet18).expanduser().resolve()),
            '--gpt2', str(Path(args.gpt2).expanduser().resolve()),
            '--only', case, '--device', 'cpu', '--threads', str(args.threads),
            '--policy', CPU_FP32_POLICY, '--scratch-dir', str(folder),
            '--out', str(folder / 'parity.json'),
        ]
    result = {'commands': commands, 'cases': {}, 'vision_complete': False,
              'release_complete': False, 'limits': {
                  'minimum_available_gib': args.min_free_memory_gib,
                  'maximum_process_rss_gib': args.max_rss_gib,
                  'system_reserve_gib': args.reserve_memory_gib,
                  'timeout_per_case_s': args.timeout_per_case_s,
              }}
    if not args.execute:
        print(json.dumps(result, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=False)

    def save():
        path = out / 'summary.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(path)

    save()
    for case, command in commands.items():
        available = psutil.virtual_memory().available
        if available < args.min_free_memory_gib * GIB:
            result['cases'][case] = {'passed': False, 'stop_reason': 'insufficient available RAM',
                                     'available_bytes': available}
            save()
            return 1
        folder = Path(command[-1]).parent
        folder.mkdir()
        result['cases'][case] = {'status': 'running', 'passed': False}
        save()
        print(f'Running {case}; log: {folder / "scoring.log"}', flush=True)
        try:
            status = _run_case(
                command, directory=folder, timeout_s=args.timeout_per_case_s,
                max_rss_bytes=args.max_rss_gib * GIB,
                min_available_bytes=args.reserve_memory_gib * GIB,
            )
        except BaseException as error:
            result['cases'][case] = {'passed': False, 'stop_reason': type(error).__name__}
            save()
            raise
        status['passed'] = (status['returncode'] == 0 and not status['stop_reason']
                            and _completed(folder / 'parity.json', case))
        if status['passed']:
            checkpoints = json.loads((folder / 'parity.json').read_text()).get('checkpoints')
            if not checkpoints or checkpoints != result.setdefault('checkpoints', checkpoints):
                status.update(passed=False, stop_reason='checkpoint identity missing or changed between cases')
        result['cases'][case] = status
        save()
        print(f'{case}: {"PASS" if status["passed"] else "FAIL"}', flush=True)
        if not status['passed']:
            return 1
    result['vision_complete'] = True
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
