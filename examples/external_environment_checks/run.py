"""Model-free CPU checks for long runs, tool effects and independent workers.

This exercises a fixed-weight fixture, not a trained policy or scientific benchmark.
Run from an installed UMI environment: python run.py --out results --steps 1000.
"""
import argparse
from collections import deque
from contextlib import contextmanager
import hashlib
import json
import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import resource
import subprocess
import sys
import time
import tracemalloc

import numpy as np
import torch
from brainscore.experiments import (
    Experiment, RecordActivity, RecordInputsOutputs, ScaleActivity,
    SessionProtocol, TorchInstrumentation,
)
from brainscore_core.environment import ArraySpec, MappingSpec
from brainscore_core.events import EnvironmentResponse, EnvironmentStep
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.streaming_helpers import EnvironmentSession


class CounterEnvironment:
    """A deterministic fixture; action hashes compare all commands without retaining them."""
    def __init__(self, steps: int):
        self.steps = steps
        self.index = 0
        self.actions = hashlib.sha256()
        self.closed = False

    def observation_spec(self):
        return MappingSpec({'state': ArraySpec((1,), dtype='float32')})

    def action_spec(self):
        return ArraySpec((1,), dtype='float32')

    def observation(self):
        return EnvironmentStep(
            observation={'state': np.array([1 + self.index % 17], dtype=np.float32)},
            step_num=self.index,
            is_last=self.index == self.steps,
        )

    def reset(self):
        return self.observation()

    def step(self, action):
        self.actions.update(np.asarray(action).tobytes())
        self.index += 1
        return self.observation()

    def close(self):
        self.closed = True


def _run(directory: Path, steps: int, report: dict) -> dict:
    if steps < 1:
        raise ValueError('steps must be positive')
    layer = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(2)
    latency = deque(maxlen=1024)

    def policy(step):
        started = time.perf_counter()
        with torch.no_grad():
            action = layer(torch.from_numpy(step.observation['state'])).numpy().copy()
        latency.append(time.perf_counter() - started)
        return EnvironmentResponse(action=action)

    subject = BrainScoreModel(identifier='fixed-weight-fixture', policy=policy)
    reports = report['conditions']
    for condition in ('baseline', 'recording', 'scaled', 'restored'):
        environment = CounterEnvironment(steps)
        sessions = []
        latency.clear()

        @contextmanager
        def factory(trial):
            with EnvironmentSession(environment, require_specs=True, history_limit=0) as session:
                sessions.append(session)
                yield session

        tools = [] if condition == 'baseline' else [RecordInputsOutputs(max_bytes=128 * 1024**2)]
        if condition == 'scaled':
            tools.append(ScaleActivity(['policy'], factor=0.5))
        if condition != 'baseline':
            tools.append(RecordActivity(['policy']))
        tracemalloc.start()
        start = time.perf_counter()
        result = Experiment(
            subject=subject,
            protocol=SessionProtocol(
                identifier='counter-episode', session_factory=factory,
                input_channels=('observation',), output_channels=('motor',),
                collect_outputs=False,
            ),
            tools=tools,
            instrumentation=TorchInstrumentation({'policy': layer}),
            output_dir=directory / condition,
        ).run()
        elapsed = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        require(result.manifest['status'] == 'complete' and environment.closed, 'Incomplete run or unclosed environment')
        require(environment.index == steps and not layer._forward_hooks, 'Missing steps or leaked hooks')
        require(all(not history for session in sessions for history in (
            session.input_events, session.action_events, session.emitted)), 'Session retained history')
        activity_count = 0
        if tools:
            for event in result.record.events():
                if event['kind'] == 'activity':
                    activity_count += 1
                    actual = event['payload']['value']['array']
                    multiplier = 1 if condition == 'scaled' else 2
                    np.testing.assert_array_equal(actual, [multiplier * (1 + (activity_count - 1) % 17)])
            require(activity_count == steps, 'Activity event count differs from step count')
        reports[condition] = {
            'steps': steps, 'action_sha256': environment.actions.hexdigest(),
            'seconds_with_tracemalloc': elapsed, 'python_peak_bytes': peak,
            'artifact_bytes': sum(p.stat().st_size for p in result.directory.rglob('*') if p.is_file()),
            'policy_latency_last_1024_seconds': dict(zip(('p50', 'p95'), np.percentile(latency, [50, 95]).tolist())),
            'activity_events_verified': activity_count,
        }
    require(reports['baseline']['action_sha256'] == reports['recording']['action_sha256'] == reports['restored']['action_sha256'], 'Recording or restoration changed actions')
    require(reports['scaled']['action_sha256'] != reports['baseline']['action_sha256'], 'Scaling did not change actions')
    repeated = check_repeated_failure(subject, layer, directory, min(steps, 10))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report.update(status='complete', repeated_episodes=repeated,
                  process_peak_rss_bytes=rss if sys.platform == 'darwin' else rss * 1024)
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def check_repeated_failure(subject, layer, directory: Path, steps: int) -> dict:
    """Reuse one subject across episodes, an interrupted intervention, and restoration."""
    summaries = {}
    for condition in ('before_failure', 'failure', 'after_failure'):
        environments = []

        class Episode(CounterEnvironment):
            def step(self, action):
                if condition == 'failure' and len(environments) == 3 and self.index == 1:
                    raise RuntimeError('Injected environment failure after two complete episodes')
                return super().step(action)

        @contextmanager
        def factory(trial):
            environment = Episode(max(steps, 2))
            environments.append(environment)
            with EnvironmentSession(
                environment, require_specs=True, history_limit=0,
            ) as session:
                yield session

        tools = [RecordInputsOutputs()]
        if condition == 'failure':
            tools.append(ScaleActivity(['policy'], factor=0.5))
        experiment = Experiment(
            subject=subject,
            protocol=SessionProtocol(
                identifier='repeated-episodes', session_factory=factory,
                trials=(0, 1, 2), input_channels=('observation',),
                output_channels=('motor',), collect_outputs=False,
            ),
            tools=tools,
            instrumentation=TorchInstrumentation({'policy': layer}),
            output_dir=directory / condition,
        )
        try:
            experiment.run()
        except RuntimeError as error:
            if condition != 'failure' or 'Injected environment failure' not in str(error):
                raise
        else:
            require(condition != 'failure', 'Injected failure was swallowed')
        manifest = json.loads((experiment.output_dir / 'experiment.json').read_text())
        expected = 'failed' if condition == 'failure' else 'complete'
        require(manifest['status'] == expected, 'Incorrect failure status')
        require(not layer._forward_hooks, 'Hooks leaked after repeated episodes')
        require(len(environments) == 3 and all(e.closed for e in environments),
                'Repeated episodes did not close all environments')
        if condition == 'failure':
            require(bool(manifest.get('error_chain')), 'Failure chain was not saved')
        summaries[condition] = {
            'status': manifest['status'],
            'action_sha256': [e.actions.hexdigest() for e in environments],
            'completed_steps': [e.index for e in environments],
        }
    require(summaries['before_failure']['action_sha256'] == summaries['after_failure']['action_sha256'],
            'Failure or intervention leaked into the next experiment')
    return summaries


def require(condition: bool, message: str) -> None:
    """Qualification checks must also run under python -O."""
    if not condition:
        raise RuntimeError(message)


def source_state(path: Path) -> dict:
    """Git state of the checkout holding path. source_commit is null unless the file is
    tracked and the checkout has no modified, staged or untracked files, so a commit
    is reported only when it alone identifies the code that ran."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(path.parent), *args],
                                       text=True, stderr=subprocess.DEVNULL)
    try:
        head = git('rev-parse', 'HEAD').strip()
    except (OSError, subprocess.CalledProcessError):
        return {'source_commit': None, 'git_head': None, 'source_dirty': None}
    try:
        git('ls-files', '--error-unmatch', '--', path.name)
    except subprocess.CalledProcessError:
        # E.g. an installed package inside an unrelated (ignored) checkout.
        return {'source_commit': None, 'git_head': head, 'source_dirty': None}
    dirty = bool(git('status', '--porcelain', '--untracked-files=all').strip())
    return {'source_commit': None if dirty else head, 'git_head': head, 'source_dirty': dirty}


def provenance() -> dict:
    import brainscore
    import brainscore_core
    modules = {}
    for name, module in (('brainscore', brainscore), ('brainscore-core', brainscore_core)):
        path = Path(module.__file__).resolve()
        try:
            installed_version = version(name)
        except PackageNotFoundError:
            installed_version = None
        modules[name] = {
            'imported_file': str(path), **source_state(path),
            'installed_distribution_version': installed_version,
        }
    return {
        'python': sys.version, 'platform': platform.platform(),
        'torch': torch.__version__, 'modules': modules,
        'version_note': 'Installed distribution metadata may differ from source imports; '
                        'imported_file identifies the code used. source_commit is null '
                        'for a modified or untracked checkout; see git_head and source_dirty.',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def run(directory: Path, steps: int = 1000) -> dict:
    directory = Path(directory)
    if tracemalloc.is_tracing():
        raise RuntimeError('Run this allocation check without an existing tracemalloc session')
    # Create outside the failure handler: never write into another run's directory.
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        'status': 'running', 'conditions': {},
        'scope': 'fixed-weight stateless CPU fixture; no trained-policy qualification',
        'provenance': provenance(),
    }
    try:
        return _run(directory, steps, report)
    except BaseException as error:
        tracemalloc.stop()
        report.update(status='failed', error=str(error), error_type=type(error).__name__)
        try:
            (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        except OSError as write_error:
            error.add_note(f'Could not save qualification report: {write_error}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1 or args.workers < 1:
        parser.error('steps and workers must be positive')
    if args.workers == 1:
        run(args.out, args.steps)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    processes = []
    worker_report = {'status': 'running', 'workers': args.workers}
    try:
        for index in range(args.workers):
            processes.append(subprocess.Popen([
                sys.executable, *(['-O'] if sys.flags.optimize else []), str(Path(__file__).resolve()), '--out', str(args.out / f'worker-{index}'),
                '--steps', str(args.steps),
            ]))
        codes = [process.wait() for process in processes]
        worker_report['return_codes'] = codes
        if any(codes):
            raise RuntimeError(f'Worker failures: {codes}')
    except BaseException as error:
        worker_report.update(status='failed', error=str(error))
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        if worker_report['status'] == 'failed':
            (args.out / 'report.json').write_text(json.dumps(worker_report, indent=2) + '\n')
    try:
        reports = [json.loads((args.out / f'worker-{i}' / 'report.json').read_text()) for i in range(args.workers)]
        require(len({r['conditions']['baseline']['action_sha256'] for r in reports}) == 1, 'Independent workers disagree')
        (args.out / 'workers.json').write_text(json.dumps(reports, indent=2) + '\n')
        worker_report.update(status='complete')
        (args.out / 'report.json').write_text(json.dumps(worker_report, indent=2) + '\n')
    except BaseException as error:
        worker_report.update(status='failed', error=str(error))
        (args.out / 'report.json').write_text(json.dumps(worker_report, indent=2) + '\n')
        raise



if __name__ == '__main__':
    main()
