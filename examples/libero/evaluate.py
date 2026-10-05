"""Run the pinned OpenPI evaluator with per-trial receipts and visible errors.

Run in the LIBERO environment (Python 3.8 is supported). UMI runs in a separate
bridge process. Task logic, initial states, preprocessing, and scoring are
provided by the upstream evaluator, not reimplemented here.
"""
import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np

OPENPI_REVISION = '215abfb217dbac7d5f1273282331b9b1866c0479'
LIBERO_REVISION = 'f78abd68ee283de9f9be3c8f7e2a9ad60246e95c'


class TrialAudit:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.trials = []
        self.current = None
        self.errors = []

    def finish(self):
        if self.current is not None:
            self.trials.append(self.current)
            with (self.directory / 'trials.jsonl').open('a') as handle:
                handle.write(json.dumps(self.current) + '\n')
            self.current = None

    def wrap(self, env, task):
        audit = self

        class AuditedEnvironment:
            def reset(self):
                audit.finish()
                audit.current = {'task': task, 'trial': sum(t['task'] == task for t in audit.trials),
                                 'steps': 0, 'success': False, 'errors': []}
                return env.reset()

            def set_init_state(self, state):
                state = np.asarray(state)
                audit.current['initial_state_sha256'] = hashlib.sha256(state.tobytes()).hexdigest()
                return env.set_init_state(state)

            def step(self, action):
                result = env.step(action)
                audit.current['steps'] += 1
                audit.current['success'] = bool(result[2])
                return result

        return AuditedEnvironment()


class ErrorAudit(logging.Handler):
    def __init__(self, audit):
        super().__init__(logging.ERROR)
        self.audit = audit

    def emit(self, record):
        message = record.getMessage()
        self.audit.errors.append(message)
        if self.audit.current is not None:
            self.audit.current['errors'].append(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openpi', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--route', choices=('reference', 'umi'), required=True)
    parser.add_argument('--suite', choices=('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'),
                        default='libero_spatial')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    if not 1 <= args.trials <= 50:
        parser.error('Use 1 to 50 trials per task')
    for root, expected in ((args.openpi, OPENPI_REVISION),
                           (args.openpi / 'third_party/libero', LIBERO_REVISION)):
        git = ['git', '-c', 'safe.directory=' + str(root.resolve()), '-C', str(root)]
        actual = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise RuntimeError('Unexpected revision at ' + str(root))
        if subprocess.check_output(git + ['diff', '--name-only', 'HEAD'], text=True).strip():
            raise RuntimeError('Modified upstream source at ' + str(root))
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.openpi / 'examples/libero/main.py'
    spec = importlib.util.spec_from_file_location('openpi_libero_reference', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    audit = TrialAudit(args.out)
    errors = ErrorAudit(audit)
    logging.getLogger().addHandler(errors)
    make_environment = module._get_libero_env
    environments = []

    def audited_environment(task, resolution, seed):
        # Close each completed task's environment; this does not alter trials.
        for previous in environments:
            previous.close()
        environments.clear()
        env, description = make_environment(task, resolution, seed)
        environments.append(env)
        return audit.wrap(env, task.name), description

    module._get_libero_env = audited_environment
    write_video = module.imageio.mimwrite

    def unique_video(path, frames, **kwargs):
        # Upstream filenames overwrite repeated trials of the same task.
        path = Path(path)
        return write_video(path.with_name('%04d-%s' % (len(audit.trials), path.name)), frames, **kwargs)

    module.imageio.mimwrite = unique_video
    config = module.Args(host='127.0.0.1', port=args.port, task_suite_name=args.suite,
                         num_trials_per_task=args.trials, video_out_path=str(args.out / 'videos'))
    completed = False
    try:
        module.eval_libero(config)
        completed = True
    finally:
        audit.finish()
        for env in environments:
            env.close()
        logging.getLogger().removeHandler(errors)
        expected = module.benchmark.get_benchmark_dict()[args.suite]().n_tasks * args.trials
        valid = completed and not audit.errors and len(audit.trials) == expected
        report = {'status': 'complete' if valid else 'failed', 'route': args.route,
                  'suite': args.suite, 'trials_per_task': args.trials,
                  'episodes': len(audit.trials), 'expected_episodes': expected,
                  'successes': sum(t['success'] for t in audit.trials),
                  'success_rate': (sum(t['success'] for t in audit.trials) / len(audit.trials)
                                   if audit.trials else None),
                  'errors': audit.errors, 'openpi_revision': OPENPI_REVISION,
                  'libero_revision': LIBERO_REVISION,
                  'evaluator_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'settings': vars(config), 'python': sys.version,
                  'qualification': 'smoke-only' if args.trials < 50 else 'single-suite-evaluation'}
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if not valid:
        raise SystemExit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
