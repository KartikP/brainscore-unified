"""Run the pinned OpenPI evaluator with per-trial receipts and visible errors.

Run in the LIBERO environment (Python 3.8 is supported). UMI runs in a separate
bridge process. Task logic, initial states, preprocessing, and scoring are
provided by the upstream evaluator, not reimplemented here.

The upstream websocket client is replaced by BoundedClient: identical request
bytes and reply parsing, but bounded connect, exchange (send plus reply) and
close, and no resend after an uncertain exchange. With serve_tool_bridge.py it
also exchanges a start message, per-call annotations, a receipt and a finish
acknowledgment (see the README).

Deadlines come from brainscore/experiments/backends/socket_deadline.py, loaded
by path from the same unified checkout (standard library only), so the
brainscore package is not imported here.
"""
import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

OPENPI_REVISION = '215abfb217dbac7d5f1273282331b9b1866c0479'
LIBERO_REVISION = 'f78abd68ee283de9f9be3c8f7e2a9ad60246e95c'
RECEIPT_PROTOCOL = 'brainscore.libero.receipt.v1'
# Copied from eval_libero at OPENPI_REVISION (checked in test_libero_transport.py).
MAX_STEPS = {'libero_spatial': 220, 'libero_object': 280, 'libero_goal': 300,
             'libero_10': 520, 'libero_90': 400}
# Per-episode fields sent in the receipt, plus an error count.
LEDGER_FIELDS = ('task', 'trial', 'calls', 'full_calls', 'last_executed', 'executed_total',
                 'wait_steps', 'success', 'ended_by', 'prefix_match', 'execution_unknown')


def _load_socket_deadline():
    path = Path(__file__).resolve().parents[2] / 'brainscore/experiments/backends/socket_deadline.py'
    spec = importlib.util.spec_from_file_location('umi_socket_deadline', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


socket_deadline = _load_socket_deadline()


class TrialAudit:
    """Per-episode outcome plus a compact ledger of chunks and executed actions.

    Only the current chunk is kept. A received chunk is not evidence that any
    of its actions ran: an action is counted only after env.step returns, and a
    step that raises leaves execution_unknown set.
    """

    def __init__(self, directory, *, horizon, num_steps_wait, max_steps):
        self.directory = Path(directory)
        self.horizon, self.num_steps_wait, self.max_steps = horizon, num_steps_wait, max_steps
        self.trials = []
        self.current = None
        self.errors = []
        self.chunk = None
        self.calls = 0

    def finish(self):
        if self.current is not None:
            self.current['ended_by'] = self.ending(self.current)
            self.trials.append(self.current)
            self.current = None
            with (self.directory / 'trials.jsonl').open('a') as handle:
                handle.write(json.dumps(self.trials[-1]) + '\n')

    def ending(self, trial):
        """success, step_limit, or error for anything upstream may have swallowed.

        Upstream stops without error only on done or after max_steps policy
        steps, so a stop short of max_steps without done is an error even if no
        error was logged. The last chunk may end before or exactly at H.
        """
        horizon, last = self.horizon, trial['last_executed']
        consistent = (trial['calls'] > 0 and 1 <= last <= horizon and trial['prefix_match']
                      and not trial['execution_unknown'] and not trial['errors']
                      and trial['wait_steps'] == self.num_steps_wait
                      and trial['executed_total'] == horizon * (trial['calls'] - 1) + last)
        if consistent and trial['success']:
            return 'success'
        if consistent and trial['executed_total'] == self.max_steps:
            return 'step_limit'
        return 'error'

    def wrap(self, env, task):
        audit = self

        class AuditedEnvironment:
            def reset(self):
                audit.finish()
                audit.chunk = None
                audit.current = {'task': task, 'trial': sum(t['task'] == task for t in audit.trials),
                                 'steps': 0, 'success': False, 'errors': [],
                                 'calls': 0, 'full_calls': 0, 'last_executed': 0,
                                 'executed_total': 0, 'wait_steps': 0, 'prefix_match': True,
                                 'execution_unknown': False}
                return env.reset()

            def set_init_state(self, state):
                state = np.asarray(state)
                audit.current['initial_state_sha256'] = hashlib.sha256(state.tobytes()).hexdigest()
                return env.set_init_state(state)

            def step(self, action):
                current = audit.current
                if current['calls']:
                    index, chunk = current['last_executed'], audit.chunk
                    if index >= len(chunk) or not np.array_equal(
                            np.asarray(action, dtype=chunk.dtype), chunk[index]):
                        current['prefix_match'] = False
                # Stays set if env.step raises: its effect on the simulator is unknown.
                current['execution_unknown'] = True
                result = env.step(action)
                current['execution_unknown'] = False
                current['steps'] += 1
                current['success'] = bool(result[2])
                if current['calls']:
                    current['last_executed'] += 1
                    current['executed_total'] += 1
                    current['full_calls'] += current['last_executed'] == audit.horizon
                else:
                    current['wait_steps'] += 1
                return result

        return AuditedEnvironment()

    def wrap_policy(self, client):
        audit = self

        class AuditedPolicy:
            def infer(self, observation):
                current = audit.current
                annotation = {'index': audit.calls, 'task': current['task'], 'trial': current['trial']}
                result = client.infer(observation, annotation)
                audit.chunk = np.asarray(result['actions'])[:audit.horizon]
                audit.calls += 1
                current['calls'] += 1
                current['last_executed'] = 0
                return result

            def reset(self):
                pass

        return AuditedPolicy()


class ErrorAudit(logging.Handler):
    def __init__(self, audit):
        super().__init__(logging.ERROR)
        self.audit = audit

    def emit(self, record):
        message = record.getMessage()
        self.audit.errors.append(message)
        if self.audit.current is not None:
            self.audit.current['errors'].append(message)


class BoundedClient:
    """Upstream WebsocketClientPolicy bytes and parsing, with bounded waits.

    Connection refusals are retried until the deadline, before any inference.
    Each exchange (send plus reply) has one deadline; when it runs out the
    socket is shut down, so a peer that stops reading cannot hold a send. After
    any failed exchange the connection is closed and never reused, so a late
    reply cannot be read as the answer to another request. Closing is bounded
    by close_timeout. Inference already running on the peer is not cancelled.
    """

    def __init__(self, uri, *, connect_timeout, infer_timeout,
                 close_timeout=socket_deadline.CLOSE_TIMEOUT):
        _positive(connect_timeout)
        _positive(infer_timeout)
        _positive(close_timeout)
        from openpi_client import msgpack_numpy
        import websockets.sync.client
        self._codec, self._packer = msgpack_numpy, msgpack_numpy.Packer()
        self.infer_timeout, self.close_timeout = infer_timeout, close_timeout
        self.broken = False
        deadline = time.monotonic() + connect_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Policy server unavailable at ' + uri)
            try:
                self._ws = websockets.sync.client.connect(
                    uri, compression=None, max_size=None, open_timeout=remaining,
                    close_timeout=close_timeout)
                break
            except ConnectionRefusedError:
                time.sleep(min(1.0, remaining))
        self.metadata = self._exchange(None, max(deadline - time.monotonic(), 0.001))
        self.receipt = isinstance(self.metadata, dict) and 'receipt_protocol' in self.metadata
        if self.receipt:
            run_id = self.metadata.get('umi_run_id')
            if self.metadata['receipt_protocol'] != RECEIPT_PROTOCOL or not isinstance(run_id, str) or not run_id:
                self.close()
                raise RuntimeError('Unsupported bridge receipt protocol')
            self.run_id = run_id
            self.bridge = {key: self.metadata.get(key)
                           for key in ('receipt_protocol', 'umi_run_id', 'experiment_tools')}

    def _exchange(self, data, timeout):
        if self.broken:
            raise RuntimeError('Connection closed after a failed exchange; requests are not resent')
        try:
            with socket_deadline.deadline(self._ws, timeout) as remaining:
                if data is not None:
                    self._ws.send(data)
                reply = self._ws.recv(timeout=remaining())
            if isinstance(reply, str):
                raise RuntimeError('Error in inference server:\n' + reply)
            return self._codec.unpackb(reply)
        except BaseException:
            self.close()
            raise

    def infer(self, observation, annotation=None):
        if not self.receipt:
            return self._exchange(self._packer.pack(observation), self.infer_timeout)
        reply = self._exchange(self._packer.pack(dict(observation, **{'umi/call': annotation})),
                               self.infer_timeout)
        if isinstance(reply, dict) and 'umi/finish_ack' in reply:
            self.close()
            raise RuntimeError('Bridge stopped the run: %r' % (reply,))
        return reply

    def control(self, message, timeout):
        _positive(timeout)
        return self._exchange(self._packer.pack(message), timeout)

    def close(self):
        """Bounded and never raises; repeated calls do nothing."""
        if not self.broken:
            self.broken = True
            socket_deadline.close(self._ws, self.close_timeout)


_positive = socket_deadline.positive_seconds


def _seconds(value):
    try:
        return _positive(float(value))
    except ValueError:
        raise argparse.ArgumentTypeError('Use a positive, finite number of seconds')


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _finish_ack(client, digest, status, audit, timeout):
    """Send the receipt and wait for the bridge's verdict; never raises."""
    if digest is None or client.broken:
        client.close()
        return {'ack': 'not_sent', 'run_id': client.run_id}
    episodes = [dict({key: trial[key] for key in LEDGER_FIELDS}, errors=len(trial['errors']))
                for trial in audit.trials]
    receipt = {'version': RECEIPT_PROTOCOL, 'run_id': client.run_id, 'report_sha256': digest,
               'status': status, 'episodes': episodes}
    try:
        reply = client.control({'umi/receipt': receipt}, timeout)
    except TimeoutError:
        return {'ack': 'timeout', 'run_id': client.run_id}
    except Exception as error:
        logging.warning('No finish acknowledgment: %r', error)
        return {'ack': 'closed', 'run_id': client.run_id}
    finally:
        client.close()
    reply = reply if isinstance(reply, dict) else {}
    ack = reply.get('umi/finish_ack')
    manifest = reply.get('manifest_sha256') if ack == 'complete' else None
    if (reply.get('run_id') != client.run_id or ack not in ('complete', 'failed')
            or (ack == 'complete' and not (isinstance(manifest, str) and len(manifest) == 64))):
        return {'ack': 'invalid', 'run_id': client.run_id}
    return {'ack': ack, 'run_id': client.run_id, 'manifest_sha256': manifest}


def _finalize(args, config, source, audit, environments, handler, session, completed, expected):
    """Write report.json, then exchange the receipt; returns True only if both succeeded."""
    cleanup_errors = []
    steps = [('trial audit', audit.finish)] + [('environment', env.close) for env in environments]
    steps.append(('error handler', lambda: logging.getLogger().removeHandler(handler)))
    for name, step in steps:
        try:
            step()
        except Exception as error:
            cleanup_errors.append('%s: %r' % (name, error))
    keys = [(trial['task'], trial['trial']) for trial in audit.trials]
    valid = (completed and not audit.errors and not cleanup_errors and keys == expected
             and all(trial['ended_by'] != 'error' for trial in audit.trials))
    client = session.get('client')
    receipt = client is not None and client.receipt
    successes = sum(trial['success'] for trial in audit.trials)
    report = {'status': 'complete' if valid else 'failed', 'route': args.route,
              'suite': args.suite, 'trials_per_task': args.trials,
              'episodes': len(audit.trials), 'expected_episodes': len(expected),
              'successes': successes,
              'success_rate': successes / len(audit.trials) if audit.trials else None,
              'errors': audit.errors, 'cleanup_errors': cleanup_errors,
              'execution_horizon': audit.horizon, 'num_steps_wait': audit.num_steps_wait,
              'max_steps': audit.max_steps, 'bridge': client.bridge if receipt else None,
              'openpi_revision': OPENPI_REVISION, 'libero_revision': LIBERO_REVISION,
              'evaluator_sha256': _sha256(Path(source).read_bytes()),
              'settings': vars(config), 'python': sys.version,
              'qualification': 'smoke-only' if args.trials < 50 else 'single-suite-evaluation'}
    digest = None
    try:
        data = (json.dumps(report, indent=2) + '\n').encode()
        (args.out / 'report.json').write_bytes(data)
        digest = _sha256(data)
    except Exception as error:
        logging.error('report.json was not saved: %r', error)
    if not receipt:
        if client is not None:
            client.close()
        return valid and digest is not None
    finish = _finish_ack(client, digest, report['status'], audit, args.finish_timeout)
    try:
        (args.out / 'finish.json').write_text(json.dumps(finish, indent=2) + '\n')
    except Exception as error:
        logging.error('finish.json was not saved: %r', error)
        return False
    return valid and digest is not None and finish['ack'] == 'complete'


def run(module, source, args):
    """Evaluate with an already loaded upstream module; exits nonzero unless complete."""
    args.out.mkdir(parents=True, exist_ok=False)
    suite = module.benchmark.get_benchmark_dict()[args.suite]()
    tasks = [suite.get_task(index).name for index in range(suite.n_tasks)]
    expected = [(task, trial) for task in tasks for trial in range(args.trials)]
    config = module.Args(host='127.0.0.1', port=args.port, task_suite_name=args.suite,
                         num_trials_per_task=args.trials, video_out_path=str(args.out / 'videos'))
    audit = TrialAudit(args.out, horizon=config.replan_steps,
                       num_steps_wait=config.num_steps_wait, max_steps=MAX_STEPS[args.suite])
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
    session = {}

    def connect(host, port):
        client = BoundedClient('ws://%s:%d' % (host, port), connect_timeout=args.connect_timeout,
                               infer_timeout=args.infer_timeout)
        session['client'] = client
        if client.receipt:
            start = {'version': RECEIPT_PROTOCOL, 'run_id': client.run_id, 'suite': args.suite,
                     'trials_per_task': args.trials, 'execution_horizon': config.replan_steps,
                     'num_steps_wait': config.num_steps_wait, 'max_steps': MAX_STEPS[args.suite],
                     'tasks': tasks}
            reply = client.control({'umi/start': start}, args.control_timeout)
            if reply != {'umi/start_ack': 'accepted', 'run_id': client.run_id}:
                client.close()
                raise RuntimeError('Bridge did not accept the evaluation plan')
        return audit.wrap_policy(client)

    module._websocket_client_policy = SimpleNamespace(WebsocketClientPolicy=connect)
    completed = valid = False
    try:
        module.eval_libero(config)
        completed = True
    finally:
        # Never raises, so an original exception keeps propagating.
        try:
            valid = _finalize(args, config, source, audit, environments, errors, session,
                              completed, expected)
        except Exception:
            logging.exception('Evaluation finalization failed')
    if not valid:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openpi', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--route', choices=('reference', 'umi'), required=True)
    parser.add_argument('--suite', choices=('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'),
                        default='libero_spatial')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--port', type=int, default=8001)
    # Conservative operational defaults, not qualified performance budgets.
    parser.add_argument('--connect-timeout', type=_seconds, default=120)
    parser.add_argument('--infer-timeout', type=_seconds, default=600)
    parser.add_argument('--control-timeout', type=_seconds, default=60)
    parser.add_argument('--finish-timeout', type=_seconds, default=600)
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
    source = args.openpi / 'examples/libero/main.py'
    spec = importlib.util.spec_from_file_location('openpi_libero_reference', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    run(module, source, args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
