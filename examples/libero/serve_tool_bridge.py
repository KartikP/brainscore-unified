"""Connect the official LIBERO evaluator to the OpenPI tool server.

Run one evaluator per bridge, and only examples/libero/evaluate.py: the bridge
requires its start message, per-call annotations and final receipt. The
evaluator owns observations, scheduling, resets and scoring; Brain-Score
records calls and applies the selected tools. A recorded call shows that a
chunk was produced and sent, not that any of its actions ran in the simulator.

The run is complete only after a valid receipt is accepted and Experiment.run
has saved its records. Only then is the 'complete' acknowledgment sent.
"""

import argparse
import hashlib
import logging
import math
from pathlib import Path
import threading
from brainscore.experiments import (
    Experiment,
    CallableProtocol,
    RecordInputsOutputs,
    RecordActivity,
    Ablate,
    OpenPIPolicyClient,
    RemoteOpenPIInstrumentation,
)
from brainscore.experiments.backends import socket_deadline
from brainscore.harnesses.libero_policy import LiberoChunkPolicy
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.events import EnvironmentStep, Selection

RECEIPT_PROTOCOL = 'brainscore.libero.receipt.v1'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
MODES = ('baseline', 'recording', 'ablation')
START_FIELDS = {'version', 'run_id', 'suite', 'trials_per_task', 'execution_horizon',
                'num_steps_wait', 'max_steps', 'tasks'}
EPISODE_FIELDS = {'task', 'trial', 'calls', 'full_calls', 'last_executed', 'executed_total',
                  'wait_steps', 'success', 'ended_by', 'prefix_match', 'execution_unknown', 'errors'}
COUNTS = ('calls', 'full_calls', 'last_executed', 'executed_total', 'wait_steps', 'errors')
FLAGS = ('success', 'prefix_match', 'execution_unknown')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(value, low=0, high=None):
    return type(value) is int and value >= low and (high is None or value <= high)


def check_start(start, *, run_id, suite, trials_per_task, horizon):
    """Return the episode keys the evaluator declared, in evaluation order.

    Suite, trials and horizon are checked against the bridge's own settings.
    Task names, max_steps and num_steps_wait are evaluator declarations from
    its pinned source; they are checked for consistency with calls and receipt.
    """
    _require(isinstance(start, dict) and set(start) == START_FIELDS, 'Malformed start message')
    _require(start['version'] == RECEIPT_PROTOCOL, 'Unsupported receipt protocol')
    _require(start['run_id'] == run_id, 'Start message names another run')
    _require(start['suite'] == suite and _count(start['trials_per_task'], 1, 50)
             and start['trials_per_task'] == trials_per_task
             and _count(start['execution_horizon'], 1) and start['execution_horizon'] == horizon,
             'Evaluator settings differ from the bridge configuration')
    _require(_count(start['num_steps_wait'], 0, 1000) and _count(start['max_steps'], 1, 1000),
             'Invalid step limits')
    tasks = start['tasks']
    _require(isinstance(tasks, list) and 1 <= len(tasks) <= 90
             and all(isinstance(task, str) and task for task in tasks)
             and len(set(tasks)) == len(tasks), 'Invalid task list')
    return [(task, trial) for task in tasks for trial in range(trials_per_task)]


def check_receipt(receipt, *, run_id, start, keys, calls, horizon):
    """Reject a receipt unless every episode completed consistently with our calls.

    calls[i] is the bridge's own count of chunks sent for keys[i].
    """
    _require(isinstance(receipt, dict) and set(receipt) == {
        'version', 'run_id', 'report_sha256', 'status', 'episodes'}, 'Malformed receipt')
    _require(receipt['version'] == RECEIPT_PROTOCOL, 'Unsupported receipt protocol')
    _require(receipt['run_id'] == run_id, 'Receipt names another run')
    digest = receipt['report_sha256']
    _require(isinstance(digest, str) and len(digest) == 64
             and set(digest) <= set('0123456789abcdef'), 'Invalid report digest')
    _require(receipt['status'] == 'complete', 'Evaluator reported an incomplete evaluation')
    episodes = receipt['episodes']
    _require(isinstance(episodes, list) and len(episodes) == len(keys), 'Wrong number of episodes')
    for row, key, sent in zip(episodes, keys, calls):
        _require(isinstance(row, dict) and set(row) == EPISODE_FIELDS, 'Malformed episode')
        _require(isinstance(row['task'], str) and _count(row['trial'])
                 and all(_count(row[name]) for name in COUNTS)
                 and all(type(row[name]) is bool for name in FLAGS), 'Invalid episode values')
        _require((row['task'], row['trial']) == key, 'Episodes differ from the declared order')
        _require(row['calls'] == sent, 'Episode calls differ from the calls served')
        _require(row['errors'] == 0 and row['prefix_match'] and not row['execution_unknown'],
                 'Episode had errors or unverified actions')
        _require(row['wait_steps'] == start['num_steps_wait'], 'Unexpected wait steps')
        last, total = row['last_executed'], row['executed_total']
        _require(sent >= 1 and 1 <= last <= horizon and total == horizon * (sent - 1) + last
                 and row['full_calls'] == sent - 1 + (last == horizon),
                 'Executed actions do not match whole chunks')
        _require(total <= start['max_steps'], 'More actions than the step limit')
        _require((row['ended_by'] == 'success' and row['success'])
                 or (row['ended_by'] == 'step_limit' and not row['success']
                     and total == start['max_steps']), 'Episode did not end by success or step limit')


def _send(socket, codec, message, timeout):
    """Send once within timeout; on timeout the connection is shut down."""
    with socket_deadline.deadline(socket, timeout):
        socket.send(codec.packb(message))


def _best_effort(socket, codec, message, timeout):
    """Send once; a lost acknowledgment never changes the saved record."""
    try:
        _send(socket, codec, message, timeout)
    except Exception as error:
        logging.warning('Could not send %s: %r', message, error)


class Bridge:
    def __init__(self, upstream, output, mode, *, suite, trials_per_task, horizon,
                 idle_timeout=900):
        _require(mode in MODES and suite in SUITES, 'Unknown mode or suite')
        _require(_count(trials_per_task, 1, 50) and _count(horizon, 1),
                 'trials_per_task must be an integer from 1 to 50 and horizon a positive integer')
        _require(type(idle_timeout) in (int, float) and math.isfinite(idle_timeout)
                 and idle_timeout > 0, 'idle_timeout must be positive and finite')
        self.upstream, self.output, self.mode = upstream, Path(output), mode
        self.suite, self.trials_per_task, self.horizon = suite, trials_per_task, horizon
        self.idle_timeout = idle_timeout
        self.lock = threading.Lock()
        self.connection = 0

    def handler(self, socket):
        from openpi_client import msgpack_numpy as codec

        if not self.lock.acquire(blocking=False):
            socket_deadline.close(socket, code=1013, reason='Dedicated evaluator already connected')
            return
        state = {}
        try:
            self.connection += 1
            with OpenPIPolicyClient(self.upstream) as client:
                subject = BrainScoreModel(
                    'pi05-libero',
                    policy=LiberoChunkPolicy(client, execution_horizon=self.horizon),
                )
                target = Selection(layer='action_out_proj', indices=[0])
                tools = [RecordInputsOutputs()]
                if self.mode == 'ablation':
                    tools.extend(
                        [
                            RecordActivity([target], when='before', name='before'),
                            Ablate([target]),
                        ]
                    )
                if self.mode != 'baseline':
                    tools.append(RecordActivity([target], name='after'))
                experiment = Experiment(
                    subject=subject,
                    protocol=CallableProtocol(
                        'libero-evaluator',
                        lambda model, context: self.serve(socket, codec, model, context, state),
                    ),
                    tools=tools,
                    instrumentation=RemoteOpenPIInstrumentation(client),
                    output_dir=self.output / f'connection-{self.connection}',
                    metadata={
                        'mode': self.mode,
                        'evaluator_owns_resets_and_scoring': True,
                        'receipt_protocol': RECEIPT_PROTOCOL,
                        'suite': self.suite,
                        'trials_per_task': self.trials_per_task,
                        'execution_horizon': self.horizon,
                    },
                )
                try:
                    experiment.run()
                except BaseException as error:
                    if 'run_id' in state:
                        _best_effort(socket, codec, {'umi/finish_ack': 'failed',
                                                     'run_id': state['run_id'],
                                                     'error_type': type(error).__name__},
                                     self.idle_timeout)
                    raise
            # Experiment.run has returned: records and the final manifest are saved.
            manifest = (experiment.output_dir / 'experiment.json').read_bytes()
            _best_effort(socket, codec, {'umi/finish_ack': 'complete', 'run_id': state['run_id'],
                                         'manifest_sha256': hashlib.sha256(manifest).hexdigest()},
                         self.idle_timeout)
        finally:
            # A failed or stalled close must neither keep the bridge locked nor
            # replace the run's own error: it is bounded and never raises.
            try:
                socket_deadline.close(socket)
            finally:
                self.lock.release()

    def serve(self, socket, codec, model, context, state):
        """Serve one evaluation; a disconnect or timeout before the receipt fails the run.

        Each wait for the evaluator and each send to it is bounded by idle_timeout.
        """

        def receive():
            message = codec.unpackb(socket.recv(timeout=self.idle_timeout))
            _require(isinstance(message, dict), 'Expected a message mapping')
            return message

        run_id = state['run_id'] = context.manifest['run_id']
        _send(socket, codec, {'experiment_tools': self.mode, 'receipt_protocol': RECEIPT_PROTOCOL,
                              'umi_run_id': run_id, 'suite': self.suite,
                              'trials_per_task': self.trials_per_task,
                              'execution_horizon': self.horizon}, self.idle_timeout)
        message = receive()
        _require(set(message) == {'umi/start'}, 'Expected the evaluator start message')
        start = message['umi/start']
        context.publish('evaluator_start', start)
        keys = check_start(start, run_id=run_id, suite=self.suite,
                           trials_per_task=self.trials_per_task, horizon=self.horizon)
        _send(socket, codec, {'umi/start_ack': 'accepted', 'run_id': run_id}, self.idle_timeout)
        calls, position, served = [0] * len(keys), -1, 0
        while True:
            message = receive()
            if set(message) == {'umi/receipt'}:
                receipt = message['umi/receipt']
                context.publish('evaluator_receipt', receipt)
                check_receipt(receipt, run_id=run_id, start=start, keys=keys, calls=calls,
                              horizon=self.horizon)
                return {'calls': served, 'report_sha256': receipt['report_sha256'], 'run_id': run_id}
            annotation = message.pop('umi/call', None)
            _require(isinstance(annotation, dict) and set(annotation) == {'index', 'task', 'trial'}
                     and _count(annotation['index']) and annotation['index'] == served
                     and isinstance(annotation['task'], str) and _count(annotation['trial']),
                     'Missing or out-of-sequence call annotation')
            key = (annotation['task'], annotation['trial'])
            if position < 0 or key != keys[position]:
                _require(position + 1 < len(keys) and key == keys[position + 1],
                         'Call for an unexpected episode')
                position += 1
            response = model.process(
                EnvironmentStep(
                    observation=message,
                    instruction=message['prompt'],
                    step_num=served,
                    context={'umi_call': annotation},
                )
            )
            result = dict(response.metadata['policy_output'], actions=response.action)
            _send(socket, codec, result, self.idle_timeout)
            calls[position] += 1
            served += 1


def _seconds(value):
    seconds = float(value)
    if not (math.isfinite(seconds) and seconds > 0):
        raise argparse.ArgumentTypeError('Use a positive, finite number of seconds')
    return seconds


if __name__ == '__main__':
    from websockets.sync.server import serve

    p = argparse.ArgumentParser()
    p.add_argument('--upstream', default='ws://127.0.0.1:8000')
    p.add_argument('--port', type=int, default=8001)
    p.add_argument('--out', required=True)
    p.add_argument(
        '--mode', choices=MODES, required=True
    )
    p.add_argument('--suite', choices=SUITES, required=True)
    p.add_argument('--trials-per-task', type=int, required=True)
    p.add_argument('--execution-horizon', type=int, required=True)
    # Bounds each wait for and send to the evaluator. Must exceed evaluate.py's
    # --infer-timeout; an operational default, not a budget.
    p.add_argument('--idle-timeout', type=_seconds, default=900)
    a = p.parse_args()
    if not 1 <= a.trials_per_task <= 50 or a.execution_horizon < 1:
        p.error('Use 1 to 50 trials per task and a positive execution horizon')
    bridge = Bridge(a.upstream, a.out, a.mode, suite=a.suite, trials_per_task=a.trials_per_task,
                    horizon=a.execution_horizon, idle_timeout=a.idle_timeout)
    with serve(
        bridge.handler, '127.0.0.1', a.port, compression=None, max_size=64 * 1024 * 1024
    ) as server:
        server.serve_forever()
