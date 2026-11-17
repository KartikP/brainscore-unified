"""Offline DROID/RLDS integration with explicit continuous-action semantics.

This module accepts NumPy episodes (for example tfds.as_numpy output). It does
not download data, drive hardware, or infer units from an unnamed action vector.
"""

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from brainscore_core.streaming import NamedStreams, StreamEvent, row_events
from brainscore_core.streaming_helpers import environment_session


from brainscore_core.environment import ActionSpec


def validate_cameras(cameras) -> None:
    """Reject malformed camera frames before they reach a policy.

    ``cameras`` maps a camera name to its frame. Each frame must be (H, W, 3)
    channels-last RGB. Raises ``ValueError`` listing every bad camera.
    Robot-specific, so it lives here, not in core.
    """
    from brainscore_core.io_catalog import check_payload
    if cameras is None:
        return
    if not isinstance(cameras, dict):
        raise ValueError('Environment cameras must be a name-to-frame mapping.')
    issues = []
    for name, rgb in cameras.items():
        issues += [f"camera '{name}': {m}" for m in check_payload('vision:image', rgb)]
        shape = getattr(rgb, 'shape', None)
        if shape is not None and not (len(shape) == 3 and shape[-1] == 3):
            issues.append(f"camera '{name}': rgb shape {tuple(shape)} is not "
                          "(H, W, 3) channels-last RGB.")
    if issues:
        raise ValueError("Environment camera payload(s) failed validation:\n  - "
                         + "\n  - ".join(issues))


def _vector(observation, key, size):
    value = np.asarray(observation[key])
    if value.shape != (size,) or value.dtype.kind not in 'fiu' or not np.all(np.isfinite(value)):
        raise ValueError(f'{key} must be a finite numeric vector of length {size}')
    return value.copy()


def droid_steps(episode, *, action_spec, action_source, timestamp_key=None):
    """Yield (row, target, info) triples without exposing targets to the policy.

    ``row`` maps ``vision:image`` (named cameras), ``proprioception`` and
    ``instruction`` to their payloads; ``info`` holds ``t_ms``, ``time_source``,
    ``time_inferred``, ``step_num``, ``is_last``, ``is_terminal`` and the action spec.

    action_source is an explicit callable selecting the demonstration action,
    e.g. concatenate action_dict fields in the order specified by ActionSpec.
    A timestamp_key names an observation field already expressed in milliseconds.
    Without one, time is derived from the caller's declared sampling period and
    marked as inferred. Recorded policy evaluation is open-loop action agreement.
    """
    steps = episode['steps']
    previous_time = None
    ended = False
    for index, row in enumerate(steps):
        if ended:
            raise ValueError('Episode contains steps after is_last')
        if bool(row.get('is_first', index == 0)) != (index == 0):
            raise ValueError('is_first must identify exactly the first step')
        obs = row['observation']
        cameras = {}
        for source, name in (
                ('wrist_image_left', 'wrist'), ('exterior_image_1_left', 'exterior_1'),
                ('exterior_image_2_left', 'exterior_2')):
            rgb = np.asarray(obs[source])
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f'{source} must be an H x W x 3 uint8 image')
            cameras[name] = rgb.copy()
        body = {'joint_position': _vector(obs, 'joint_position', 7),
                'cartesian_position': _vector(obs, 'cartesian_position', 6),
                'gripper_position': _vector(obs, 'gripper_position', 1)}
        t_ms = float(obs[timestamp_key]) if timestamp_key else index * action_spec.period_ms
        if not np.isfinite(t_ms) or (previous_time is not None and t_ms <= previous_time):
            raise ValueError('Episode timestamps must be finite and strictly increasing')
        previous_time = t_ms
        instruction = row.get('language_instruction', '')
        if isinstance(instruction, bytes):
            instruction = instruction.decode('utf-8')
        if not isinstance(instruction, str):
            raise ValueError('language_instruction must be text or UTF-8 bytes')
        ended = bool(row.get('is_last', False))
        validate_cameras(cameras)
        step = {'vision:image': NamedStreams(cameras), 'proprioception': body,
                'instruction': instruction}
        info = {'t_ms': t_ms, 'time_source': timestamp_key or 'declared_period',
                'time_inferred': timestamp_key is None, 'step_num': index, 'is_last': ended,
                'is_terminal': bool(row.get('is_terminal', False)),
                'action_spec': asdict(action_spec)}
        # Reward/discount and demonstration actions are deliberately held out.
        yield step, action_spec.validate(action_source(row)), info
    if previous_time is None or not ended:
        raise ValueError('Expected a nonempty episode ending with is_last')


class _RecordedEpisode:
    """A recorded episode as an environment: each step serves the next recorded row.

    The subject's action never changes the next row (open loop). The session owns
    the loop; this environment keeps the predictions, targets and latencies, and
    writes the recorder's input and output events.
    """

    def __init__(self, steps, action_spec, recorder):
        self._steps = steps
        self._spec = action_spec
        self._recorder = recorder
        self._target = self._info = self._served = None
        self.predictions, self.targets, self.latencies = [], [], []

    def _next_row(self):
        row, self._target, self._info = next(self._steps)
        if self._recorder:
            for event in row_events(row, self._info['t_ms'], step_num=self._info['step_num']):
                self._recorder.record('input', event)
        self._served = time.perf_counter()
        return row

    def clock(self):
        return {key: self._info[key] for key in ('t_ms', 'time_source', 'time_inferred')}

    def reset(self):
        return self._next_row()

    def step(self, outputs):
        # From the row leaving the environment to the action reaching it (the session included).
        elapsed = (time.perf_counter() - self._served) * 1000
        action = outputs['motor']
        self.predictions.append(self._spec.validate(action))
        self.targets.append(self._target)
        self.latencies.append(elapsed)
        if self._recorder:
            info = self._info
            self._recorder.record('output', StreamEvent('motor', action, info['t_ms'],
                {'step_num': info['step_num'], 'target': self._target, 'latency_ms': elapsed,
                 'deadline_missed': elapsed > self._spec.period_ms,
                 'evaluation': 'recorded_trajectory', 'action_status': 'predicted',
                 'deadline_enforced': False}))
        try:
            return self._next_row(), None
        except StopIteration:
            return {}, 'terminal'


def evaluate_droid_episode(subject, episode, *, action_spec, action_source,
                           recorder=None, timestamp_key=None):
    """Evaluate a fixed trajectory in one closed-loop session; its ``begin`` clears
    the subject's episode state first.

    The subject keeps its interventions and task setup (no ``reset()``). It must
    declare the inputs its policy reads (``policy_in_channels``, for example
    ``DroidPolicy.inputs``): the session sends only those. Returns predictions,
    targets and latencies (from the row leaving the episode to the action
    reaching it). This cannot measure task success under policy control because
    observations are demonstrations.
    """
    environment = _RecordedEpisode(
        droid_steps(episode, action_spec=action_spec, action_source=action_source,
                    timestamp_key=timestamp_key),
        action_spec, recorder)
    # history_limit=0: the session keeps no copy of the camera frames; the episode keeps the results.
    with environment_session(environment, history_limit=0) as session:
        subject.participate(session)
    if not session.complete:
        raise RuntimeError('The recorded episode ended before its last step')
    return {'predictions': np.stack(environment.predictions),
            'targets': np.stack(environment.targets),
            'latency_ms': np.asarray(environment.latencies), 'evaluation': 'recorded_trajectory'}


def staged_droid_episodes(directory, *, split='train'):
    """Read a prepared local TFDS directory. Never calls download_and_prepare."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    try:
        import tensorflow_datasets as tfds
    except ImportError as error:
        raise ImportError('Install brainscore[robotics-data] to read staged TFDS data') from error
    builder = tfds.builder_from_directory(str(directory))
    yield from tfds.as_numpy(builder.as_dataset(split=split))
