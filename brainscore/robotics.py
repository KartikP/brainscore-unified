"""Offline DROID/RLDS integration with explicit continuous-action semantics.

This module accepts NumPy episodes (for example tfds.as_numpy output). It does
not download data, drive hardware, or infer units from an unnamed action vector.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np

from brainscore_core.events import CameraFrame, EnvironmentStep, EnvironmentResponse, Proprioception
from brainscore_core.streaming import StreamEvent


@dataclass(frozen=True)
class ActionSpec:
    names: tuple[str, ...]
    units: tuple[str, ...]
    frame: str
    period_ms: float
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self):
        n = len(self.names)
        if not n or len(set(self.names)) != n or any(not x for x in self.names):
            raise ValueError('Action names must be nonempty and unique')
        if any(len(x) != n for x in (self.units, self.lower, self.upper)):
            raise ValueError('Action names, units and bounds must have equal length')
        if not self.frame or any(not x for x in self.units):
            raise ValueError('Explicit units and coordinate frame are required')
        if not np.isfinite(self.period_ms) or self.period_ms <= 0:
            raise ValueError('period_ms must be positive and finite')
        if not np.all(np.isfinite([self.lower, self.upper])) or np.any(
                np.asarray(self.lower) > np.asarray(self.upper)):
            raise ValueError('Action bounds must be finite and ordered')

    def validate(self, action):
        action = np.asarray(action)
        if action.shape != (len(self.names),) or action.dtype.kind not in 'fiu':
            raise ValueError(f'Expected numeric action shape {(len(self.names),)}')
        if not np.all(np.isfinite(action)):
            raise ValueError('Action contains nonfinite values')
        if np.any(action < self.lower) or np.any(action > self.upper):
            raise ValueError('Action exceeds declared bounds; no implicit clipping')
        return action.copy()


def _vector(observation, key, size):
    value = np.asarray(observation[key])
    if value.shape != (size,) or value.dtype.kind not in 'fiu' or not np.all(np.isfinite(value)):
        raise ValueError(f'{key} must be a finite numeric vector of length {size}')
    return value.copy()


def droid_steps(episode, *, action_spec, action_source, timestamp_key=None):
    """Yield (observation, target) pairs without exposing targets to the policy.

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
            cameras[name] = CameraFrame(rgb=rgb.copy())
        body = Proprioception(
            joint_position=_vector(obs, 'joint_position', 7),
            cartesian_position=_vector(obs, 'cartesian_position', 6),
            gripper_position=_vector(obs, 'gripper_position', 1))
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
        step = EnvironmentStep(
            observation={'cameras': cameras, 'proprioception': body},
            instruction=instruction, step_num=index, is_first=index == 0,
            is_last=ended, is_terminal=bool(row.get('is_terminal', False)),
            context={'t_ms': t_ms, 'time_source': timestamp_key or 'declared_period',
                     'time_inferred': timestamp_key is None, 'action_spec': asdict(action_spec)})
        # Reward/discount and demonstration actions are deliberately held out.
        yield step, action_spec.validate(action_source(row))
    if previous_time is None or not ended:
        raise ValueError('Expected a nonempty episode ending with is_last')


def evaluate_droid_episode(subject, episode, *, action_spec, action_source,
                           recorder=None, timestamp_key=None):
    """Evaluate a fixed trajectory, resetting policy state at both boundaries.

    Returns predictions, targets and inference latencies. This cannot measure
    task success under policy control because observations are demonstrations.
    """
    predictions, targets, latencies = [], [], []
    subject.reset()
    try:
        for step, target in droid_steps(episode, action_spec=action_spec,
                                      action_source=action_source, timestamp_key=timestamp_key):
            if recorder:
                recorder.record('input', step)
            start = time.perf_counter()
            response = subject.process(step)
            elapsed = (time.perf_counter() - start) * 1000
            if not isinstance(response, EnvironmentResponse):
                raise TypeError('Robotics policy must return EnvironmentResponse')
            prediction = action_spec.validate(response.action)
            predictions.append(prediction)
            targets.append(target)
            latencies.append(elapsed)
            if recorder:
                recorder.record('output', StreamEvent('motor', response, step.context['t_ms'],
                    {'step_num': step.step_num, 'target': target, 'latency_ms': elapsed,
                     'deadline_missed': elapsed > action_spec.period_ms,
                     'evaluation': 'recorded_trajectory'}))
    finally:
        subject.reset()
    return {'predictions': np.stack(predictions), 'targets': np.stack(targets),
            'latency_ms': np.asarray(latencies), 'evaluation': 'recorded_trajectory'}


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
