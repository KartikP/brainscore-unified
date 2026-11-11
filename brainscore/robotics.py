"""Offline DROID/RLDS integration with explicit continuous-action semantics.

This module accepts NumPy episodes (for example tfds.as_numpy output). It does
not download data, drive hardware, or infer units from an unnamed action vector.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Mapping
import time

import numpy as np

from brainscore_core.streaming import NamedStreams, Session, StreamEvent, begin_event, row_events


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


def evaluate_droid_episode(subject, episode, *, action_spec, action_source,
                           recorder=None, timestamp_key=None):
    """Evaluate a fixed trajectory; a ``begin`` clears the subject's episode state first.

    The subject keeps its interventions and task setup (no ``reset()``).
    Returns predictions, targets and inference latencies. This cannot measure
    task success under policy control because observations are demonstrations.
    """
    predictions, targets, latencies = [], [], []
    subject.participate(Session([begin_event()], ('motor',)))
    for step, target, info in droid_steps(episode, action_spec=action_spec,
                                          action_source=action_source,
                                          timestamp_key=timestamp_key):
        if recorder:
            for event in row_events(step, info['t_ms'], step_num=info['step_num']):
                recorder.record('input', event)
        start = time.perf_counter()
        outputs = subject.process(step)
        elapsed = (time.perf_counter() - start) * 1000
        if not isinstance(outputs, Mapping) or 'motor' not in outputs:
            raise TypeError("Robotics policy must return {'motor': action}")
        response = outputs['motor']
        action = response.payload if isinstance(response, StreamEvent) else response
        prediction = action_spec.validate(action)
        predictions.append(prediction)
        targets.append(target)
        latencies.append(elapsed)
        if recorder:
            recorder.record('output', StreamEvent('motor', action, info['t_ms'],
                {'step_num': info['step_num'], 'target': target, 'latency_ms': elapsed,
                 'deadline_missed': elapsed > action_spec.period_ms,
                 'evaluation': 'recorded_trajectory', 'action_status': 'predicted',
                 'deadline_enforced': False}))
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
