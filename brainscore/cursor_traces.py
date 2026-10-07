"""Read nodekit traces and compute cursor-movement metrics.

A nodekit ``Trace`` is JSON: ``{"nodekit_version", "graph"?, "events": [...]}``.
Humans (nodekit's browser runtime) and models (``harnesses.nodekit_browser``)
produce the same format, so one reader serves both. nodekit itself is not
imported (it needs Python 3.12); only the event list is read.

Coordinates are nodekit Board coordinates: origin at the board centre, y up,
1 unit = 1 CSS px on the 1024 x 1024 normative board.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

POINTER_KINDS = ('move', 'down', 'up')


@dataclass
class TrialTrace:
    """Pointer samples and outcome of one node (one trial step).

    ``samples`` is ``(n, 4)`` float: ``t_ms, x, y, kind`` where ``kind`` indexes
    :data:`POINTER_KINDS`. Times are page-clock ms since the trace started.
    """
    node_address: Tuple[str, ...]
    t_start: float
    t_end: Optional[float]
    action: Optional[Dict[str, Any]]
    t_action: Optional[float]
    samples: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))

    @property
    def node_id(self) -> str:
        return self.node_address[-1]

    @property
    def completed(self) -> bool:
        return self.t_end is not None


def load_trace(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Load a trace from a dict, a JSON string, or a JSON file path."""
    if isinstance(source, dict):
        return source
    if isinstance(source, Path) or (isinstance(source, str) and not source.lstrip().startswith('{')):
        return json.loads(Path(source).read_text())
    return json.loads(source)


def read_trials(source: Union[str, Path, Dict[str, Any]]) -> List[TrialTrace]:
    """Split a trace into per-node :class:`TrialTrace` records, in order.

    Each pointer sample is assigned to the node that was running when it was
    taken; samples outside any node (e.g. between nodes) are dropped.
    """
    events = sorted(load_trace(source)['events'], key=lambda e: e['t'])
    trials: List[TrialTrace] = []
    current: Optional[TrialTrace] = None
    rows: List[List[float]] = []
    for event in events:
        kind = event['event_type']
        if kind == 'NodeStartedEvent':
            current = TrialTrace(tuple(event['node_address']), float(event['t']),
                                 None, None, None)
            rows = []
            trials.append(current)
        elif current is None:
            continue
        elif kind == 'PointerSampledEvent':
            rows.append([event['t'], event['x'], event['y'], POINTER_KINDS.index(event['kind'])])
        elif kind == 'ActionTakenEvent':
            current.action, current.t_action = event.get('action'), float(event['t'])
        elif kind == 'NodeEndedEvent':
            current.t_end = float(event['t'])
            current.samples = np.asarray(rows, dtype=float).reshape(-1, 4)
            current = None
    if current is not None:  # trace ended mid-node (e.g. step budget ran out)
        current.samples = np.asarray(rows, dtype=float).reshape(-1, 4)
    return trials


def trajectory_metrics(trial: TrialTrace, start: Sequence[float],
                       target: Optional[Sequence[float]] = None,
                       target_size: Optional[Sequence[float]] = None,
                       move_threshold: float = 2.0) -> Dict[str, float]:
    """Standard mouse-tracking measures for one trial.

    :param start: (x, y) where the movement starts (e.g. the home button).
    :param target: (x, y) centre of the target; used for deviation, endpoint
        error and hit. Defaults to the final pointer position.
    :param target_size: (w, h) of the target rectangle, for ``hit``.
    :param move_threshold: board px the pointer must leave ``start`` by to count
        as movement onset.

    Returns ``initiation_time`` (node start to movement onset), ``response_time``
    (node start to action), ``movement_time`` (onset to action), ``path_length``,
    ``mad`` (signed max deviation from the start-target line, positive = left of
    the direction of travel), ``auc`` (signed area between path and that line),
    ``x_flips``, ``endpoint_error`` and ``hit`` (1.0/0.0). Undefined values are NaN.
    """
    nan = float('nan')
    out = dict(initiation_time=nan, response_time=nan, movement_time=nan,
               path_length=nan, mad=nan, auc=nan, x_flips=nan,
               endpoint_error=nan, hit=nan)
    if trial.t_action is not None:
        out['response_time'] = trial.t_action - trial.t_start
    xy = trial.samples[:, 1:3]
    if len(xy) == 0:
        return out
    start = np.asarray(start, dtype=float)
    path = np.vstack([start, xy])
    end = np.asarray(target, dtype=float) if target is not None else path[-1]
    moved = np.flatnonzero(np.linalg.norm(xy - start, axis=1) > move_threshold)
    if len(moved):
        onset = trial.samples[moved[0], 0]
        out['initiation_time'] = onset - trial.t_start
        if trial.t_action is not None:
            out['movement_time'] = trial.t_action - onset
    out['path_length'] = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length > 0:
        unit = direction / length
        rel = path - start
        along = rel @ unit
        across = rel[:, 0] * unit[1] * -1 + rel[:, 1] * unit[0]  # 2-D cross product
        out['mad'] = float(across[np.argmax(np.abs(across))])
        order = np.argsort(along, kind='stable')
        out['auc'] = float(np.trapz(across[order], along[order]))
    dx = np.sign(np.diff(path[:, 0]))
    dx = dx[dx != 0]
    out['x_flips'] = float(np.count_nonzero(np.diff(dx)))
    final = xy[-1]
    if target is not None:
        out['endpoint_error'] = float(np.linalg.norm(final - end))
        if target_size is not None:
            half = np.asarray(target_size, dtype=float) / 2
            out['hit'] = float(np.all(np.abs(final - end) <= half))
    return out


def resample(trial: TrialTrace, n: int = 101) -> np.ndarray:
    """Time-normalize the (x, y) path to ``n`` points (mouse-tracking convention)."""
    s = trial.samples
    if len(s) == 0:
        return np.full((n, 2), np.nan)
    if len(s) == 1:
        return np.repeat(s[:, 1:3], n, axis=0)
    t = (s[:, 0] - s[0, 0]) / max(s[-1, 0] - s[0, 0], 1e-9)
    grid = np.linspace(0, 1, n)
    return np.column_stack([np.interp(grid, t, s[:, 1]), np.interp(grid, t, s[:, 2])])
