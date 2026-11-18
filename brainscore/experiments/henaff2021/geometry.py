"""Discrete curvature in a declared representation, in degrees."""

import numpy as np


def curvature(trajectory: np.ndarray, stride: int = 1) -> float:
    """Average all local turning angles, pooling the interleaved phases.

    With 11 frames and stride 2, the odd phase contributes four angles and
    the even phase three. Average those seven angles, not the two phase means.
    A repeated point makes an angle undefined; it is never counted as zero.
    """
    if stride not in (1, 2):
        raise ValueError('stride must be 1 or 2')
    x = np.asarray(trajectory, dtype=np.float64)
    if x.ndim < 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('trajectory needs at least three finite feature vectors')
    x = x.reshape(len(x), -1)
    angles = []
    for phase in range(stride):
        points = x[phase::stride]
        if len(points) < 3:
            raise ValueError('each stride phase needs at least three frames')
        delta = np.diff(points, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        if np.any(lengths == 0) or not np.isfinite(lengths).all():
            raise ValueError('undefined curvature: zero or nonfinite displacement')
        directions = delta / lengths[:, None]
        cosine = np.sum(directions[:-1] * directions[1:], axis=1)
        angles.extend(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return float(np.mean(angles))
