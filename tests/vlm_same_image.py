"""The rule that one image's vector does not depend on the other images of its batch."""
import numpy as np


def assert_same_image(alone, batched, other):
    """The vector of one image is the same alone and in a batch, up to number-type noise.

    A cloud measurement (2026-11-16, ``umi-work-record/cloud_verify/scripts/vlm_batch_probe.py``,
    Qwen2.5-VL-3B ``blocks.28``, the two images of the test): in float16 on the GPU the vector
    of image A alone and in a batch with B differs by 0.0016 on average and by up to 0.18 on
    one unit (a unit of size 299); in float32 by at most 9e-5. The vector of B
    differs from A's by 0.32 on average. So the rule is on the mean, not on each unit: the mean
    absolute difference is below 0.01, and at least 20 times smaller than the difference to
    the other image's vector (a mixed-up segmentation gives a difference near 0.32).
    """
    alone, batched, other = (np.asarray(v, dtype=np.float64) for v in (alone, batched, other))
    same = np.abs(alone - batched).mean()
    different = np.abs(batched - other).mean()
    assert same < 0.01, f'image A moves by {same} on average when batched with B'
    assert 20 * same <= different, f'A alone vs batched: {same}; A vs B: {different}'
