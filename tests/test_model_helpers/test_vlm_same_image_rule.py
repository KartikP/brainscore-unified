"""The batch rule of the slow VLM wrapper test, on stand-in arrays."""
import numpy as np
import pytest

from tests.vlm_same_image import assert_same_image


def test_the_same_image_rule_separates_noise_from_a_mixed_up_image():
    rng = np.random.RandomState(0)
    a, b = rng.normal(0, 0.4, 2048), rng.normal(0, 0.4, 2048)
    noisy = a + rng.normal(0, 0.002, a.size)
    noisy[7] += 0.18                                   # one unit off by the measured maximum
    assert_same_image(a, noisy, b)
    with pytest.raises(AssertionError):
        assert_same_image(a, (a + b) / 2, b)          # patches of A and B mixed
    with pytest.raises(AssertionError):
        assert_same_image(a, b, a)                    # the vectors of A and B swapped
