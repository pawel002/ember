import numpy as np
import pytest

import ember as em
from ember import Tensor


def test_uniform_generation():
    low = 0.0
    high = 1.0
    shape = (10, 10)
    t = em.random.uniform(low, high, shape)

    assert isinstance(t, Tensor)
    assert t.shape == shape

    arr = t.to_np()

    assert np.all(arr >= low)
    assert np.all(arr <= high)
    assert np.std(arr) > 0


def test_large_tensor():
    t = em.random.uniform(5.0, -5.0, (100, 100))
    arr = t.to_np()

    assert arr.size == 10000
    assert np.all(arr >= -5.0)
    assert np.all(arr <= 5.0)


def test_negative_dimensions():
    with pytest.raises(ValueError):
        em.random.uniform(1.0, 0.0, (10, -5))
