import numpy as np
import pytest

import ember as em
from ember import Tensor


class TestRandomGeneration:
    SHAPES = [(2,), (10, 10), (5, 5, 15)]

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("interval", [(0, 1), (10, 15), (-5, 9)])
    def test_uniform_basic_properties(
        self, interval: tuple[float, float], shape: tuple[int, ...]
    ):
        t = em.random.uniform(interval[0], interval[1], shape)

        assert isinstance(t, Tensor)
        assert t.shape == shape

        arr = t.to_np()
        assert np.all(arr >= interval[0]), "Values generated below lower bound"
        assert np.all(arr <= interval[1]), "Values generated above upper bound"
        assert np.std(arr) > 0, "Uniform distribution generated constant values"

    @pytest.mark.parametrize("shape", SHAPES)
    def test_ones(self, shape: tuple[int, ...]):
        t = em.random.ones(shape)
        arr = t.to_np()

        assert np.allclose(arr, np.ones(shape))

    @pytest.mark.parametrize("shape", SHAPES)
    def test_zeros(self, shape: tuple[int, ...]):
        t = em.random.zeros(shape)
        arr = t.to_np()

        assert np.allclose(arr, np.zeros(shape))

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("value", [1.5, 10.0, 15.3])
    def test_constant(self, shape: tuple[int, ...], value: float):
        t = em.random.constant(value, shape)
        arr = t.to_np()

        expected = np.full(shape, value)
        assert np.allclose(arr, expected)

    def test_negative_dimensions_raise_error(self):
        with pytest.raises(ValueError):
            em.random.uniform(1.0, 0.0, (10, -5))
