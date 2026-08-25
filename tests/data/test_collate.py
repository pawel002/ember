import numpy as np
import pytest

import ember.data as data
from ember.tensor import Tensor


class TestDefaultCollate:
    def test_float_arrays_become_tensor(self):
        np.random.seed(42)
        samples = [np.random.randn(4).astype(np.float32) for _ in range(3)]

        batch = data.default_collate(samples)

        assert isinstance(batch, Tensor)
        assert batch.shape == (3, 4)
        np.testing.assert_allclose(
            batch.to_np(), np.stack(samples), rtol=1e-6, atol=1e-7
        )

    def test_integer_arrays_stay_numpy(self):
        samples = [np.array([1, 2], dtype=np.int64), np.array([3, 4], dtype=np.int64)]

        batch = data.default_collate(samples)

        assert isinstance(batch, np.ndarray)
        assert batch.dtype.kind in "iu"
        np.testing.assert_array_equal(batch, [[1, 2], [3, 4]])

    def test_tensor_samples_become_tensor(self):
        np.random.seed(42)
        arrs = np.random.randn(5, 2).astype(np.float32)
        samples = [Tensor.from_np(a) for a in arrs]

        batch = data.default_collate(samples)

        assert isinstance(batch, Tensor)
        assert batch.shape == (5, 2)
        np.testing.assert_allclose(batch.to_np(), arrs, rtol=1e-6, atol=1e-7)

    def test_scalar_floats_become_tensor(self):
        batch = data.default_collate([1.5, 2.5, 3.5])

        assert isinstance(batch, Tensor)
        assert batch.shape == (3,)
        np.testing.assert_allclose(batch.to_np(), [1.5, 2.5, 3.5])

    def test_scalar_ints_stay_numpy(self):
        batch = data.default_collate([1, 2, 3])

        assert isinstance(batch, np.ndarray)
        assert batch.dtype.kind in "iu"
        np.testing.assert_array_equal(batch, [1, 2, 3])

    def test_scalar_bools_stay_numpy(self):
        batch = data.default_collate([True, False, True])

        assert isinstance(batch, np.ndarray)
        assert batch.dtype.kind == "b"

    def test_strings_pass_through(self):
        batch = data.default_collate(["a", "b"])

        assert batch == ["a", "b"]

    def test_tuple_samples_collated_fieldwise(self):
        np.random.seed(42)
        xs = np.random.randn(4, 3).astype(np.float32)
        ys = np.random.randint(0, 5, size=(4,))
        samples = [(xs[i], ys[i]) for i in range(4)]

        xb, yb = data.default_collate(samples)

        assert isinstance(xb, Tensor)
        assert isinstance(yb, np.ndarray)
        np.testing.assert_allclose(xb.to_np(), xs, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(yb, ys)

    def test_list_samples_collated_fieldwise(self):
        samples = [[float(i), i] for i in range(3)]

        floats, ints = data.default_collate(samples)

        assert isinstance(floats, Tensor)
        assert isinstance(ints, np.ndarray)

    def test_dict_samples_collated_keywise(self):
        samples = [
            {"x": np.array([1.0], dtype=np.float32), "name": "a"},
            {"x": np.array([2.0], dtype=np.float32), "name": "b"},
        ]

        batch = data.default_collate(samples)

        assert isinstance(batch["x"], Tensor)
        np.testing.assert_allclose(batch["x"].to_np(), [[1.0], [2.0]])
        assert batch["name"] == ["a", "b"]

    def test_nested_structures(self):
        samples = [
            (np.array([1.0], dtype=np.float32), {"y": 1}),
            (np.array([2.0], dtype=np.float32), {"y": 2}),
        ]

        xb, yb = data.default_collate(samples)

        assert isinstance(xb, Tensor)
        np.testing.assert_array_equal(yb["y"], [1, 2])

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="default_collate"):
            data.default_collate([object(), object()])
