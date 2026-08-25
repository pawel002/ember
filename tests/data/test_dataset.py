import numpy as np
import pytest

import ember.data as data
from ember.tensor import Tensor


class ListDataset(data.Dataset):
    """Minimal map-style dataset over a Python list."""

    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


class TestTensorDataset:
    def test_len_and_getitem(self):
        np.random.seed(42)
        x = np.random.randn(10, 4).astype(np.float32)
        y = np.random.randint(0, 3, size=(10,))

        ds = data.TensorDataset(x, y)

        assert len(ds) == 10
        x_i, y_i = ds[3]
        np.testing.assert_allclose(x_i, x[3])
        np.testing.assert_array_equal(y_i, y[3])

    def test_accepts_tensors(self):
        np.random.seed(42)
        x_np = np.random.randn(8, 3).astype(np.float32)
        ds = data.TensorDataset(Tensor.from_np(x_np))

        assert len(ds) == 8
        np.testing.assert_allclose(ds[5][0], x_np[5], rtol=1e-6, atol=1e-7)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same size"):
            data.TensorDataset(np.zeros((5, 2)), np.zeros((6, 2)))

    def test_no_arrays_raise(self):
        with pytest.raises(ValueError, match="at least one"):
            data.TensorDataset()


class TestSubset:
    def test_index_mapping(self):
        ds = ListDataset([10, 20, 30, 40])
        sub = data.Subset(ds, [3, 1])

        assert len(sub) == 2
        assert sub[0] == 40
        assert sub[1] == 20

    def test_accepts_numpy_indices(self):
        ds = ListDataset([10, 20, 30])
        sub = data.Subset(ds, np.array([2, 0], dtype=np.int64))

        assert sub[0] == 30
        assert sub[1] == 10


class TestConcatDataset:
    def test_len_and_getitem(self):
        a = ListDataset([1, 2])
        b = ListDataset([3, 4, 5])
        cat = data.ConcatDataset([a, b])

        assert len(cat) == 5
        assert [cat[i] for i in range(5)] == [1, 2, 3, 4, 5]
        assert cat.cumulative_sizes == [2, 5]

    def test_add_operator(self):
        cat = ListDataset([1]) + ListDataset([2, 3])

        assert isinstance(cat, data.ConcatDataset)
        assert [cat[i] for i in range(3)] == [1, 2, 3]

    def test_negative_index(self):
        cat = ListDataset([1]) + ListDataset([2, 3])

        assert cat[-1] == 3
        assert cat[-3] == 1

    def test_out_of_range_raises(self):
        cat = ListDataset([1]) + ListDataset([2])

        with pytest.raises(IndexError):
            cat[2]
        with pytest.raises(IndexError):
            cat[-3]

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError, match="at least one"):
            data.ConcatDataset([])
        with pytest.raises(ValueError, match="empty"):
            data.ConcatDataset([ListDataset([]), ListDataset([1])])


class TestRandomSplit:
    def test_disjoint_full_coverage(self):
        ds = ListDataset(range(20))
        a, b, c = data.random_split(ds, [8, 7, 5], seed=0)

        assert (len(a), len(b), len(c)) == (8, 7, 5)
        indices = sorted(a.indices + b.indices + c.indices)
        assert indices == list(range(20))

    def test_reproducible_with_seed(self):
        ds = ListDataset(range(50))
        a1, b1 = data.random_split(ds, [30, 20], seed=123)
        a2, b2 = data.random_split(ds, [30, 20], seed=123)

        assert a1.indices == a2.indices
        assert b1.indices == b2.indices

    def test_wrong_total_raises(self):
        ds = ListDataset(range(10))
        with pytest.raises(ValueError, match="sum"):
            data.random_split(ds, [6, 6])

    def test_negative_length_raises(self):
        ds = ListDataset(range(10))
        with pytest.raises(ValueError, match="non-negative"):
            data.random_split(ds, [12, -2])
