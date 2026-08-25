import numpy as np
import pytest

import ember.data as data
import ember.loss as loss
import ember.nn as nn
import ember.optim as optim
from ember.tensor import Tensor


def make_xy(n=64, in_dim=4, out_dim=1, seed=42):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    x = rng.standard_normal((n, in_dim)).astype(np.float32)
    y = (x @ w + 0.5).astype(np.float32)
    return x, y


class TestDataLoaderBatching:
    def test_sequential_full_pass_in_order(self):
        x, y = make_xy()
        loader = data.DataLoader(data.TensorDataset(x, y), batch_size=16)

        batches = list(loader)

        assert len(batches) == 4
        xb = np.concatenate([b[0].to_np() for b in batches])
        yb = np.concatenate([b[1].to_np() for b in batches])
        np.testing.assert_allclose(xb, x, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(yb, y, rtol=1e-6, atol=1e-7)

    def test_partial_last_batch(self):
        x, y = make_xy(n=10)
        loader = data.DataLoader(data.TensorDataset(x, y), batch_size=4)

        shapes = [b[0].shape for b in loader]

        assert shapes == [(4, 4), (4, 4), (2, 4)]

    def test_drop_last(self):
        x, y = make_xy(n=10)
        loader = data.DataLoader(data.TensorDataset(x, y), batch_size=4, drop_last=True)

        batches = list(loader)

        assert len(loader) == 2
        assert all(b[0].shape == (4, 4) for b in batches)

    def test_len_matches_batches(self):
        x, y = make_xy(n=10)
        for batch_size, drop_last in [(1, False), (3, False), (3, True), (4, True)]:
            loader = data.DataLoader(
                data.TensorDataset(x, y), batch_size=batch_size, drop_last=drop_last
            )
            assert len(loader) == len(list(loader))

    def test_default_batch_size_is_one(self):
        x, y = make_xy(n=3)
        loader = data.DataLoader(data.TensorDataset(x, y))

        batches = list(loader)

        assert len(batches) == 3
        assert batches[0][0].shape == (1, 4)

    def test_integer_targets_stay_numpy(self):
        rng = np.random.default_rng(0)
        tokens = rng.integers(0, 100, size=(16, 5))
        x = rng.standard_normal((16, 5)).astype(np.float32)
        loader = data.DataLoader(data.TensorDataset(tokens, x), batch_size=8)

        tok_b, x_b = next(iter(loader))

        assert isinstance(tok_b, np.ndarray)
        assert tok_b.dtype.kind in "iu"
        assert isinstance(x_b, Tensor)


class TestDataLoaderShuffling:
    def test_shuffle_preserves_content_not_order(self):
        x, y = make_xy(n=64)
        loader = data.DataLoader(
            data.TensorDataset(x, y), batch_size=64, shuffle=True, seed=0
        )

        xb, _ = next(iter(loader))

        # same rows, (almost surely) different order
        assert not np.allclose(xb.to_np(), x)
        assert np.allclose(np.sort(xb.to_np(), axis=0), np.sort(x, axis=0))

    def test_seed_reproducibility(self):
        x, y = make_xy(n=32)

        def epoch_rows():
            loader = data.DataLoader(
                data.TensorDataset(x, y), batch_size=8, shuffle=True, seed=123
            )
            return np.concatenate([b[0].to_np() for b in loader])

        np.testing.assert_allclose(epoch_rows(), epoch_rows())

    def test_reshuffles_each_epoch(self):
        x, y = make_xy(n=256)
        loader = data.DataLoader(
            data.TensorDataset(x, y), batch_size=256, shuffle=True, seed=0
        )

        first, second = (next(iter(loader))[0].to_np() for _ in range(2))

        assert not np.allclose(first, second)

    def test_rows_stay_paired_with_targets(self):
        x, y = make_xy(n=50)
        loader = data.DataLoader(
            data.TensorDataset(x, y), batch_size=10, shuffle=True, seed=1
        )

        for xb, yb in loader:
            # reconstruct row indices and check x/y alignment
            for row_x, row_y in zip(xb.to_np(), yb.to_np(), strict=True):
                matches = np.isclose(x, row_x).all(axis=1)
                assert matches.sum() == 1
                assert np.isclose(y[matches.argmax()], row_y).all()


class TestDataLoaderSamplers:
    def test_custom_sampler(self):
        x, y = make_xy(n=6)

        class ReversedSampler(data.Sampler):
            def __iter__(self):
                return iter([5, 4, 3, 2, 1, 0])

            def __len__(self):
                return 6

        loader = data.DataLoader(
            data.TensorDataset(x, y), batch_size=2, sampler=ReversedSampler()
        )
        xb = np.concatenate([b[0].to_np() for b in loader])

        np.testing.assert_allclose(xb, x[::-1], rtol=1e-6, atol=1e-7)

    def test_custom_batch_sampler(self):
        x, y = make_xy(n=4)
        loader = data.DataLoader(
            data.TensorDataset(x, y),
            batch_sampler=data.BatchSampler(
                data.SequentialSampler(range(4)), batch_size=4, drop_last=False
            ),
        )

        batches = list(loader)

        assert len(batches) == 1
        assert batches[0][0].shape == (4, 4)


class TestDataLoaderValidation:
    def test_sampler_and_shuffle_raise(self):
        x, y = make_xy(n=4)
        with pytest.raises(ValueError, match="mutually exclusive"):
            data.DataLoader(
                data.TensorDataset(x, y),
                shuffle=True,
                sampler=data.SequentialSampler(range(4)),
            )

    def test_batch_sampler_exclusions(self):
        x, y = make_xy(n=4)
        bs = data.BatchSampler(data.SequentialSampler(range(4)), 2, False)

        for kwargs in [
            {"batch_size": 2},
            {"shuffle": True},
            {"sampler": data.SequentialSampler(range(4))},
            {"drop_last": True},
        ]:
            with pytest.raises(ValueError, match="mutually exclusive"):
                data.DataLoader(data.TensorDataset(x, y), batch_sampler=bs, **kwargs)

    def test_seed_requires_shuffle(self):
        x, y = make_xy(n=4)
        with pytest.raises(ValueError, match="seed"):
            data.DataLoader(data.TensorDataset(x, y), seed=1)
        with pytest.raises(ValueError, match="seed"):
            data.DataLoader(
                data.TensorDataset(x, y),
                sampler=data.SequentialSampler(range(4)),
                seed=1,
            )


class TestDataLoaderCollate:
    def test_custom_collate_fn(self):
        x, y = make_xy(n=4)
        loader = data.DataLoader(
            data.TensorDataset(x, y),
            batch_size=4,
            collate_fn=lambda samples: np.stack([s[0] for s in samples]),
        )

        batch = next(iter(loader))

        assert isinstance(batch, np.ndarray)
        np.testing.assert_allclose(batch, x, rtol=1e-6, atol=1e-7)


class TestEndToEndTraining:
    def test_mlp_learns_linear_regression(self):
        x, y = make_xy(n=128, in_dim=4, out_dim=1, seed=42)

        loader = data.DataLoader(
            data.TensorDataset(x, y), batch_size=16, shuffle=True, seed=0
        )

        np.random.seed(0)
        model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
        opt = optim.SGD(model.parameters(), lr=0.05)
        criterion = loss.MSELoss()

        first_loss = None
        last_loss = None
        for _ in range(60):
            for xb, yb in loader:
                grad = criterion.gradient(model(xb, training=True), yb)
                model.backward(grad)
                opt.apply(model.gradients())
            # measure true epoch loss over the full dataset
            epoch_loss = criterion(
                model(Tensor.from_np(x), training=False), Tensor.from_np(y)
            )
            if first_loss is None:
                first_loss = epoch_loss
            last_loss = epoch_loss

        assert last_loss < first_loss
        assert last_loss < 0.1
