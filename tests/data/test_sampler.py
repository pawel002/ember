import numpy as np
import pytest

import ember.data as data


class TestSequentialSampler:
    def test_order_and_len(self):
        sampler = data.SequentialSampler([0] * 5)

        assert list(sampler) == [0, 1, 2, 3, 4]
        assert len(sampler) == 5


class TestRandomSampler:
    def test_is_a_permutation(self):
        sampler = data.RandomSampler(list(range(100)), seed=0)
        order = list(sampler)

        assert sorted(order) == list(range(100))
        assert order != list(range(100))  # actually shuffled

    def test_reproducible_across_instances(self):
        a = list(data.RandomSampler(list(range(50)), seed=7))
        b = list(data.RandomSampler(list(range(50)), seed=7))

        assert a == b

    def test_fresh_permutation_each_epoch(self):
        sampler = data.RandomSampler(list(range(1000)), seed=0)

        first = list(sampler)
        second = list(sampler)

        assert sorted(first) == sorted(second) == list(range(1000))
        assert first != second

    def test_replacement(self):
        sampler = data.RandomSampler(
            list(range(10)), replacement=True, num_samples=25, seed=0
        )
        draws = list(sampler)

        assert len(draws) == 25
        assert len(sampler) == 25
        assert all(0 <= i < 10 for i in draws)

    def test_num_samples_defaults_to_len(self):
        sampler = data.RandomSampler(list(range(12)), seed=0)

        assert len(sampler) == 12

    def test_invalid_num_samples_raises(self):
        with pytest.raises(ValueError, match="num_samples"):
            data.RandomSampler(list(range(5)), num_samples=0)


class TestBatchSampler:
    def test_batches_keep_last_partial(self):
        sampler = data.SequentialSampler(list(range(10)))
        bs = data.BatchSampler(sampler, batch_size=4, drop_last=False)

        assert list(bs) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
        assert len(bs) == 3

    def test_drop_last(self):
        sampler = data.SequentialSampler(list(range(10)))
        bs = data.BatchSampler(sampler, batch_size=4, drop_last=True)

        assert list(bs) == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert len(bs) == 2

    def test_exact_division_no_empty_tail(self):
        sampler = data.SequentialSampler(list(range(8)))
        bs = data.BatchSampler(sampler, batch_size=4, drop_last=False)

        assert list(bs) == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert len(bs) == 2

    def test_len_matches_yielded_batches(self):
        for n, batch_size, drop_last in [
            (10, 4, False),
            (10, 4, True),
            (7, 1, False),
            (3, 5, False),
            (3, 5, True),
        ]:
            sampler = data.SequentialSampler(list(range(n)))
            bs = data.BatchSampler(sampler, batch_size, drop_last)
            assert len(bs) == len(list(bs))

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            data.BatchSampler(data.SequentialSampler([0] * 3), 0, False)

    def test_wraps_random_sampler(self):
        np.random.seed(0)
        bs = data.BatchSampler(
            data.RandomSampler(list(range(10)), seed=0), batch_size=5, drop_last=False
        )
        batches = list(bs)

        assert [len(b) for b in batches] == [5, 5]
        assert sorted(i for b in batches for i in b) == list(range(10))
