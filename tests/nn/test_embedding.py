import numpy as np
import pytest

import ember.nn as nn
from ember import Tensor


class TestEmbedding:
    @pytest.mark.parametrize("shape", [(8,), (2, 5), (3, 4)])
    def test_forward(self, shape):
        np.random.seed(0)
        vocab, dim = 20, 6
        emb = nn.Embedding(vocab, dim)
        w = emb.weight.to_np()
        idx = np.random.randint(0, vocab, size=shape).astype(np.int32)
        y = emb(idx).to_np()
        assert y.shape == (*shape, dim)
        np.testing.assert_allclose(y, w[idx], rtol=1e-5, atol=1e-6)

    def test_backward_scatter(self):
        np.random.seed(1)
        vocab, dim = 10, 4
        emb = nn.Embedding(vocab, dim)
        idx = np.array([[1, 1, 2], [3, 1, 9]], np.int32)  # repeated ids must accumulate
        emb(idx)
        dout = np.random.randn(2, 3, dim).astype(np.float32)
        emb.backward(Tensor.from_np(dout))

        ref = np.zeros((vocab, dim), np.float32)
        np.add.at(ref, idx.ravel(), dout.reshape(-1, dim))
        np.testing.assert_allclose(emb.grad_weight.to_np(), ref, rtol=1e-4, atol=1e-4)

    def test_backward_returns_none(self):
        emb = nn.Embedding(5, 3)
        emb(np.array([0, 1, 2], np.int32))
        assert emb.backward(Tensor.from_np(np.ones((3, 3), np.float32))) is None
