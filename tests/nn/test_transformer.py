import math

import numpy as np
import pytest

import ember as em
import ember.loss as loss
import ember.nn as nn
import ember.optim as optim
from ember import Tensor

from ._gradcheck import numeric_grad_input


class TestFeedForward:
    def test_forward(self):
        np.random.seed(0)
        dim, hidden = 8, 16
        x = np.random.randn(2, 3, dim).astype(np.float32)
        ff = nn.FeedForward(dim, hidden, activation="gelu")
        w1, b1 = ff.fc1.w.to_np(), ff.fc1.b.to_np()
        w2, b2 = ff.fc2.w.to_np(), ff.fc2.b.to_np()
        y = ff(Tensor.from_np(x)).to_np()

        x2 = x.reshape(-1, dim)
        h = x2 @ w1 + b1
        h = 0.5 * h * (1.0 + np.tanh(0.8 * h, dtype=np.float32))  # ember GELU
        ref = (h @ w2 + b2).reshape(x.shape)
        np.testing.assert_allclose(y, ref, rtol=1e-3, atol=1e-3)

    def test_gradcheck_input(self):
        np.random.seed(1)
        dim = 6
        x = (0.4 * np.random.randn(2, 3, dim)).astype(np.float32)
        c = np.random.randn(2, 3, dim).astype(np.float32)
        ff = nn.FeedForward(dim, 12, activation="relu")
        ff(Tensor.from_np(x))
        dx = ff.backward(Tensor.from_np(c)).to_np()
        num = numeric_grad_input(lambda xt: ff(xt), x, c)
        np.testing.assert_allclose(dx, num, rtol=3e-2, atol=3e-2)


class TestPositionalEncoding:
    def test_forward_matches_sinusoid(self):
        dim, seq = 8, 5
        pe = nn.PositionalEncoding(dim, max_len=16)
        x = np.zeros((2, seq, dim), dtype=np.float32)
        y = pe(Tensor.from_np(x)).to_np()

        ref = np.zeros((seq, dim), dtype=np.float32)
        pos = np.arange(seq)[:, None]
        div = np.exp(np.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        ref[:, 0::2] = np.sin(pos * div)
        ref[:, 1::2] = np.cos(pos * div)
        np.testing.assert_allclose(y[0], ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(y[1], ref, rtol=1e-4, atol=1e-5)

    def test_backward_identity(self):
        pe = nn.PositionalEncoding(4, max_len=8)
        g = np.random.randn(2, 3, 4).astype(np.float32)
        pe(Tensor.from_np(np.zeros((2, 3, 4), np.float32)))
        np.testing.assert_array_equal(pe.backward(Tensor.from_np(g)).to_np(), g)


class TestTransformerEncoderLayer:
    @pytest.mark.parametrize("causal", [False, True])
    def test_forward_shape(self, causal):
        np.random.seed(0)
        x = np.random.randn(2, 5, 16).astype(np.float32)
        blk = nn.TransformerEncoderLayer(16, 4, ff_hidden=32, causal=causal)
        y = blk(Tensor.from_np(x))
        assert y.shape == (2, 5, 16)

    @pytest.mark.parametrize("causal", [False, True])
    def test_gradcheck_input(self, causal):
        # Validates the pre-norm residual backward wiring end-to-end.
        np.random.seed(5)
        B, S, E, H = 1, 2, 4, 2
        x = (0.3 * np.random.randn(B, S, E)).astype(np.float32)
        c = np.random.randn(B, S, E).astype(np.float32)
        blk = nn.TransformerEncoderLayer(E, H, ff_hidden=8, causal=causal)
        blk(Tensor.from_np(x))
        dx = blk.backward(Tensor.from_np(c)).to_np()
        num = numeric_grad_input(lambda xt: blk(xt), x, c)
        np.testing.assert_allclose(dx, num, rtol=4e-2, atol=4e-2)


class TestTransformerEncoder:
    def test_forward_backward_shapes(self):
        np.random.seed(0)
        x = np.random.randn(2, 6, 16).astype(np.float32)
        enc = nn.TransformerEncoder(3, 16, 4, ff_hidden=32)
        y = enc(Tensor.from_np(x))
        assert y.shape == (2, 6, 16)
        dx = enc.backward(Tensor.from_np(np.random.randn(2, 6, 16).astype(np.float32)))
        assert dx.shape == (2, 6, 16)
        # every parameter has a matching gradient after backward
        grads = enc.gradients()
        assert len(grads) == len(enc.parameters())
        assert all(g is not None for g in grads)

    def test_gradcheck_input(self):
        np.random.seed(6)
        x = (0.3 * np.random.randn(1, 2, 8)).astype(np.float32)
        c = np.random.randn(1, 2, 8).astype(np.float32)
        enc = nn.TransformerEncoder(2, 8, 2, ff_hidden=16, final_norm=True)
        enc(Tensor.from_np(x))
        dx = enc.backward(Tensor.from_np(c)).to_np()
        num = numeric_grad_input(lambda xt: enc(xt), x, c)
        np.testing.assert_allclose(dx, num, rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize("causal", [False, True])
    def test_overfit(self, causal):
        # End-to-end: forward + composed hand-written backward + Adam must drive
        # a fixed batch to near-zero loss. Exercises the whole stack together.
        em.random.seed(0)
        np.random.seed(0)
        B, S, E, H = 4, 6, 16, 4
        enc = nn.TransformerEncoder(
            2, E, H, ff_hidden=32, causal=causal, final_norm=False
        )
        x = Tensor.from_np(np.random.randn(B, S, E).astype(np.float32))
        y = Tensor.from_np(np.random.randn(B, S, E).astype(np.float32))
        opt = optim.Adam(enc.parameters(), lr=1e-2)
        crit = loss.MSELoss()

        first = None
        for _ in range(150):
            pred = enc(x, training=True)
            cur = crit(pred, y)
            if first is None:
                first = cur
            enc.backward(crit.backward())
            opt.apply(enc.gradients())

        assert first > 0.5
        assert cur < 0.05 * first
