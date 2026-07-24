import numpy as np
import pytest

import ember.nn as nn
from ember import Tensor

from ._gradcheck import numeric_grad_input, numeric_grad_param


def _softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def _ref_mha(x, params, H, causal):
    """NumPy reference matching MultiHeadAttention (bias-ed projections)."""
    Wq, bq, Wk, bk, Wv, bv, Wo, bo = params
    B, S, E = x.shape
    dh = E // H
    scale = 1.0 / np.sqrt(dh)
    x2 = x.reshape(B * S, E)

    def heads(t):
        return t.reshape(B, S, H, dh).transpose(0, 2, 1, 3)  # (B,H,S,dh)

    qh = heads(x2 @ Wq + bq)
    kh = heads(x2 @ Wk + bk)
    vh = heads(x2 @ Wv + bv)

    scores = scale * (qh @ kh.transpose(0, 1, 3, 2))  # (B,H,S,S)
    if causal:
        mask = np.triu(np.ones((S, S), dtype=bool), k=1)
        scores = np.where(mask[None, None], -np.inf, scores)
    p = _softmax(scores, -1)
    o = p @ vh  # (B,H,S,dh)
    o = o.transpose(0, 2, 1, 3).reshape(B * S, E)
    return (o @ Wo + bo).reshape(B, S, E)


def _params_of(mha):
    return [
        mha.wq.w.to_np(),
        mha.wq.b.to_np(),
        mha.wk.w.to_np(),
        mha.wk.b.to_np(),
        mha.wv.w.to_np(),
        mha.wv.b.to_np(),
        mha.wo.w.to_np(),
        mha.wo.b.to_np(),
    ]


class TestMultiHeadAttention:
    CONFIGS = [
        # (batch, seq, embed, heads)
        (2, 4, 8, 2),
        (1, 6, 12, 3),
        (3, 5, 16, 4),
    ]

    @pytest.mark.parametrize("B,S,E,H", CONFIGS)
    @pytest.mark.parametrize("causal", [False, True])
    def test_forward(self, B, S, E, H, causal):
        np.random.seed(0)
        x = np.random.randn(B, S, E).astype(np.float32)
        mha = nn.MultiHeadAttention(E, H, causal=causal)
        y = mha(Tensor.from_np(x)).to_np()
        ref = _ref_mha(x, _params_of(mha), H, causal)
        np.testing.assert_allclose(y, ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("causal", [False, True])
    def test_gradcheck_input(self, causal):
        np.random.seed(3)
        B, S, E, H = 2, 3, 8, 2
        x = (0.4 * np.random.randn(B, S, E)).astype(np.float32)
        c = np.random.randn(B, S, E).astype(np.float32)
        mha = nn.MultiHeadAttention(E, H, causal=causal)

        mha(Tensor.from_np(x))
        dx = mha.backward(Tensor.from_np(c)).to_np()
        num = numeric_grad_input(lambda xt: mha(xt), x, c)
        np.testing.assert_allclose(dx, num, rtol=3e-2, atol=3e-2)

    @pytest.mark.parametrize("causal", [False, True])
    def test_gradcheck_params(self, causal):
        np.random.seed(4)
        B, S, E, H = 1, 3, 8, 2
        x = (0.4 * np.random.randn(B, S, E)).astype(np.float32)
        c = np.random.randn(B, S, E).astype(np.float32)
        mha = nn.MultiHeadAttention(E, H, causal=causal)

        mha(Tensor.from_np(x))
        mha.backward(Tensor.from_np(c))

        # analytic vs numeric for each projection's weight and bias
        for proj in (mha.wq, mha.wk, mha.wv, mha.wo):
            num_w = numeric_grad_param(lambda: mha(Tensor.from_np(x)), proj.w, c)
            num_b = numeric_grad_param(lambda: mha(Tensor.from_np(x)), proj.b, c)
            # re-run forward/backward to refresh analytic grads (params restored)
            mha(Tensor.from_np(x))
            mha.backward(Tensor.from_np(c))
            np.testing.assert_allclose(proj.grad_w.to_np(), num_w, rtol=4e-2, atol=4e-2)
            np.testing.assert_allclose(proj.grad_b.to_np(), num_b, rtol=4e-2, atol=4e-2)
