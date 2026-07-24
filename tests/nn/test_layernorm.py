import numpy as np
import pytest

import ember.nn as nn
from ember import Tensor

from ._gradcheck import numeric_grad_input, numeric_grad_param


def _ref_layernorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    return xhat * gamma + beta


class TestLayerNorm:
    SHAPES = [(4, 8), (2, 3, 16), (1, 32)]

    @pytest.mark.parametrize("shape", SHAPES)
    def test_forward(self, shape):
        np.random.seed(0)
        dim = shape[-1]
        x = np.random.randn(*shape).astype(np.float32)
        ln = nn.LayerNorm(dim)
        g = ln.gamma.to_np()
        b = ln.beta.to_np()
        y = ln(Tensor.from_np(x)).to_np()
        np.testing.assert_allclose(y, _ref_layernorm(x, g, b), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_backward_analytic(self, shape):
        np.random.seed(1)
        dim = shape[-1]
        x = np.random.randn(*shape).astype(np.float32)
        dout = np.random.randn(*shape).astype(np.float32)

        ln = nn.LayerNorm(dim)
        g = ln.gamma.to_np()
        ln(Tensor.from_np(x))
        dx = ln.backward(Tensor.from_np(dout)).to_np()

        # numpy analytic reference
        eps = 1e-5
        xr = x.reshape(-1, dim)
        dr = dout.reshape(-1, dim)
        mu = xr.mean(-1, keepdims=True)
        var = xr.var(-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + eps)
        xhat = (xr - mu) * rstd
        dxh = dr * g
        c1 = dxh.mean(-1, keepdims=True)
        c2 = (dxh * xhat).mean(-1, keepdims=True)
        ref_dx = (rstd * (dxh - c1 - xhat * c2)).reshape(shape)
        ref_dg = (dr * xhat).sum(0)
        ref_db = dr.sum(0)

        np.testing.assert_allclose(dx, ref_dx, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(ln.grad_gamma.to_np(), ref_dg, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(ln.grad_beta.to_np(), ref_db, rtol=1e-3, atol=1e-3)

    def test_gradcheck(self):
        np.random.seed(2)
        dim = 6
        x = (0.5 * np.random.randn(3, dim)).astype(np.float32)
        c = np.random.randn(3, dim).astype(np.float32)
        ln = nn.LayerNorm(dim)

        # randomize gamma/beta so their grads are non-trivial
        ln.gamma.copy_from_numpy(np.random.randn(dim).astype(np.float32))
        ln.beta.copy_from_numpy(np.random.randn(dim).astype(np.float32))

        def fwd(xt):
            return ln(xt)

        ln(Tensor.from_np(x))
        dx = ln.backward(Tensor.from_np(c)).to_np()
        num_dx = numeric_grad_input(fwd, x, c)
        np.testing.assert_allclose(dx, num_dx, rtol=2e-2, atol=2e-2)

        ln(Tensor.from_np(x))
        ln.backward(Tensor.from_np(c))
        num_dg = numeric_grad_param(lambda: ln(Tensor.from_np(x)), ln.gamma, c)
        num_db = numeric_grad_param(lambda: ln(Tensor.from_np(x)), ln.beta, c)
        np.testing.assert_allclose(ln.grad_gamma.to_np(), num_dg, rtol=2e-2, atol=2e-2)
        np.testing.assert_allclose(ln.grad_beta.to_np(), num_db, rtol=2e-2, atol=2e-2)
