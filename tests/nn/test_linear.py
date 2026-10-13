import numpy as np
import pytest

import ember.nn as nn
from ember.tensor import Tensor


class TestLinear:
    # (batch_size, in_features, out_features)
    SHAPES = [
        (32, 128, 64),
        (1, 10, 5),
        (64, 32, 128),
        (16, 50, 50),
    ]

    def _get_inputs(self, N, in_feat, out_feat):
        np.random.seed(42)
        x_np = np.random.randn(N, in_feat).astype(np.float32)
        grad_y_np = np.random.randn(N, out_feat).astype(np.float32)

        return x_np, grad_y_np

    @pytest.mark.parametrize("N, in_feat, out_feat", SHAPES)
    def test_linear_forward_backward(self, N, in_feat, out_feat):
        x_np, grad_y_np = self._get_inputs(N, in_feat, out_feat)

        layer = nn.Linear(in_features=in_feat, out_features=out_feat)

        x_t = Tensor.from_np(x_np)
        grad_y_t = Tensor.from_np(grad_y_np)

        w_ref = layer.weight.to_np()
        b_ref = layer.bias.to_np()

        # forward pass
        y_t = layer.forward(x_t, training=True)
        expected_y = x_np @ w_ref + b_ref

        np.testing.assert_allclose(
            y_t.to_np(),
            expected_y,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Forward pass output mismatch",
        )

        # backward pass
        grad_x_t = layer.backward(grad_y_t)
        expected_grad_x = grad_y_np @ w_ref.T

        np.testing.assert_allclose(
            grad_x_t.to_np(),
            expected_grad_x,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Gradient w.r.t Input (grad_x) mismatch",
        )

        # gradients w.r.t. weights
        expected_grad_w = x_np.T @ grad_y_np

        assert layer.grad_weight is not None, "grad_weight was not populated"
        np.testing.assert_allclose(
            layer.grad_weight.to_np(),
            expected_grad_w,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Gradient w.r.t Weights (grad_weight) mismatch",
        )

        # gradients w.r.t. bias
        expected_grad_b = np.sum(grad_y_np, axis=0)

        assert layer.grad_bias is not None, "grad_bias was not populated"
        np.testing.assert_allclose(
            layer.grad_bias.to_np(),
            expected_grad_b,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Gradient w.r.t Bias (grad_bias) mismatch",
        )


def test_linear_accepts_leading_dims():
    """(batch, seq, in) behaves like the flattened (batch*seq, in) input."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    g = rng.standard_normal((2, 3, 5)).astype(np.float32)

    nd = nn.Linear(4, 5)
    flat = nn.Linear(4, 5)
    flat.weight.copy_from_numpy(nd.weight.to_np())
    flat.bias.copy_from_numpy(nd.bias.to_np())

    y = nd(Tensor.from_np(x))
    assert y.shape == (2, 3, 5)
    dx = nd.backward(Tensor.from_np(g))
    assert dx.shape == (2, 3, 4)

    y_ref = flat(Tensor.from_np(x.reshape(6, 4)))
    dx_ref = flat.backward(Tensor.from_np(g.reshape(6, 5)))
    np.testing.assert_allclose(y.to_np().reshape(6, 5), y_ref.to_np(), rtol=1e-6)
    np.testing.assert_allclose(dx.to_np().reshape(6, 4), dx_ref.to_np(), rtol=1e-6)
    np.testing.assert_allclose(
        nd.grad_weight.to_np(), flat.grad_weight.to_np(), rtol=1e-5
    )
    np.testing.assert_allclose(nd.grad_bias.to_np(), flat.grad_bias.to_np(), rtol=1e-5)


def test_linear_rejects_wrong_feature_dim():
    with pytest.raises(ValueError, match=r"\(\.\.\., 4\)"):
        nn.Linear(4, 5)(Tensor.from_np(np.ones((3, 6), np.float32)))
