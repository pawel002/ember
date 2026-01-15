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

        w_ref = layer.w.to_np()
        b_ref = layer.b.to_np()

        # forward pass
        y_t = layer.forward(x_t, training=True)
        expected_y = x_np @ w_ref + b_ref

        np.testing.assert_allclose(
            y_t.to_np(), expected_y, rtol=1e-5, err_msg="Forward pass output mismatch"
        )

        # backward pass
        grad_x_t = layer.backward(grad_y_t)
        expected_grad_x = grad_y_np @ w_ref.T

        np.testing.assert_allclose(
            grad_x_t.to_np(),
            expected_grad_x,
            rtol=1e-5,
            err_msg="Gradient w.r.t Input (grad_x) mismatch",
        )

        # gradients w.r.t. weights
        expected_grad_w = x_np.T @ grad_y_np

        assert layer.grad_w is not None, "grad_w was not populated"
        np.testing.assert_allclose(
            layer.grad_w.to_np(),
            expected_grad_w,
            rtol=1e-5,
            err_msg="Gradient w.r.t Weights (grad_w) mismatch",
        )

        # gradients w.r.t. bias
        expected_grad_b = np.sum(grad_y_np, axis=0)

        assert layer.grad_b is not None, "grad_b was not populated"
        np.testing.assert_allclose(
            layer.grad_b.to_np(),
            expected_grad_b,
            rtol=1e-5,
            err_msg="Gradient w.r.t Bias (grad_b) mismatch",
        )
