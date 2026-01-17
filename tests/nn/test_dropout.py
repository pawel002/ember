import numpy as np
import pytest

import ember.nn as nn
from ember.tensor import Tensor


class TestDropout:
    PROBS = [0.2, 0.5, 0.8]

    @pytest.mark.parametrize("p", PROBS)
    def test_dropout_training_inverted(self, p):
        shape = (100, 100)
        x_np = np.ones(shape, dtype=np.float32)
        x_t = Tensor.from_np(x_np)

        layer = nn.Dropout(p=p)
        y_t = layer.forward(x_t, training=True)
        y_np = y_t.to_np()
        mask_np = layer.mask.to_np()

        expected_scale = 1.0 / (1.0 - p)

        expected_output = x_np * mask_np * expected_scale
        np.testing.assert_allclose(y_np, expected_output, rtol=1e-5)

        assert abs(np.mean(y_np) - 1.0) < 0.1

    @pytest.mark.parametrize("p", PROBS)
    def test_dropout_inference_identity(self, p):
        shape = (10, 10)
        x_np = np.random.randn(*shape).astype(np.float32)
        x_t = Tensor.from_np(x_np)

        layer = nn.Dropout(p=p)

        y_t = layer.forward(x_t, training=False)

        np.testing.assert_allclose(y_t.to_np(), x_np)
        assert layer.mask is None

    def test_dropout_backward_scaled(self):
        p = 0.5
        layer = nn.Dropout(p=p)
        x_t = Tensor.from_np(np.ones((5, 5), dtype=np.float32))

        layer.forward(x_t, training=True)
        mask_val = layer.mask.to_np()

        grad_out = Tensor.from_np(np.ones((5, 5), dtype=np.float32))
        grad_in = layer.backward(grad_out)

        scale = 1.0 / (1.0 - p)
        expected_grad = mask_val * scale

        np.testing.assert_allclose(grad_in.to_np(), expected_grad)
