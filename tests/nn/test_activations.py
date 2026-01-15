import numpy as np
import pytest

import ember.nn as nn
from ember import Tensor


class TestActivations:
    SHAPES = [(64,), (32, 128), (8, 10, 64)]
    ACTIVATION_CONFIGS = [
        ("ReLU", nn.ReLU(), lambda x: np.maximum(x, 0), lambda x, _, g: g * (x > 0)),
        (
            "Sigmoid",
            nn.Sigmoid(),
            lambda x: 1.0 / (1.0 + np.exp(-x)),
            lambda _, y, g: g * (y * (1.0 - y)),
        ),
        ("Tanh", nn.Tanh(), lambda x: np.tanh(x), lambda _, y, g: g * (1.0 - y**2)),
        (
            "GELU",
            nn.GELU(),
            lambda x: 0.5 * x * (1.0 + np.tanh(0.8 * x, dtype=np.float32)),
            lambda x, y, g: g * ((1.0 + np.tanh(0.8 * x)) * (0.5 + 0.8 * (x - y))),
        ),
    ]

    def _get_inputs(self, shape):
        np.random.seed(42)
        x_np = np.random.randn(*shape).astype(np.float32)
        grad_np = np.random.randn(*shape).astype(np.float32)

        x_t = Tensor.from_np(x_np)
        grad_t = Tensor.from_np(grad_np)
        return x_np, grad_np, x_t, grad_t

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize(
        "name, layer, ref_forward, ref_backward", ACTIVATION_CONFIGS
    )
    def test_activation(self, shape, layer, name, ref_forward, ref_backward):
        x_np, grad_np, x_t, grad_t = self._get_inputs(shape)

        y_t = layer.forward(x_t, training=True)
        y_np_expected = ref_forward(x_np)

        np.testing.assert_allclose(
            y_t.to_np(),
            y_np_expected,
            rtol=1e-4,
            err_msg=f"{name} Forward mismatch at shape {shape}",
        )

        dx_t = layer.backward(grad_t)
        dx_np_expected = ref_backward(x_np, y_np_expected, grad_np)

        np.testing.assert_allclose(
            dx_t.to_np(),
            dx_np_expected,
            rtol=1e-4,
            err_msg=f"{name} Backward mismatch at shape {shape}",
        )
