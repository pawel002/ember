import numpy as np
import pytest

import ember.loss as loss
from ember import Tensor


class TestMSELoss:
    SHAPES = [(10,), (32, 5), (8, 3, 4)]

    @pytest.mark.parametrize("shape", SHAPES)
    def test_forward_backward(self, shape):
        np.random.seed(0)
        pred_np = np.random.randn(*shape).astype(np.float32)
        target_np = np.random.randn(*shape).astype(np.float32)

        fn = loss.MSELoss()
        value = fn(Tensor.from_np(pred_np), Tensor.from_np(target_np))

        expected = np.mean((pred_np - target_np) ** 2)
        assert abs(value - expected) < 1e-4

        grad = fn.backward().to_np()
        expected_grad = 2.0 * (pred_np - target_np) / pred_np.size
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-4, atol=1e-5)


class TestCrossEntropyLoss:
    SHAPES = [(4, 3), (16, 10), (1, 5)]

    def _one_hot(self, n, c):
        labels = np.random.randint(0, c, size=n)
        oh = np.zeros((n, c), dtype=np.float32)
        oh[np.arange(n), labels] = 1.0
        return oh

    @pytest.mark.parametrize("n, c", SHAPES)
    def test_forward_backward(self, n, c):
        np.random.seed(1)
        logits_np = np.random.randn(n, c).astype(np.float32)
        target_np = self._one_hot(n, c)

        fn = loss.CrossEntropyLoss()
        value = fn(Tensor.from_np(logits_np), Tensor.from_np(target_np))

        # numpy reference (log-sum-exp)
        shifted = logits_np - logits_np.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        expected = -(target_np * log_probs).sum() / n
        assert abs(value - expected) < 1e-4

        grad = fn.backward().to_np()
        probs = np.exp(log_probs)
        expected_grad = (probs - target_np) / n
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-4, atol=1e-5)
