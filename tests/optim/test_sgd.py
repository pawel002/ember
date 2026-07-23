import numpy as np
import pytest

import ember.optim as optim
from ember import Tensor


class TestSGD:
    def _np_step(self, p, g, v, lr, momentum):
        v_new = momentum * v + g
        p_new = p - lr * v_new
        return p_new, v_new

    @pytest.mark.parametrize("momentum", [0.0, 0.9])
    def test_step_math(self, momentum):
        lr = 0.1
        np_param = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np_grad = np.array([[0.1, -0.2], [0.5, 0.0]], dtype=np.float32)

        t_param = Tensor.from_np(np_param)
        opt = optim.SGD([t_param], lr=lr, momentum=momentum)

        np_v = np.zeros_like(np_param)
        for _ in range(3):
            opt.apply([Tensor.from_np(np_grad)])
            np_param, np_v = self._np_step(np_param, np_grad, np_v, lr, momentum)
            np.testing.assert_allclose(t_param.to_np(), np_param, rtol=1e-5, atol=1e-6)

    def test_gradient_count_mismatch_raises(self):
        t_param = Tensor.from_np(np.zeros((2, 2), dtype=np.float32))
        opt = optim.SGD([t_param])
        with pytest.raises(ValueError):
            opt.apply([])

    def test_converges_toward_minimum(self):
        # minimize f(x) = x^2, grad = 2x, starting at x = 5
        t_param = Tensor.from_np(np.array([5.0], dtype=np.float32))
        opt = optim.SGD([t_param], lr=0.1)

        for _ in range(50):
            grad = Tensor.from_np(2.0 * t_param.to_np())
            opt.apply([grad])

        assert abs(t_param.to_np()[0]) < 0.1
