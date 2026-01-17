import numpy as np

import ember.optim as optim
from ember import Tensor


class TestAdam:
    def _numpy_adam_step(self, param, grad, m, v, t, lr, b1, b2, eps):
        m_new = b1 * m + (1 - b1) * grad
        v_new = b2 * v + (1 - b2) * (grad**2)

        m_hat = m_new / (1 - b1**t)
        v_hat = v_new / (1 - b2**t)

        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        param_new = param - update

        return param_new, m_new, v_new

    def test_adam_initialization(self):
        shape = (2, 3)
        t_param = Tensor.from_np(np.random.randn(*shape).astype(np.float32))
        optimizer = optim.Adam([t_param], lr=0.1)

        assert len(optimizer.means) == 1
        assert len(optimizer.variances) == 1

        np.testing.assert_allclose(optimizer.means[0].to_np(), 0.0)
        np.testing.assert_allclose(optimizer.variances[0].to_np(), 0.0)
        assert optimizer.t == 0

    def test_adam_step_math(self):
        lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8

        np_param = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np_grad = np.array([[0.1, -0.1], [0.5, 0.0]], dtype=np.float32)

        np_m = np.zeros_like(np_param)
        np_v = np.zeros_like(np_param)

        t_param = Tensor.from_np(np_param)
        t_grad = Tensor.from_np(np_grad)

        optimizer = optim.Adam([t_param], lr=lr, betas=(b1, b2), eps=eps)

        optimizer.apply([t_grad])

        expected_param, expected_m, expected_v = self._numpy_adam_step(
            np_param, np_grad, np_m, np_v, t=1, lr=lr, b1=b1, b2=b2, eps=eps
        )

        np.testing.assert_allclose(
            t_param.to_np(),
            expected_param,
            rtol=1e-5,
            err_msg="Parameter update failed",
        )

        np.testing.assert_allclose(
            optimizer.means[0].to_np(),
            expected_m,
            rtol=1e-5,
            err_msg="Momentum (m) buffer failed",
        )

        np.testing.assert_allclose(
            optimizer.variances[0].to_np(),
            expected_v,
            rtol=1e-5,
            err_msg="Variance (v) buffer failed",
        )

        assert optimizer.t == 1

    def test_adam_multistep_convergence(self):
        lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8

        np_param = np.array([2.0], dtype=np.float32)
        t_param = Tensor.from_np(np_param)

        optimizer = optim.Adam([t_param], lr=lr, betas=(b1, b2), eps=eps)

        np_m = np.zeros_like(np_param)
        np_v = np.zeros_like(np_param)

        for t in range(1, 6):
            curr_x_np = np_param
            grad_np = 2 * curr_x_np

            curr_x_t_val = t_param.to_np()
            t_grad = Tensor.from_np(2 * curr_x_t_val)

            optimizer.apply([t_grad])

            np_param, np_m, np_v = self._numpy_adam_step(
                np_param, grad_np, np_m, np_v, t, lr, b1, b2, eps
            )

            np.testing.assert_allclose(t_param.to_np(), np_param, rtol=1e-5)

        assert t_param.to_np()[0] < 1.0
        assert t_param.to_np()[0] > -0.5
