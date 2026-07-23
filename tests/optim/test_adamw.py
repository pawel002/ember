import numpy as np

import ember.optim as optim
from ember import Tensor


class TestAdamW:
    def _np_step(self, p, g, m, v, t, lr, b1, b2, eps, wd):
        m_new = b1 * m + (1 - b1) * g
        v_new = b2 * v + (1 - b2) * (g**2)

        m_hat = m_new / (1 - b1**t)
        v_hat = v_new / (1 - b2**t)

        p_new = p - lr * wd * p
        p_new = p_new - lr * m_hat / (np.sqrt(v_hat) + eps)
        return p_new, m_new, v_new

    def test_step_math(self):
        lr, b1, b2, eps, wd = 0.01, 0.9, 0.999, 1e-8, 0.1

        np_param = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np_grad = np.array([[0.1, -0.1], [0.5, 0.0]], dtype=np.float32)
        np_m = np.zeros_like(np_param)
        np_v = np.zeros_like(np_param)

        t_param = Tensor.from_np(np_param)
        opt = optim.AdamW([t_param], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)

        for step in range(1, 4):
            opt.apply([Tensor.from_np(np_grad)])
            np_param, np_m, np_v = self._np_step(
                np_param, np_grad, np_m, np_v, step, lr, b1, b2, eps, wd
            )
            np.testing.assert_allclose(t_param.to_np(), np_param, rtol=1e-4, atol=1e-6)
            assert opt.t == step

    def test_weight_decay_shrinks_params_without_grad(self):
        # With zero gradient, AdamW still decays the parameter toward zero.
        np_param = np.full((3,), 2.0, dtype=np.float32)
        t_param = Tensor.from_np(np_param)
        opt = optim.AdamW([t_param], lr=0.1, weight_decay=0.5)

        zero_grad = Tensor.from_np(np.zeros((3,), dtype=np.float32))
        opt.apply([zero_grad])

        # p -= lr*wd*p (the Adam term is ~0 because the gradient is 0)
        assert np.all(t_param.to_np() < 2.0)
