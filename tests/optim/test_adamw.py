import numpy as np
import pytest

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


class TestAdamWForeach:
    """``foreach=True`` updates every parameter in one grouped kernel launch; it
    must reproduce the per-parameter path exactly, including across the
    re-upload that happens when the parameter set changes."""

    SHAPES = [(65, 256), (256,), (256, 1024), (1024,), (7,), (1,)]

    def _run(self, foreach, shapes, steps):
        rng = np.random.default_rng(0)
        params = [
            Tensor.from_np(rng.standard_normal(s).astype(np.float32)) for s in shapes
        ]
        opt = optim.AdamW(params, lr=1e-3, weight_decay=0.1, foreach=foreach)
        for grads in steps:
            opt.apply([Tensor.from_np(g) for g in grads])
        return [p.to_np() for p in params]

    def _grad_steps(self, shapes, n):
        rng = np.random.default_rng(1)
        return [
            [rng.standard_normal(s).astype(np.float32) for s in shapes]
            for _ in range(n)
        ]

    def test_matches_per_parameter_path(self):
        steps = self._grad_steps(self.SHAPES, 5)
        for a, b in zip(
            self._run(False, self.SHAPES, steps),
            self._run(True, self.SHAPES, steps),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("shapes", [[(4,)], [(3, 5), (7,)], [(1,)] * 20])
    def test_various_parameter_counts(self, shapes):
        steps = self._grad_steps(shapes, 3)
        for a, b in zip(
            self._run(False, shapes, steps), self._run(True, shapes, steps), strict=True
        ):
            np.testing.assert_array_equal(a, b)

    def test_two_optimizers_interleaved(self):
        """The grouped path caches its device pointer arrays; two optimizers
        alternating must each still see their own parameters."""
        rng = np.random.default_rng(2)
        a = [Tensor.from_np(rng.standard_normal((4,)).astype(np.float32))]
        b = [Tensor.from_np(rng.standard_normal((9,)).astype(np.float32))]
        a_ref = [Tensor.from_np(a[0].to_np())]
        b_ref = [Tensor.from_np(b[0].to_np())]
        oa = optim.AdamW(a, lr=1e-2, foreach=True)
        ob = optim.AdamW(b, lr=1e-2, foreach=True)
        oa_ref = optim.AdamW(a_ref, lr=1e-2)
        ob_ref = optim.AdamW(b_ref, lr=1e-2)
        for _ in range(4):
            ga = rng.standard_normal((4,)).astype(np.float32)
            gb = rng.standard_normal((9,)).astype(np.float32)
            oa.apply([Tensor.from_np(ga)])
            ob.apply([Tensor.from_np(gb)])
            oa_ref.apply([Tensor.from_np(ga)])
            ob_ref.apply([Tensor.from_np(gb)])
        np.testing.assert_array_equal(a[0].to_np(), a_ref[0].to_np())
        np.testing.assert_array_equal(b[0].to_np(), b_ref[0].to_np())

    def test_capturable_wins_over_foreach(self):
        p = [Tensor.from_np(np.ones((2,), dtype=np.float32))]
        opt = optim.AdamW(p, capturable=True, foreach=True)
        assert not opt.foreach
