import numpy as np
import pytest

import ember.optim as optim
from ember import Tensor

from .._reference import adam_step


class TestAdam:
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

        expected_param, expected_m, expected_v = adam_step(
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

            np_param, np_m, np_v = adam_step(
                np_param, grad_np, np_m, np_v, t, lr, b1, b2, eps
            )

            np.testing.assert_allclose(t_param.to_np(), np_param, rtol=1e-5)

        # After 5 Adam steps (lr=0.1) from x=2, the parameter has moved
        # meaningfully toward the minimum at 0 but not yet reached it (~1.5).
        assert t_param.to_np()[0] < 2.0
        assert t_param.to_np()[0] > 0.0


@pytest.mark.parametrize("opt_cls", [optim.Adam, optim.AdamW])
@pytest.mark.parametrize("mode", ["foreach", "capturable"])
def test_variants_match_per_parameter_path(opt_cls, mode):
    """The grouped and device-step-counter paths compute the same update as
    the default one-launch-per-parameter path."""
    rng = np.random.default_rng(0)
    shapes = [(3, 4), (7,), (2, 5)]
    init = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    grads = [
        [rng.standard_normal(s).astype(np.float32) for s in shapes] for _ in range(4)
    ]

    def run(**kw):
        params = [Tensor.from_np(a) for a in init]
        opt = opt_cls(params, lr=1e-2, **kw)
        for step in grads:
            opt.apply([Tensor.from_np(g) for g in step])
        return [p.to_np() for p in params]

    # capturable computes the bias corrections in float32 on the device (the
    # default path uses double on the host), so it agrees to float32 precision.
    tol = {"rtol": 1e-6, "atol": 1e-7} if mode == "foreach" else {"rtol": 1e-5}
    for a, b in zip(run(), run(**{mode: True}), strict=True):
        np.testing.assert_allclose(a, b, **tol)


def test_adam_is_adamw_without_weight_decay():
    rng = np.random.default_rng(1)
    init = rng.standard_normal((4, 3)).astype(np.float32)
    grads = [rng.standard_normal((4, 3)).astype(np.float32) for _ in range(3)]

    def run(opt_cls, **kw):
        p = Tensor.from_np(init)
        opt = opt_cls([p], lr=1e-2, **kw)
        for g in grads:
            opt.apply([Tensor.from_np(g)])
        return p.to_np()

    np.testing.assert_array_equal(run(optim.Adam), run(optim.AdamW, weight_decay=0.0))


def test_missing_gradient_raises():
    p = Tensor.from_np(np.ones((2,), dtype=np.float32))
    with pytest.raises(ValueError, match="None"):
        optim.Adam([p]).apply([None])
