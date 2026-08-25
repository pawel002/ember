"""Tests for the TF32 matmul toggle and the transposed-operand GEMM path.

``ember.cuda.set_matmul_tf32`` swaps every cuBLAS GEMM onto the tensor cores,
which rounds the inputs to a 10-bit mantissa. It has to stay a real toggle (the
default is full fp32), and results must still be correct to TF32 precision.
"""

import numpy as np
import pytest

import ember as em
from ember import Tensor
from ember.nn._functional import mm


@pytest.fixture(autouse=True)
def _restore_precision():
    before = em.cuda.get_matmul_tf32()
    yield
    em.cuda.set_matmul_tf32(before)


class TestMatmulPrecision:
    def test_default_is_fp32(self):
        assert em.cuda.get_matmul_tf32() is False

    def test_toggle_roundtrip(self):
        em.cuda.set_matmul_tf32(True)
        assert em.cuda.get_matmul_tf32() is True
        em.cuda.set_matmul_tf32(False)
        assert em.cuda.get_matmul_tf32() is False

    @pytest.mark.parametrize("tf32", [False, True])
    def test_matmul_accuracy(self, tf32):
        em.cuda.set_matmul_tf32(tf32)
        rng = np.random.default_rng(0)
        a = rng.standard_normal((128, 96)).astype(np.float32)
        b = rng.standard_normal((96, 64)).astype(np.float32)
        got = (Tensor.from_np(a) @ Tensor.from_np(b)).to_np()
        want = a.astype(np.float64) @ b.astype(np.float64)
        # TF32 keeps 10 mantissa bits on the inputs; fp32 keeps 24.
        rtol = 3e-3 if tf32 else 1e-5
        np.testing.assert_allclose(got, want, rtol=rtol, atol=rtol)


class TestTransposedGemm:
    """``mm`` passes transposes as cuBLAS OP_T flags rather than materializing a
    transposed copy -- Linear.backward's two gradient products."""

    @pytest.mark.parametrize(
        "n, m, k", [(64, 32, 128), (1, 8, 16), (16, 1, 32), (5, 7, 3)]
    )
    @pytest.mark.parametrize("trans_a, trans_b", [(0, 0), (1, 0), (0, 1), (1, 1)])
    def test_against_numpy(self, n, m, k, trans_a, trans_b):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((k, n) if trans_a else (n, k)).astype(np.float32)
        b = rng.standard_normal((m, k) if trans_b else (k, m)).astype(np.float32)
        got = mm(
            Tensor.from_np(a),
            Tensor.from_np(b),
            n,
            m,
            k,
            trans_a=bool(trans_a),
            trans_b=bool(trans_b),
        )
        want = (a.T if trans_a else a).astype(np.float64) @ (
            b.T if trans_b else b
        ).astype(np.float64)
        assert got.shape == (n, m)
        np.testing.assert_allclose(got.to_np(), want, rtol=1e-4, atol=1e-5)

    def test_alpha_scales(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((8, 4)).astype(np.float32)
        b = rng.standard_normal((4, 6)).astype(np.float32)
        got = mm(Tensor.from_np(a), Tensor.from_np(b), 8, 6, 4, alpha=2.5)
        np.testing.assert_allclose(got.to_np(), 2.5 * (a @ b), rtol=1e-5, atol=1e-6)


class TestSumAxisReduction:
    """``sum(x, axis=0)`` (the Linear bias gradient) takes a two-pass split
    reduction for wide inputs and the simple kernel otherwise -- both paths, and
    the shapes on either side of the switch, must agree with NumPy."""

    @pytest.mark.parametrize(
        "shape, axis",
        [
            ((16384, 256), 0),  # split path, tall
            ((16384, 1024), 0),  # split path, wide
            ((16384, 65), 0),  # split path, narrow
            ((255, 256), 0),  # below the axis_dim threshold
            ((333, 7), 0),
            ((64, 128), 1),  # contiguous-row reduction
            ((4, 5, 6), 1),  # middle axis
            ((4, 5, 6), 0),
            ((4, 5, 6), 2),
            ((1, 1), 0),
        ],
    )
    def test_matches_numpy(self, shape, axis):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(shape).astype(np.float32)
        got = em.sum(Tensor.from_np(a), axis=axis).to_np()
        want = a.astype(np.float64).sum(axis)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    def test_repeated_calls_do_not_accumulate(self):
        # The split path writes partials into a pooled scratch buffer; a stale
        # buffer would show up as a drifting result.
        rng = np.random.default_rng(0)
        a = Tensor.from_np(rng.standard_normal((16384, 256)).astype(np.float32))
        first = em.sum(a, axis=0).to_np()
        for _ in range(5):
            np.testing.assert_array_equal(em.sum(a, axis=0).to_np(), first)
