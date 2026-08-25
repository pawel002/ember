"""Tests for the TF32 matmul toggle.

``ember.cuda.set_matmul_tf32`` swaps every cuBLAS GEMM onto the tensor cores,
which rounds the inputs to a 10-bit mantissa. It has to stay a real toggle (the
default is full fp32), and results must still be correct to TF32 precision.
"""

import numpy as np
import pytest

import ember as em
from ember import Tensor


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
