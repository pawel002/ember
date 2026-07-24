"""Tests for the fused softmax backend kernels that back the attention path
(forward over an arbitrary axis / contiguous rows, causal masking, backward)."""

import numpy as np
import pytest

from ember import Tensor
from ember._core import (
    _softmax_bwd,
    _softmax_fwd,
    _softmax_rows,
    _softmax_rows_causal,
)


def _core(a):
    return Tensor.from_np(np.ascontiguousarray(a, np.float32))._core


def _np(c, shape):
    return Tensor._from_core(c, shape, "float32").to_np()


def _softmax(z, axis):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


class TestFusedSoftmax:
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_forward_axis(self, axis):
        np.random.seed(0)
        x = np.random.randn(3, 4, 5).astype(np.float32)
        out = _np(_softmax_fwd(_core(x), tuple(x.shape), axis), x.shape)
        np.testing.assert_allclose(out, _softmax(x, axis), rtol=1e-4, atol=1e-5)

    def test_backward_axis(self):
        np.random.seed(1)
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = _softmax(x, -1)
        dy = np.random.randn(*x.shape).astype(np.float32)
        dx = _np(_softmax_bwd(_core(dy), _core(y), tuple(x.shape), 2), x.shape)
        ref = y * (dy - (dy * y).sum(-1, keepdims=True))
        np.testing.assert_allclose(dx, ref, rtol=1e-4, atol=1e-5)

    def test_rows(self):
        np.random.seed(2)
        rows, d = 9, 7
        x = np.random.randn(rows, d).astype(np.float32)
        out = _np(_softmax_rows(_core(x), rows, d), (rows, d))
        np.testing.assert_allclose(out, _softmax(x, -1), rtol=1e-4, atol=1e-5)

    def test_rows_causal(self):
        np.random.seed(3)
        rows, d = 6, 6  # sq == d (self-attention square)
        x = np.random.randn(rows, d).astype(np.float32)
        out = _np(_softmax_rows_causal(_core(x), rows, d, rows), (rows, d))
        ref = np.zeros_like(x)
        for r in range(rows):
            v = x[r, : r + 1]
            e = np.exp(v - v.max())
            ref[r, : r + 1] = e / e.sum()
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)
        # masked entries are exactly zero
        assert np.all(out[np.triu(np.ones((rows, d), bool), 1)] == 0.0)
