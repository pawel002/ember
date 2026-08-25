"""Tests for the fused ("flash") attention backend kernels.

The kernels are specialized per head_dim, so these cover every supported one
(and the unsupported-head_dim fallback), plus sequence lengths that do and do
not divide the tile sizes. Forward and backward are checked against a float64
NumPy reference, and the layer is checked against its own composed path so the
two stay interchangeable.
"""

import numpy as np
import pytest

import ember as em
import ember.nn as nn
from ember import Tensor
from ember._core import _attention_bwd, _attention_fwd, _attention_supported

SUPPORTED_HEAD_DIMS = [16, 32, 64, 128]


def _ref(q, k, v, dout, scale, causal):
    """Reference attention forward + backward in float64."""
    q, k, v, dout = (a.astype(np.float64) for a in (q, k, v, dout))
    s = scale * np.einsum("bid,bjd->bij", q, k)
    if causal:
        sq, sk = s.shape[1], s.shape[2]
        mask = np.arange(sk)[None, :] > np.arange(sq)[:, None]
        s = np.where(mask[None], -np.inf, s)
    m = s.max(-1, keepdims=True)
    e = np.exp(s - m)
    l = e.sum(-1, keepdims=True)
    p = e / l
    o = np.einsum("bij,bjd->bid", p, v)

    dv = np.einsum("bij,bid->bjd", p, dout)
    dp = np.einsum("bid,bjd->bij", dout, v)
    ds = p * (dp - (dout * o).sum(-1, keepdims=True))
    dq = scale * np.einsum("bij,bjd->bid", ds, k)
    dk = scale * np.einsum("bij,bid->bjd", ds, q)
    return o, (m + np.log(l))[..., 0], dq, dk, dv


def _run(q, k, v, dout, scale, causal):
    b, s, dh = q.shape
    qt, kt, vt, gt = (Tensor.from_np(a.reshape(-1, dh)) for a in (q, k, v, dout))
    o_c, lse_c = _attention_fwd(
        qt._core, kt._core, vt._core, b, s, s, dh, scale, causal
    )
    dq_c, dk_c, dv_c = _attention_bwd(
        gt._core,
        qt._core,
        kt._core,
        vt._core,
        o_c,
        lse_c,
        b,
        s,
        s,
        dh,
        scale,
        causal,
    )
    out = [Tensor._from_core(o_c, (b, s, dh), "float32").to_np()]
    out.append(Tensor._from_core(lse_c, (b, s), "float32").to_np())
    for c in (dq_c, dk_c, dv_c):
        out.append(Tensor._from_core(c, (b, s, dh), "float32").to_np())
    return out


class TestFlashAttention:
    @pytest.mark.parametrize("head_dim", SUPPORTED_HEAD_DIMS)
    def test_supported(self, head_dim):
        assert _attention_supported(head_dim)

    @pytest.mark.parametrize("head_dim", [1, 4, 8, 48, 96])
    def test_unsupported_head_dims(self, head_dim):
        # Not an error: the layer falls back to the composed path for these.
        assert not _attention_supported(head_dim)

    @pytest.mark.parametrize("head_dim", SUPPORTED_HEAD_DIMS)
    @pytest.mark.parametrize("causal", [True, False])
    # 64/128 divide the tile sizes; 100/17/1 exercise the ragged tail.
    @pytest.mark.parametrize("seq", [128, 100, 17, 1])
    def test_against_reference(self, head_dim, causal, seq):
        if not _attention_supported(head_dim):
            pytest.skip("no fused kernel for this head_dim")
        rng = np.random.default_rng(0)
        batch = 3
        shape = (batch, seq, head_dim)
        q, k, v, dout = (
            rng.standard_normal(shape).astype(np.float32) for _ in range(4)
        )
        scale = 1.0 / np.sqrt(head_dim)

        got = _run(q, k, v, dout, scale, causal)
        want = _ref(q, k, v, dout, scale, causal)
        names = ["out", "lse", "dq", "dk", "dv"]
        for name, g, w in zip(names, got, want, strict=True):
            np.testing.assert_allclose(
                g, w, rtol=3e-5, atol=1e-5, err_msg=f"{name} mismatch"
            )

    def test_causal_first_row_ignores_later_keys(self):
        """Query 0 attends only to key 0, so its output must equal v[0]."""
        rng = np.random.default_rng(1)
        b, s, dh = 2, 64, 32
        q, k, v = (rng.standard_normal((b, s, dh)).astype(np.float32) for _ in range(3))
        dout = np.zeros((b, s, dh), dtype=np.float32)
        out = _run(q, k, v, dout, 1.0 / np.sqrt(dh), True)[0]
        np.testing.assert_allclose(out[:, 0], v[:, 0], rtol=1e-6, atol=1e-6)


class TestFusedVsComposedLayer:
    """The fused kernel and the composed bmm+softmax path must agree."""

    @pytest.mark.parametrize(
        "embed_dim, heads, seq, batch, causal",
        [
            (256, 8, 64, 2, True),  # head_dim 32
            (256, 8, 40, 2, False),  # head_dim 32, ragged seq
            (256, 4, 32, 2, True),  # head_dim 64
            (256, 2, 33, 2, True),  # head_dim 128, ragged seq
            (128, 8, 48, 3, False),  # head_dim 16
        ],
    )
    def test_matches_composed(self, embed_dim, heads, seq, batch, causal):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((batch, seq, embed_dim)).astype(np.float32) * 0.5
        g = rng.standard_normal((batch, seq, embed_dim)).astype(np.float32) * 0.5

        def run(fused):
            em.random.seed(0)
            mha = nn.MultiHeadAttention(embed_dim, heads, causal=causal)
            assert mha.fused, "expected a fused kernel for this head_dim"
            mha.fused = fused
            y = mha.forward(Tensor.from_np(x), training=True)
            dx = mha.backward(Tensor.from_np(g))
            return y.to_np(), dx.to_np(), [t.to_np() for t in mha.gradients()]

        y_f, dx_f, gr_f = run(True)
        y_c, dx_c, gr_c = run(False)
        np.testing.assert_allclose(y_f, y_c, rtol=2e-4, atol=1e-5)
        np.testing.assert_allclose(dx_f, dx_c, rtol=2e-4, atol=1e-5)
        for a, b in zip(gr_f, gr_c, strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-4, atol=1e-4)

    def test_small_head_dim_uses_fallback(self):
        mha = nn.MultiHeadAttention(8, 2)  # head_dim 4
        assert not mha.fused
        x = Tensor.from_np(np.zeros((1, 4, 8), dtype=np.float32))
        assert mha.forward(x, training=True).shape == (1, 4, 8)
