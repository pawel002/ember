import math
from collections.abc import Sequence

from ember._core import (
    _attention_bwd,
    _attention_fwd,
    _attention_supported,
    _softmax_bwd,
    _softmax_rows,
    _softmax_rows_causal,
)
from ember.tensor import Tensor

from ._functional import bmm, permute_0213
from .base import Layer
from .layers import Linear


class MultiHeadAttention(Layer):
    """Multi-head self-attention.

    Splits the model dimension ``embed_dim`` into ``num_heads`` heads of size
    ``head_dim = embed_dim / num_heads`` and computes, per head,

        Attention(Q, K, V) = softmax(Q K^T / sqrt(head_dim)) V

    Input and output are ``(batch, seq, embed_dim)``. The Q/K/V/output
    projections are plain :class:`Linear` layers (fused GEMM+bias).

    The attention itself uses the fused ("flash") kernel when the backend has
    one for this ``head_dim``: scores are tiled through shared memory and never
    materialized, which for a seq-256 model removes ~1.5 GB of memory traffic
    per block per step. That path also reads the projections in their natural
    ``(batch, seq, heads, head_dim)`` layout, so the split-into-heads transpose
    (four copies forward, four more backward) disappears with it.

    Otherwise it falls back to the composed path -- transpose to heads, batched
    GEMM, fused row-softmax with the causal mask folded in, batched GEMM --
    which materializes the ``(seq, seq)`` score matrix.
    """

    def __init__(self, embed_dim: int, num_heads: int, causal: bool = False):
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = causal
        self.fused = bool(_attention_supported(self.head_dim))

        # projections
        self.wq = Linear(embed_dim, embed_dim)
        self.wk = Linear(embed_dim, embed_dim)
        self.wv = Linear(embed_dim, embed_dim)
        self.wo = Linear(embed_dim, embed_dim)

        # Saved for backward: the fused path keeps (q, k, v, out, lse) in their
        # natural layout, the composed path the head-major (qh, kh, vh) and the
        # softmax p.
        self._saved: tuple[Tensor, ...] = ()
        self._shape: tuple[int, int, int] | None = None

    def sublayers(self) -> Sequence[Layer]:
        return (self.wq, self.wk, self.wv, self.wo)

    def _to_heads(self, t: Tensor, B: int, S: int) -> Tensor:
        # (B*S, E) -> (B, S, H, dh) -> (B, H, S, dh) == (B*H, S, dh)
        h = t.reshape((B, S, self.num_heads, self.head_dim))
        h = permute_0213(h, B, S, self.num_heads, self.head_dim)
        return h.reshape((B * self.num_heads, S, self.head_dim))

    def _from_heads(self, t: Tensor, B: int, S: int) -> Tensor:
        # (B*H, S, dh) -> (B, H, S, dh) -> (B, S, H, dh) -> (B*S, E)
        h = t.reshape((B, self.num_heads, S, self.head_dim))
        h = permute_0213(h, B, self.num_heads, S, self.head_dim)
        return h.reshape((B * S, self.embed_dim))

    def forward(self, x: Tensor, training: bool) -> Tensor:
        if len(x.shape) != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"MultiHeadAttention expects (batch, seq, {self.embed_dim}), "
                f"got {x.shape}"
            )
        B, S, E = x.shape
        H, dh = self.num_heads, self.head_dim
        BH = B * H
        self._shape = (B, S, E)

        x2 = x.reshape((B * S, E))
        q = self.wq(x2, training)
        k = self.wk(x2, training)
        v = self.wv(x2, training)

        if self.fused:
            # One kernel: scores, softmax and P@V, tiled through shared memory.
            # `lse` (the per-row log-sum-exp) is what lets backward rebuild the
            # softmax without the score matrix ever existing. The trailing
            # (H, E) arguments describe the interleaving, so the kernel walks
            # q/k/v as (B, S, H, dh) in place -- nothing is copied into
            # head-major order, and the output comes back in the same layout.
            o_core, lse_core = _attention_fwd(
                q._core, k._core, v._core, BH, S, S, dh, self.scale, self.causal, H, E
            )
            o = Tensor._from_core(o_core, (B * S, E), x.dtype)
            lse = Tensor._from_core(lse_core, (BH, S), x.dtype)
            self._saved = (q, k, v, o, lse)
            out2 = self.wo(o, training)  # (B*S, E)
            return out2.reshape((B, S, E))

        qh = self._to_heads(q, B, S)
        kh = self._to_heads(k, B, S)
        vh = self._to_heads(v, B, S)

        # scores = scale * Q @ K^T  ->  (B*H, S, S)
        scores = bmm(
            qh, kh, BH, S, S, dh, trans_a=False, trans_b=True, alpha=self.scale
        )

        # row softmax over the key axis (fused; causal mask folded in)
        if self.causal:
            p_core = _softmax_rows_causal(scores._core, BH * S, S, S)
        else:
            p_core = _softmax_rows(scores._core, BH * S, S)
        p = Tensor._from_core(p_core, (BH, S, S), x.dtype)

        # O = P @ V  ->  (B*H, S, dh)
        o = bmm(p, vh, BH, S, dh, S)
        self._saved = (qh, kh, vh, p)

        out2 = self.wo(self._from_heads(o, B, S), training)  # (B*S, E)
        return out2.reshape((B, S, E))

    def backward(self, grad_y: Tensor) -> Tensor:
        if self._shape is None:
            raise RuntimeError("forward() must run before backward()")
        B, S, E = self._shape
        H, dh = self.num_heads, self.head_dim
        BH = B * H

        g2 = grad_y.reshape((B * S, E))
        do2 = self.wo.backward(g2)  # dL/d(concat heads), (B*S, E)

        if self.fused:
            q, k, v, o, lse = self._saved
            # Same interleaved layout on the way in and out: dq/dk/dv land as
            # (B*S, E) and feed the projections directly.
            dq_c, dk_c, dv_c = _attention_bwd(
                do2._core,
                q._core,
                k._core,
                v._core,
                o._core,
                lse._core,
                BH,
                S,
                S,
                dh,
                self.scale,
                self.causal,
                H,
                E,
            )
            dtype = grad_y.dtype
            dxq = self.wq.backward(Tensor._from_core(dq_c, (B * S, E), dtype))
            dxk = self.wk.backward(Tensor._from_core(dk_c, (B * S, E), dtype))
            dxv = self.wv.backward(Tensor._from_core(dv_c, (B * S, E), dtype))
            return (dxq + dxk + dxv).reshape((B, S, E))

        qh, kh, vh, p = self._saved
        do = self._to_heads(do2, B, S)  # (B*H, S, dh)

        # dP = dO @ V^T ; dV = P^T @ dO
        dp = bmm(do, vh, BH, S, S, dh, trans_a=False, trans_b=True)
        dv = bmm(p, do, BH, S, dh, S, trans_a=True, trans_b=False)

        # softmax backward over the key axis (P zeros make causal grads vanish)
        dscores_core = _softmax_bwd(dp._core, p._core, (BH * S, S), 1)
        dscores = Tensor._from_core(dscores_core, (BH, S, S), grad_y.dtype)

        # dQ = scale * dScores @ K ; dK = scale * dScores^T @ Q
        dq = bmm(dscores, kh, BH, S, dh, S, alpha=self.scale)
        dk = bmm(dscores, qh, BH, S, dh, S, trans_a=True, alpha=self.scale)

        dxq = self.wq.backward(self._from_heads(dq, B, S))
        dxk = self.wk.backward(self._from_heads(dk, B, S))
        dxv = self.wv.backward(self._from_heads(dv, B, S))

        return (dxq + dxk + dxv).reshape((B, S, E))
