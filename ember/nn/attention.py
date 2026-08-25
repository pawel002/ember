import math

from ember._core import (
    _attention_bwd,
    _attention_fwd,
    _attention_supported,
    _softmax_bwd,
    _softmax_rows,
    _softmax_rows_causal,
)
from ember.tensor import Tensor

from ._functional import bmm, permute_0213, view
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
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = causal
        self.fused = bool(_attention_supported(self.head_dim))
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        # projections
        self.wq = Linear(self.embed_dim, self.embed_dim)
        self.wk = Linear(self.embed_dim, self.embed_dim)
        self.wv = Linear(self.embed_dim, self.embed_dim)
        self.wo = Linear(self.embed_dim, self.embed_dim)
        # cached activations for backward: the fused path keeps the untransposed
        # projections, the composed path their head-major copies.
        self._q = self._k = self._v = self._o = self._lse = None
        self._qh = self._kh = self._vh = self._p = None
        self._shape: tuple[int, int, int] | None = None

    def parameters(self) -> list[Tensor]:
        out: list[Tensor] = []
        for proj in (self.wq, self.wk, self.wv, self.wo):
            out.extend(proj.parameters())
        return out

    def gradients(self) -> list[Tensor | None]:
        out: list[Tensor | None] = []
        for proj in (self.wq, self.wk, self.wv, self.wo):
            out.extend(proj.gradients())
        return out

    def _to_heads(self, t: Tensor, B: int, S: int) -> Tensor:
        # (B*S, E) -> (B, S, H, dh) -> (B, H, S, dh) == (B*H, S, dh)
        h = view(t, (B, S, self.num_heads, self.head_dim))
        h = permute_0213(h, B, S, self.num_heads, self.head_dim)
        return view(h, (B * self.num_heads, S, self.head_dim))

    def _from_heads(self, t: Tensor, B: int, S: int) -> Tensor:
        # (B*H, S, dh) -> (B, H, S, dh) -> (B, S, H, dh) -> (B*S, E)
        h = view(t, (B, self.num_heads, S, self.head_dim))
        h = permute_0213(h, B, self.num_heads, S, self.head_dim)
        return view(h, (B * S, self.embed_dim))

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        B, S, E = x.shape
        H, dh = self.num_heads, self.head_dim
        BH = B * H
        self._shape = (B, S, E)

        x2 = view(x, (B * S, E))
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
            self._q, self._k, self._v = q, k, v
            self._o = Tensor._from_core(o_core, (B * S, E), x.dtype)
            self._lse = Tensor._from_core(lse_core, (BH, S), x.dtype)
            out2 = self.wo(self._o, training)  # (B*S, E)
            self.y = view(out2, (B, S, E))
            return self.y

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
        self._p = Tensor._from_core(p_core, (BH, S, S), x.dtype)

        # O = P @ V  ->  (B*H, S, dh)
        o = bmm(self._p, vh, BH, S, dh, S)
        self._qh, self._kh, self._vh = qh, kh, vh

        out2 = self.wo(self._from_heads(o, B, S), training)  # (B*S, E)
        self.y = view(out2, (B, S, E))
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        assert self._shape is not None, "forward() must run before backward()"
        B, S, E = self._shape
        H, dh = self.num_heads, self.head_dim
        BH = B * H

        g2 = view(grad_y, (B * S, E))
        do2 = self.wo.backward(g2)  # dL/d(concat heads), (B*S, E)

        if self.fused:
            assert self._o is not None and self._lse is not None
            # Same interleaved layout on the way in and out: dq/dk/dv land as
            # (B*S, E) and feed the projections directly.
            dq_c, dk_c, dv_c = _attention_bwd(
                do2._core,
                self._q._core,
                self._k._core,
                self._v._core,
                self._o._core,
                self._lse._core,
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
            return view(dxq + dxk + dxv, (B, S, E))

        do = self._to_heads(do2, B, S)  # (B*H, S, dh)

        # dP = dO @ V^T ; dV = P^T @ dO
        dp = bmm(do, self._vh, BH, S, S, dh, trans_a=False, trans_b=True)
        dv = bmm(self._p, do, BH, S, dh, S, trans_a=True, trans_b=False)

        # softmax backward over the key axis (P zeros make causal grads vanish)
        dscores_core = _softmax_bwd(dp._core, self._p._core, (BH * S, S), 1)
        dscores = Tensor._from_core(dscores_core, (BH, S, S), grad_y.dtype)

        # dQ = scale * dScores @ K ; dK = scale * dScores^T @ Q
        dq = bmm(dscores, self._kh, BH, S, dh, S, alpha=self.scale)
        dk = bmm(dscores, self._qh, BH, S, dh, S, trans_a=True, alpha=self.scale)

        dxq = self.wq.backward(self._from_heads(dq, B, S))
        dxk = self.wk.backward(self._from_heads(dk, B, S))
        dxv = self.wv.backward(self._from_heads(dv, B, S))

        return view(dxq + dxk + dxv, (B, S, E))
