"""Thin Tensor-level wrappers around the transformer backend bindings.

These keep the layer code readable: shape-metadata views (no copy), the batched
GEMM primitive, and the heads permutation, all returning ``Tensor`` objects.
"""

from __future__ import annotations

from ember._core import _bmm, _permute_0213
from ember.tensor import Tensor


def view(t: Tensor, shape: tuple[int, ...]) -> Tensor:
    """A reshaped view sharing ``t``'s device buffer (contiguous reshape only)."""
    return Tensor._from_core(t._core, shape, t.dtype)


def bmm(
    a: Tensor,
    b: Tensor,
    batch: int,
    n: int,
    m: int,
    k: int,
    trans_a: bool = False,
    trans_b: bool = False,
    alpha: float = 1.0,
) -> Tensor:
    """Batched row-major GEMM: ``C = alpha * opA(a) @ opB(b)`` per batch,
    returning a ``(batch, n, m)`` tensor. See the ``bmm`` backend op."""
    core = _bmm(
        a._core, b._core, batch, n, m, k, int(trans_a), int(trans_b), float(alpha)
    )
    return Tensor._from_core(core, (batch, n, m), a.dtype)


def mm(
    a: Tensor,
    b: Tensor,
    n: int,
    m: int,
    k: int,
    trans_a: bool = False,
    trans_b: bool = False,
    alpha: float = 1.0,
) -> Tensor:
    """Single row-major GEMM ``C = alpha * opA(a) @ opB(b)`` -> ``(n, m)``.

    The transposes are cuBLAS ``OP_T`` flags, so ``a``/``b`` are read in place:
    no transposed copy is materialized and no temporary is allocated for it.
    """
    core = _bmm(a._core, b._core, 1, n, m, k, int(trans_a), int(trans_b), float(alpha))
    return Tensor._from_core(core, (n, m), a.dtype)


def permute_0213(t: Tensor, d0: int, d1: int, d2: int, d3: int) -> Tensor:
    """(d0,d1,d2,d3) -> (d0,d2,d1,d3); swaps the middle two axes."""
    core = _permute_0213(t._core, d0, d1, d2, d3)
    return Tensor._from_core(core, (d0, d2, d1, d3), t.dtype)
