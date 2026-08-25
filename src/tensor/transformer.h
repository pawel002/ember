#ifndef TRANSFORMER_OPS_H
#define TRANSFORMER_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Building blocks for transformer / attention models. Like matmul & the axis
 * reductions these are hand-written per backend (CPU in transformer_cpu.c, CUDA
 * in transformer_gpu.cu) and bound into the _tensor module (transformer_api.inc
 * -> tensor_api.c). They deliberately live in the _tensor module so they share
 * its single CUDA stream and caching allocator (a separate module would fork
 * both, breaking op ordering and CUDA-graph capture). */

/* ---- fused softmax ----
 * softmax over one axis, described the same way as sum_axis/max_axis:
 * `outer` positions before the axis, `inner` after it, `axis_dim` along it.
 * A single kernel does the stable max -> exp -> sum -> divide (no temporaries),
 * replacing the 4-op Python composition. Backward: dx = y*(dy - sum(dy*y)). */
void softmax_axis(const float *x, float *out, int outer, int inner, int axis_dim);
void softmax_axis_bwd(const float *dout, const float *y, float *dx, int outer, int inner,
                      int axis_dim);

/* Attention softmax over contiguous rows of length D (rows = batch*heads*queries).
 * The scale (1/sqrt(head_dim)) is folded into the preceding Q@K^T GEMM, so this
 * is a pure softmax. The causal variant masks keys j > (row % sq) to 0, fusing
 * the causal mask into the softmax (no separate mask tensor / extra pass). */
void softmax_rows(const float *x, float *out, int rows, int D);
void softmax_rows_causal(const float *x, float *out, int rows, int D, int sq);

/* ---- LayerNorm over the last dim (length D); N rows ----
 * Forward fuses mean/variance reduction, normalization and the affine
 * (gamma/beta) into one kernel and saves per-row mean & rstd (1/sqrt(var+eps))
 * for the backward pass. Backward computes dx in one fused kernel and
 * accumulates dgamma/dbeta across rows (both zeroed by the wrapper first). */
void layernorm_fwd(const float *x, const float *gamma, const float *beta, float *out, float *mean,
                   float *rstd, int N, int D, float eps);
void layernorm_bwd(const float *dout, const float *x, const float *gamma, const float *mean,
                   const float *rstd, float *dx, float *dgamma, float *dbeta, int N, int D);

/* ---- reshape-to-heads permutation ----
 * Maps a (d0,d1,d2,d3) tensor to (d0,d2,d1,d3) (swap the middle two axes). The
 * multi-head split (B,S,H,dh)->(B,H,S,dh) and its inverse are both this op with
 * d1/d2 swapped, so one kernel serves forward and backward. */
void permute_0213(const float *x, float *out, int d0, int d1, int d2, int d3);

/* ---- batched row-major GEMM ----
 * For each of `batch` independent matrices:
 *     C(n x m) = alpha * opA(A)(n x k) * opB(B)(k x m)
 * A is stored row-major as (transA ? k x n : n x k), B as (transB ? m x k :
 * k x m), C as (n x m). One cuBLAS strided-batched call. This single primitive
 * covers every matmul in attention forward and backward (Q@K^T, P@V, and the
 * four gradient products) with the scale folded into `alpha`. */
void bmm(const float *a, const float *b, float *out, int batch, int n, int m, int k, int transA,
         int transB, float alpha);

/* ---- embedding lookup ----
 * `idx` is a device int32 array of `n_idx` row indices into `weight` (vocab x
 * dim). Forward gathers weight rows; backward scatter-adds `dout` rows into
 * `dweight` (zeroed by the wrapper) via atomics. */
void embedding_fwd(const float *weight, const int *idx, float *out, int n_idx, int dim);
void embedding_bwd(const float *dout, const int *idx, float *dweight, int n_idx, int dim,
                   int vocab);

/* ---- fused ("flash") attention ----
 * out = softmax(scale * Q K^T) V, computed in tiles so the (seq x seq) score
 * matrix never reaches global memory. q/k/v/out hold `batch` = B*heads
 * independent (seq, head_dim) matrices; `nheads` and `sseq` (the stride between
 * consecutive sequence positions) say how they are interleaved:
 *     nheads = 1, sseq = head_dim      -> packed (batch, seq, head_dim)
 *     nheads = H, sseq = H * head_dim  -> (B, seq, H, head_dim) read in place,
 *                                         so no transpose-to-heads copy is
 *                                         needed on either side of attention.
 * `lse` is always packed (batch, sq) and holds the per-row log-sum-exp that
 * backward needs to rebuild the softmax. `causal` masks key j > query i.
 * Backward also needs a (batch, sq) scratch buffer, `rowdot`, for the
 * rowsum(dO * O) term. See flash_gpu.cu.
 *
 * The CUDA kernels are specialized per head_dim; attention_supported() says
 * whether this head_dim has one (callers fall back to the composed
 * bmm + softmax + bmm path otherwise). The CPU backend supports every size. */
int attention_supported(int dh);
void attention_fwd(const float *q, const float *k, const float *v, float *out, float *lse,
                   int batch, int sq, int sk, int dh, float scale, int causal, int nheads,
                   int sseq);
void attention_bwd(const float *dout, const float *q, const float *k, const float *v,
                   const float *out, const float *lse, float *rowdot, float *dq, float *dk,
                   float *dv, int batch, int sq, int sk, int dh, float scale, int causal,
                   int nheads, int sseq);

#ifdef __cplusplus
}
#endif

#endif
