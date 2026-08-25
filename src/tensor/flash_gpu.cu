// Fused ("flash") attention: softmax(scale * Q K^T) V without ever writing the
// S x S score matrix to memory.
//
// The composed path (batched GEMM -> fused softmax -> batched GEMM, plus four
// more GEMMs in backward) is correct but moves the score matrix through global
// memory eight times. For the tiny-gpt shapes (batch*heads = 512, S = 256) that
// matrix is 134 MB, far past the L2, and adds up to ~1.5 GB of traffic per
// transformer block per step -- which is what made attention the single most
// expensive part of the step.
//
// Instead each block keeps a tile of queries resident, streams the key/value
// tiles past it, and carries the softmax as a running (max, sum) pair, so the
// scores live and die in shared memory. Backward recomputes them the same way
// from the saved log-sum-exp, in two passes: one accumulating dK/dV per key
// tile, one accumulating dQ per query tile. Recomputing S costs ~40% extra
// FLOPs but removes all of the score traffic, and attention here is firmly
// bandwidth-bound.
//
// Layout: q/k/v/out are (batch, seq, head_dim) row-major with batch = B*heads;
// lse is (batch, seq). `causal` masks key j > query i (decoder masking).

#include <cuda_runtime.h>
#include <math.h>

#include "../core/memory.h"
#include "../core/utils_gpu.cuh"
#include "transformer.h"

// 256 threads viewed as a 16x16 grid: tx indexes the key/head_dim direction,
// ty the query direction. Every tile extent must be a multiple of 16.
#define FA_TX 16
#define FA_TY 16
#define FA_NT (FA_TX * FA_TY)

#define FA_POST_LAUNCH() GPU_ERR_CHK(cudaGetLastError())

// Reduce across the 16 lanes that share a `ty` (they are 16 consecutive lanes
// of one warp), leaving the result in all of them.
__device__ __forceinline__ float fa_row_max16(float v)
{
#pragma unroll
    for (int k = 1; k < FA_TX; k <<= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, k));
    return v;
}

__device__ __forceinline__ float fa_row_sum16(float v)
{
#pragma unroll
    for (int k = 1; k < FA_TX; k <<= 1) v += __shfl_xor_sync(0xffffffffu, v, k);
    return v;
}

/* ===================== forward ===================== */

// One block per (query tile, batch). BR queries stay resident; the key/value
// tiles stream past in chunks of BC.
template <int DH, int BR, int BC>
__global__ void k_flash_fwd(const float *__restrict__ Q, const float *__restrict__ K,
                            const float *__restrict__ V, float *__restrict__ O,
                            float *__restrict__ LSE, int sq, int sk, float scale, int causal)
{
    constexpr int RPT = BR / FA_TY;  // query rows per thread
    constexpr int CPT = BC / FA_TX;  // key columns per thread
    constexpr int DPT = DH / FA_TX;  // head_dim components per thread
    constexpr int KS = BC + 1;       // padded stride for the transposed K tile
    constexpr int SS = BC + 1;       // padded stride for the score tile

    extern __shared__ float smem[];
    float *sQ = smem;           // BR x DH
    float *sKt = sQ + BR * DH;  // DH x KS  (transposed: [d][c])
    float *sV = sKt + DH * KS;  // BC x DH
    float *sS = sV + BC * DH;   // BR x SS

    const int tid = threadIdx.x;
    const int tx = tid % FA_TX;
    const int ty = tid / FA_TX;
    const int q0 = blockIdx.x * BR;
    const size_t bat = blockIdx.y;

    const float *Qb = Q + bat * (size_t)sq * DH;
    const float *Kb = K + bat * (size_t)sk * DH;
    const float *Vb = V + bat * (size_t)sk * DH;
    float *Ob = O + bat * (size_t)sq * DH;
    float *LSEb = LSE + bat * (size_t)sq;

    for (int i = tid; i < BR * DH; i += FA_NT) {
        int r = i / DH, d = i - r * DH;
        sQ[i] = (q0 + r < sq) ? Qb[(size_t)(q0 + r) * DH + d] : 0.0f;
    }

    float acc[RPT][DPT];
    float run_m[RPT], run_l[RPT];
#pragma unroll
    for (int p = 0; p < RPT; ++p) {
        run_m[p] = -INFINITY;
        run_l[p] = 0.0f;
#pragma unroll
        for (int e = 0; e < DPT; ++e) acc[p][e] = 0.0f;
    }

    // Under causal masking no query in this tile can see past q0 + BR - 1.
    const int kmax = causal ? min(sk, q0 + BR) : sk;

    for (int j0 = 0; j0 < kmax; j0 += BC) {
        __syncthreads();
        for (int i = tid; i < BC * DH; i += FA_NT) {
            int c = i / DH, d = i - c * DH;
            int j = j0 + c;
            float kv = (j < sk) ? Kb[(size_t)j * DH + d] : 0.0f;
            sKt[d * KS + c] = kv;
            sV[i] = (j < sk) ? Vb[(size_t)j * DH + d] : 0.0f;
        }
        __syncthreads();

        // S tile: a 4x4-ish register tile per thread, so the shared-memory
        // traffic is RPT + CPT loads per RPT*CPT fused multiply-adds.
        float s[RPT][CPT];
#pragma unroll
        for (int p = 0; p < RPT; ++p)
#pragma unroll
            for (int b = 0; b < CPT; ++b) s[p][b] = 0.0f;

        for (int d = 0; d < DH; ++d) {
            float qa[RPT], kb[CPT];
#pragma unroll
            for (int p = 0; p < RPT; ++p) qa[p] = sQ[(ty + p * FA_TY) * DH + d];
#pragma unroll
            for (int b = 0; b < CPT; ++b) kb[b] = sKt[d * KS + tx + b * FA_TX];
#pragma unroll
            for (int p = 0; p < RPT; ++p)
#pragma unroll
                for (int b = 0; b < CPT; ++b) s[p][b] += qa[p] * kb[b];
        }

#pragma unroll
        for (int p = 0; p < RPT; ++p) {
            int qi = q0 + ty + p * FA_TY;
#pragma unroll
            for (int b = 0; b < CPT; ++b) {
                int kj = j0 + tx + b * FA_TX;
                bool ok = qi < sq && kj < sk && (!causal || kj <= qi);
                s[p][b] = ok ? s[p][b] * scale : -INFINITY;
            }
        }

        // Online softmax: rescale what we have by exp(old_max - new_max) and
        // fold in this tile. All 16 lanes of a row hold identical m/l, so the
        // branch below is uniform within the reduction group.
#pragma unroll
        for (int p = 0; p < RPT; ++p) {
            float local = -INFINITY;
#pragma unroll
            for (int b = 0; b < CPT; ++b) local = fmaxf(local, s[p][b]);
            float mnew = fmaxf(run_m[p], fa_row_max16(local));

            float corr, lsum = 0.0f;
            int row = (ty + p * FA_TY) * SS;
            if (mnew == -INFINITY) {
                // Every key this row can see is still masked out.
                corr = 1.0f;
#pragma unroll
                for (int b = 0; b < CPT; ++b) sS[row + tx + b * FA_TX] = 0.0f;
            } else {
                corr = __expf(run_m[p] - mnew);
#pragma unroll
                for (int b = 0; b < CPT; ++b) {
                    float pv = __expf(s[p][b] - mnew);
                    sS[row + tx + b * FA_TX] = pv;
                    lsum += pv;
                }
            }
            run_m[p] = mnew;
            run_l[p] = run_l[p] * corr + fa_row_sum16(lsum);
#pragma unroll
            for (int e = 0; e < DPT; ++e) acc[p][e] *= corr;
        }
        __syncthreads();

        // O += P V
        for (int c = 0; c < BC; ++c) {
            float pa[RPT], vb[DPT];
#pragma unroll
            for (int p = 0; p < RPT; ++p) pa[p] = sS[(ty + p * FA_TY) * SS + c];
#pragma unroll
            for (int e = 0; e < DPT; ++e) vb[e] = sV[c * DH + tx + e * FA_TX];
#pragma unroll
            for (int p = 0; p < RPT; ++p)
#pragma unroll
                for (int e = 0; e < DPT; ++e) acc[p][e] += pa[p] * vb[e];
        }
    }

#pragma unroll
    for (int p = 0; p < RPT; ++p) {
        int r = q0 + ty + p * FA_TY;
        if (r >= sq) continue;
        float inv = run_l[p] > 0.0f ? 1.0f / run_l[p] : 0.0f;
#pragma unroll
        for (int e = 0; e < DPT; ++e) Ob[(size_t)r * DH + tx + e * FA_TX] = acc[p][e] * inv;
        if (tx == 0) LSEb[r] = run_l[p] > 0.0f ? (run_m[p] + __logf(run_l[p])) : -INFINITY;
    }
}

/* ===================== backward ===================== */

// D[r] = sum_d dO[r,d] * O[r,d] -- the term that turns dP into dS. One warp per
// row; four rows per block.
__global__ void k_flash_rowdot(const float *__restrict__ dO, const float *__restrict__ O,
                               float *__restrict__ D, int rows, int dh)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= rows) return;

    const float *a = dO + (size_t)warp * dh;
    const float *b = O + (size_t)warp * dh;
    float s = 0.0f;
    for (int d = lane; d < dh; d += 32) s += a[d] * b[d];
#pragma unroll
    for (int k = 16; k > 0; k >>= 1) s += __shfl_xor_sync(0xffffffffu, s, k);
    if (lane == 0) D[warp] = s;
}

// dK / dV: one block per (key tile, batch); query tiles stream past. Each block
// owns the whole dK_j / dV_j accumulation, so no atomics are needed.
template <int DH, int BR, int BC>
__global__ void k_flash_bwd_kv(const float *__restrict__ dO, const float *__restrict__ Q,
                               const float *__restrict__ K, const float *__restrict__ V,
                               const float *__restrict__ LSE, const float *__restrict__ D,
                               float *__restrict__ dK, float *__restrict__ dV, int sq, int sk,
                               float scale, int causal)
{
    constexpr int RPT = BR / FA_TY;  // query rows per thread (score tile)
    constexpr int CPT = BC / FA_TX;  // key columns per thread (score tile)
    constexpr int APT = BC / FA_TY;  // key rows per thread (dK/dV accumulators)
    constexpr int DPT = DH / FA_TX;  // head_dim components per thread
    constexpr int KS = BC + 1;
    constexpr int SS = BC + 1;

    extern __shared__ float smem[];
    float *sQ = smem;            // BR x DH
    float *sdO = sQ + BR * DH;   // BR x DH
    float *sKt = sdO + BR * DH;  // DH x KS
    float *sVt = sKt + DH * KS;  // DH x KS
    float *sP = sVt + DH * KS;   // BR x SS   (reused for dS)
    float *sLSE = sP + BR * SS;  // BR
    float *sD = sLSE + BR;       // BR

    const int tid = threadIdx.x;
    const int tx = tid % FA_TX;
    const int ty = tid / FA_TX;
    const int j0 = blockIdx.x * BC;
    const size_t bat = blockIdx.y;

    const float *Qb = Q + bat * (size_t)sq * DH;
    const float *dOb = dO + bat * (size_t)sq * DH;
    const float *Kb = K + bat * (size_t)sk * DH;
    const float *Vb = V + bat * (size_t)sk * DH;
    const float *LSEb = LSE + bat * (size_t)sq;
    const float *Db = D + bat * (size_t)sq;
    float *dKb = dK + bat * (size_t)sk * DH;
    float *dVb = dV + bat * (size_t)sk * DH;

    for (int i = tid; i < BC * DH; i += FA_NT) {
        int c = i / DH, d = i - c * DH;
        int j = j0 + c;
        sKt[d * KS + c] = (j < sk) ? Kb[(size_t)j * DH + d] : 0.0f;
        sVt[d * KS + c] = (j < sk) ? Vb[(size_t)j * DH + d] : 0.0f;
    }

    float accK[APT][DPT], accV[APT][DPT];
#pragma unroll
    for (int a = 0; a < APT; ++a)
#pragma unroll
        for (int e = 0; e < DPT; ++e) accK[a][e] = accV[a][e] = 0.0f;

    // Only query tiles that can attend to this key tile contribute.
    int i0start = 0;
    if (causal) {
        int t = j0 - BR + 1;
        i0start = t <= 0 ? 0 : ((t + BR - 1) / BR) * BR;
    }

    for (int i0 = i0start; i0 < sq; i0 += BR) {
        __syncthreads();
        for (int i = tid; i < BR * DH; i += FA_NT) {
            int r = i / DH, d = i - r * DH;
            int qi = i0 + r;
            sQ[i] = (qi < sq) ? Qb[(size_t)qi * DH + d] : 0.0f;
            sdO[i] = (qi < sq) ? dOb[(size_t)qi * DH + d] : 0.0f;
        }
        for (int r = tid; r < BR; r += FA_NT) {
            int qi = i0 + r;
            sLSE[r] = (qi < sq) ? LSEb[qi] : -INFINITY;
            sD[r] = (qi < sq) ? Db[qi] : 0.0f;
        }
        __syncthreads();

        // Recompute S = scale * Q K^T and P = exp(S - LSE), and at the same
        // time dP = dO V^T (identical shape and thread mapping).
        float s[RPT][CPT], dp[RPT][CPT];
#pragma unroll
        for (int p = 0; p < RPT; ++p)
#pragma unroll
            for (int b = 0; b < CPT; ++b) s[p][b] = dp[p][b] = 0.0f;

        for (int d = 0; d < DH; ++d) {
            float qa[RPT], ga[RPT], kb[CPT], vb[CPT];
#pragma unroll
            for (int p = 0; p < RPT; ++p) {
                qa[p] = sQ[(ty + p * FA_TY) * DH + d];
                ga[p] = sdO[(ty + p * FA_TY) * DH + d];
            }
#pragma unroll
            for (int b = 0; b < CPT; ++b) {
                kb[b] = sKt[d * KS + tx + b * FA_TX];
                vb[b] = sVt[d * KS + tx + b * FA_TX];
            }
#pragma unroll
            for (int p = 0; p < RPT; ++p)
#pragma unroll
                for (int b = 0; b < CPT; ++b) {
                    s[p][b] += qa[p] * kb[b];
                    dp[p][b] += ga[p] * vb[b];
                }
        }

        float pval[RPT][CPT];
#pragma unroll
        for (int p = 0; p < RPT; ++p) {
            int rr = ty + p * FA_TY;
            int qi = i0 + rr;
            float lse = sLSE[rr];
#pragma unroll
            for (int b = 0; b < CPT; ++b) {
                int kj = j0 + tx + b * FA_TX;
                bool ok = qi < sq && kj < sk && (!causal || kj <= qi) && lse != -INFINITY;
                pval[p][b] = ok ? __expf(s[p][b] * scale - lse) : 0.0f;
                sP[rr * SS + tx + b * FA_TX] = pval[p][b];
            }
        }
        __syncthreads();

        // dV += P^T dO
        for (int r = 0; r < BR; ++r) {
            float pa[APT], gb[DPT];
#pragma unroll
            for (int a = 0; a < APT; ++a) pa[a] = sP[r * SS + ty + a * FA_TY];
#pragma unroll
            for (int e = 0; e < DPT; ++e) gb[e] = sdO[r * DH + tx + e * FA_TX];
#pragma unroll
            for (int a = 0; a < APT; ++a)
#pragma unroll
                for (int e = 0; e < DPT; ++e) accV[a][e] += pa[a] * gb[e];
        }
        __syncthreads();

        // dS = P * (dP - D), reusing the sP tile.
#pragma unroll
        for (int p = 0; p < RPT; ++p) {
            int rr = ty + p * FA_TY;
            float drow = sD[rr];
#pragma unroll
            for (int b = 0; b < CPT; ++b)
                sP[rr * SS + tx + b * FA_TX] = pval[p][b] * (dp[p][b] - drow);
        }
        __syncthreads();

        // dK += scale * dS^T Q
        for (int r = 0; r < BR; ++r) {
            float da[APT], qb[DPT];
#pragma unroll
            for (int a = 0; a < APT; ++a) da[a] = sP[r * SS + ty + a * FA_TY];
#pragma unroll
            for (int e = 0; e < DPT; ++e) qb[e] = sQ[r * DH + tx + e * FA_TX];
#pragma unroll
            for (int a = 0; a < APT; ++a)
#pragma unroll
                for (int e = 0; e < DPT; ++e) accK[a][e] += da[a] * qb[e];
        }
    }

#pragma unroll
    for (int a = 0; a < APT; ++a) {
        int j = j0 + ty + a * FA_TY;
        if (j >= sk) continue;
#pragma unroll
        for (int e = 0; e < DPT; ++e) {
            size_t off = (size_t)j * DH + tx + e * FA_TX;
            dKb[off] = accK[a][e] * scale;
            dVb[off] = accV[a][e];
        }
    }
}

// dQ: one block per (query tile, batch); key tiles stream past, mirroring the
// forward loop. Recomputes S/P a second time, which is cheaper than the atomics
// (or the extra pass over global memory) a single fused kernel would need.
template <int DH, int BR, int BC>
__global__ void k_flash_bwd_q(const float *__restrict__ dO, const float *__restrict__ Q,
                              const float *__restrict__ K, const float *__restrict__ V,
                              const float *__restrict__ LSE, const float *__restrict__ D,
                              float *__restrict__ dQ, int sq, int sk, float scale, int causal)
{
    constexpr int RPT = BR / FA_TY;
    constexpr int CPT = BC / FA_TX;
    constexpr int DPT = DH / FA_TX;
    constexpr int KS = BC + 1;
    constexpr int SS = BC + 1;

    extern __shared__ float smem[];
    float *sQ = smem;             // BR x DH
    float *sdO = sQ + BR * DH;    // BR x DH
    float *sKt = sdO + BR * DH;   // DH x KS  (serves both the S and dQ products)
    float *sVt = sKt + DH * KS;   // DH x KS  (for the dP product)
    float *sdS = sVt + DH * KS;   // BR x SS
    float *sLSE = sdS + BR * SS;  // BR
    float *sD = sLSE + BR;        // BR

    const int tid = threadIdx.x;
    const int tx = tid % FA_TX;
    const int ty = tid / FA_TX;
    const int q0 = blockIdx.x * BR;
    const size_t bat = blockIdx.y;

    const float *Qb = Q + bat * (size_t)sq * DH;
    const float *dOb = dO + bat * (size_t)sq * DH;
    const float *Kb = K + bat * (size_t)sk * DH;
    const float *Vb = V + bat * (size_t)sk * DH;
    const float *LSEb = LSE + bat * (size_t)sq;
    const float *Db = D + bat * (size_t)sq;
    float *dQb = dQ + bat * (size_t)sq * DH;

    for (int i = tid; i < BR * DH; i += FA_NT) {
        int r = i / DH, d = i - r * DH;
        int qi = q0 + r;
        sQ[i] = (qi < sq) ? Qb[(size_t)qi * DH + d] : 0.0f;
        sdO[i] = (qi < sq) ? dOb[(size_t)qi * DH + d] : 0.0f;
    }
    for (int r = tid; r < BR; r += FA_NT) {
        int qi = q0 + r;
        sLSE[r] = (qi < sq) ? LSEb[qi] : -INFINITY;
        sD[r] = (qi < sq) ? Db[qi] : 0.0f;
    }

    float acc[RPT][DPT];
#pragma unroll
    for (int p = 0; p < RPT; ++p)
#pragma unroll
        for (int e = 0; e < DPT; ++e) acc[p][e] = 0.0f;

    const int kmax = causal ? min(sk, q0 + BR) : sk;

    for (int j0 = 0; j0 < kmax; j0 += BC) {
        __syncthreads();
        for (int i = tid; i < BC * DH; i += FA_NT) {
            int c = i / DH, d = i - c * DH;
            int j = j0 + c;
            sKt[d * KS + c] = (j < sk) ? Kb[(size_t)j * DH + d] : 0.0f;
            sVt[d * KS + c] = (j < sk) ? Vb[(size_t)j * DH + d] : 0.0f;
        }
        __syncthreads();

        float s[RPT][CPT], dp[RPT][CPT];
#pragma unroll
        for (int p = 0; p < RPT; ++p)
#pragma unroll
            for (int b = 0; b < CPT; ++b) s[p][b] = dp[p][b] = 0.0f;

        for (int d = 0; d < DH; ++d) {
            float qa[RPT], ga[RPT], kb[CPT], vb[CPT];
#pragma unroll
            for (int p = 0; p < RPT; ++p) {
                qa[p] = sQ[(ty + p * FA_TY) * DH + d];
                ga[p] = sdO[(ty + p * FA_TY) * DH + d];
            }
#pragma unroll
            for (int b = 0; b < CPT; ++b) {
                kb[b] = sKt[d * KS + tx + b * FA_TX];
                vb[b] = sVt[d * KS + tx + b * FA_TX];
            }
#pragma unroll
            for (int p = 0; p < RPT; ++p)
#pragma unroll
                for (int b = 0; b < CPT; ++b) {
                    s[p][b] += qa[p] * kb[b];
                    dp[p][b] += ga[p] * vb[b];
                }
        }

#pragma unroll
        for (int p = 0; p < RPT; ++p) {
            int rr = ty + p * FA_TY;
            int qi = q0 + rr;
            float lse = sLSE[rr], drow = sD[rr];
#pragma unroll
            for (int b = 0; b < CPT; ++b) {
                int kj = j0 + tx + b * FA_TX;
                bool ok = qi < sq && kj < sk && (!causal || kj <= qi) && lse != -INFINITY;
                float pv = ok ? __expf(s[p][b] * scale - lse) : 0.0f;
                sdS[rr * SS + tx + b * FA_TX] = pv * (dp[p][b] - drow);
            }
        }
        __syncthreads();

        // dQ += dS K
        for (int c = 0; c < BC; ++c) {
            float da[RPT], kb[DPT];
#pragma unroll
            for (int p = 0; p < RPT; ++p) da[p] = sdS[(ty + p * FA_TY) * SS + c];
            // K[c][d] out of the transposed tile: the padded KS stride keeps
            // the 16 lanes on distinct banks here too.
#pragma unroll
            for (int e = 0; e < DPT; ++e) kb[e] = sKt[(tx + e * FA_TX) * KS + c];
#pragma unroll
            for (int p = 0; p < RPT; ++p)
#pragma unroll
                for (int e = 0; e < DPT; ++e) acc[p][e] += da[p] * kb[e];
        }
    }

#pragma unroll
    for (int p = 0; p < RPT; ++p) {
        int r = q0 + ty + p * FA_TY;
        if (r >= sq) continue;
#pragma unroll
        for (int e = 0; e < DPT; ++e) dQb[(size_t)r * DH + tx + e * FA_TX] = acc[p][e] * scale;
    }
}

/* ===================== launch ===================== */

// Tiles are sized so the working set fits shared memory. Everything but
// head_dim 128 uses 64-wide key tiles; 128 halves them to stay under the
// 99 KB per-block limit.
template <int DH>
struct fa_cfg {
    static constexpr int BR_FWD = 64;
    // A 64-row backward tile halves the shared-memory loads per FMA in the
    // score/dP product, but the tiles it needs resident grow with head_dim --
    // past 32 it no longer fits two blocks per SM, so drop back to 32 there.
    static constexpr int BR_BWD = (DH <= 32) ? 64 : 32;
    static constexpr int BC = (DH >= 128) ? 32 : 64;
};

static bool fa_raise_shared(const void *fn, size_t bytes)
{
    // Anything past 48 KB per block has to be opted into explicitly.
    if (bytes <= 48 * 1024) return true;
    cudaError_t e = cudaFuncSetAttribute((const void *)fn,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

template <int DH>
static void fa_fwd_launch(const float *q, const float *k, const float *v, float *out, float *lse,
                          int batch, int sq, int sk, float scale, int causal)
{
    constexpr int BR = fa_cfg<DH>::BR_FWD;
    constexpr int BC = fa_cfg<DH>::BC;
    size_t bytes = (size_t)(BR * DH + DH * (BC + 1) + BC * DH + BR * (BC + 1)) * sizeof(float);
    auto fn = k_flash_fwd<DH, BR, BC>;
    fa_raise_shared((const void *)fn, bytes);
    dim3 grid((sq + BR - 1) / BR, batch);
    fn<<<grid, FA_NT, bytes, ember_stream()>>>(q, k, v, out, lse, sq, sk, scale, causal);
    FA_POST_LAUNCH();
}

template <int DH>
static void fa_bwd_launch(const float *dout, const float *q, const float *k, const float *v,
                          const float *lse, const float *rowdot, float *dq, float *dk, float *dv,
                          int batch, int sq, int sk, float scale, int causal)
{
    constexpr int BR = fa_cfg<DH>::BR_BWD;
    constexpr int BC = fa_cfg<DH>::BC;

    {
        size_t bytes =
            (size_t)(2 * BR * DH + 2 * DH * (BC + 1) + BR * (BC + 1) + 2 * BR) * sizeof(float);
        auto fn = k_flash_bwd_kv<DH, BR, BC>;
        fa_raise_shared((const void *)fn, bytes);
        dim3 grid((sk + BC - 1) / BC, batch);
        fn<<<grid, FA_NT, bytes, ember_stream()>>>(dout, q, k, v, lse, rowdot, dk, dv, sq, sk,
                                                   scale, causal);
        FA_POST_LAUNCH();
    }
    {
        size_t bytes =
            (size_t)(2 * BR * DH + 2 * DH * (BC + 1) + BR * (BC + 1) + 2 * BR) * sizeof(float);
        auto fn = k_flash_bwd_q<DH, BR, BC>;
        fa_raise_shared((const void *)fn, bytes);
        dim3 grid((sq + BR - 1) / BR, batch);
        fn<<<grid, FA_NT, bytes, ember_stream()>>>(dout, q, k, v, lse, rowdot, dq, sq, sk, scale,
                                                   causal);
        FA_POST_LAUNCH();
    }
}

extern "C" {

int attention_supported(int dh)
{
    return dh == 16 || dh == 32 || dh == 64 || dh == 128;
}

void attention_fwd(const float *q, const float *k, const float *v, float *out, float *lse,
                   int batch, int sq, int sk, int dh, float scale, int causal)
{
    switch (dh) {
        case 16:
            fa_fwd_launch<16>(q, k, v, out, lse, batch, sq, sk, scale, causal);
            break;
        case 32:
            fa_fwd_launch<32>(q, k, v, out, lse, batch, sq, sk, scale, causal);
            break;
        case 64:
            fa_fwd_launch<64>(q, k, v, out, lse, batch, sq, sk, scale, causal);
            break;
        case 128:
            fa_fwd_launch<128>(q, k, v, out, lse, batch, sq, sk, scale, causal);
            break;
        default:
            break;  // caller must check attention_supported()
    }
}

void attention_bwd(const float *dout, const float *q, const float *k, const float *v,
                   const float *out, const float *lse, float *rowdot, float *dq, float *dk,
                   float *dv, int batch, int sq, int sk, int dh, float scale, int causal)
{
    int rows = batch * sq;
    // 4 warps per block, one row each.
    k_flash_rowdot<<<(rows + 3) / 4, 128, 0, ember_stream()>>>(dout, out, rowdot, rows, dh);
    FA_POST_LAUNCH();

    switch (dh) {
        case 16:
            fa_bwd_launch<16>(dout, q, k, v, lse, rowdot, dq, dk, dv, batch, sq, sk, scale, causal);
            break;
        case 32:
            fa_bwd_launch<32>(dout, q, k, v, lse, rowdot, dq, dk, dv, batch, sq, sk, scale, causal);
            break;
        case 64:
            fa_bwd_launch<64>(dout, q, k, v, lse, rowdot, dq, dk, dv, batch, sq, sk, scale, causal);
            break;
        case 128:
            fa_bwd_launch<128>(dout, q, k, v, lse, rowdot, dq, dk, dv, batch, sq, sk, scale, causal);
            break;
        default:
            break;
    }
}
}
