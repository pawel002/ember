#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>

#include "../core/memory.h"
#include "../core/utils_gpu.cuh"
#include "transformer.h"

#define TBLOCK 256  // threads per block for the block-per-row kernels (power of 2)

static int grid1(int n)
{
    return (n + TBLOCK - 1) / TBLOCK;
}

#define CUDA_POST_LAUNCH() GPU_ERR_CHK(cudaGetLastError())

// Own cuBLAS handle (the one in operators_gpu.cu is file-local). Created lazily,
// reused for the process lifetime, always bound to the ember stream.
static cublasHandle_t tf_cublas(void)
{
    static cublasHandle_t handle = NULL;
    if (!handle) CUBLAS_ERR_CHK(cublasCreate(&handle));
    return handle;
}

/* ---- block reductions (blockDim must be a power of two) ---- */
__device__ __forceinline__ float block_reduce_sum(float v, float *sh)
{
    int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_reduce_max(float v, float *sh)
{
    int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}

/* ================= softmax ================= */

// Generic, stable softmax over one strided axis (outer * inner output rows,
// axis_dim elements each). One thread per (outer,inner) position; fuses
// max -> exp -> sum -> divide.
__global__ void k_softmax_axis(const float *x, float *out, int outer, int inner, int axis_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total) return;

    int o = idx / inner;
    int i = idx % inner;
    int base = o * (axis_dim * inner) + i;

    float m = x[base];
    for (int r = 1; r < axis_dim; r++) m = fmaxf(m, x[base + r * inner]);

    float s = 0.0f;
    for (int r = 0; r < axis_dim; r++) {
        float e = expf(x[base + r * inner] - m);
        out[base + r * inner] = e;
        s += e;
    }
    float inv = 1.0f / s;
    for (int r = 0; r < axis_dim; r++) out[base + r * inner] *= inv;
}

extern "C" void softmax_axis(const float *x, float *out, int outer, int inner, int axis_dim)
{
    int total = outer * inner;
    k_softmax_axis<<<grid1(total), TBLOCK, 0, ember_stream()>>>(x, out, outer, inner, axis_dim);
    CUDA_POST_LAUNCH();
}

// Softmax backward: dx = y * (dout - sum_axis(dout * y)).
__global__ void k_softmax_axis_bwd(const float *dout, const float *y, float *dx, int outer,
                                   int inner, int axis_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total) return;

    int o = idx / inner;
    int i = idx % inner;
    int base = o * (axis_dim * inner) + i;

    float dot = 0.0f;
    for (int r = 0; r < axis_dim; r++) dot += dout[base + r * inner] * y[base + r * inner];
    for (int r = 0; r < axis_dim; r++) {
        int p = base + r * inner;
        dx[p] = y[p] * (dout[p] - dot);
    }
}

// Block-per-row softmax backward for contiguous rows (inner == 1, the common
// case): the dot product over the row is a block reduction instead of a
// per-thread serial loop, and the write pass is coalesced.
__global__ void k_softmax_rows_bwd(const float *dout, const float *y, float *dx, int rows, int D)
{
    extern __shared__ float sh[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *dr = dout + (size_t)row * D;
    const float *yr = y + (size_t)row * D;
    float *dxr = dx + (size_t)row * D;

    float part = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) part += dr[j] * yr[j];
    float dot = block_reduce_sum(part, sh);

    for (int j = threadIdx.x; j < D; j += blockDim.x) dxr[j] = yr[j] * (dr[j] - dot);
}

extern "C" void softmax_axis_bwd(const float *dout, const float *y, float *dx, int outer, int inner,
                                 int axis_dim)
{
    if (inner == 1) {
        k_softmax_rows_bwd<<<outer, TBLOCK, TBLOCK * sizeof(float), ember_stream()>>>(
            dout, y, dx, outer, axis_dim);
        CUDA_POST_LAUNCH();
        return;
    }
    int total = outer * inner;
    k_softmax_axis_bwd<<<grid1(total), TBLOCK, 0, ember_stream()>>>(dout, y, dx, outer, inner,
                                                                    axis_dim);
    CUDA_POST_LAUNCH();
}

// Attention softmax: one block per contiguous row of length D. `causal` masks
// keys j > (row % sq) to exactly 0 (fused mask, no separate tensor). Non-causal
// callers pass causal=0.
__global__ void k_softmax_rows(const float *x, float *out, int rows, int D, int causal, int sq)
{
    extern __shared__ float sh[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *xr = x + (size_t)row * D;
    float *outr = out + (size_t)row * D;
    int limit = causal ? (row % sq) + 1 : D;  // number of unmasked keys (>=1)

    float lm = -INFINITY;
    for (int j = threadIdx.x; j < limit; j += blockDim.x) lm = fmaxf(lm, xr[j]);
    float m = block_reduce_max(lm, sh);

    float ls = 0.0f;
    for (int j = threadIdx.x; j < limit; j += blockDim.x) ls += expf(xr[j] - m);
    float inv = 1.0f / block_reduce_sum(ls, sh);

    for (int j = threadIdx.x; j < D; j += blockDim.x)
        outr[j] = (j < limit) ? expf(xr[j] - m) * inv : 0.0f;
}

extern "C" void softmax_rows(const float *x, float *out, int rows, int D)
{
    k_softmax_rows<<<rows, TBLOCK, TBLOCK * sizeof(float), ember_stream()>>>(x, out, rows, D, 0, 0);
    CUDA_POST_LAUNCH();
}

extern "C" void softmax_rows_causal(const float *x, float *out, int rows, int D, int sq)
{
    k_softmax_rows<<<rows, TBLOCK, TBLOCK * sizeof(float), ember_stream()>>>(x, out, rows, D, 1,
                                                                             sq);
    CUDA_POST_LAUNCH();
}

/* ================= LayerNorm ================= */

__global__ void k_layernorm_fwd(const float *x, const float *gamma, const float *beta, float *out,
                                float *mean, float *rstd, int N, int D, float eps)
{
    extern __shared__ float sh[];
    int row = blockIdx.x;
    if (row >= N) return;

    const float *xr = x + (size_t)row * D;
    float *outr = out + (size_t)row * D;

    float s = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) s += xr[j];
    float mu = block_reduce_sum(s, sh) / D;

    float vs = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float d = xr[j] - mu;
        vs += d * d;
    }
    float var = block_reduce_sum(vs, sh) / D;
    float rs = rsqrtf(var + eps);

    if (threadIdx.x == 0) {
        mean[row] = mu;
        rstd[row] = rs;
    }
    for (int j = threadIdx.x; j < D; j += blockDim.x)
        outr[j] = (xr[j] - mu) * rs * gamma[j] + beta[j];
}

extern "C" void layernorm_fwd(const float *x, const float *gamma, const float *beta, float *out,
                              float *mean, float *rstd, int N, int D, float eps)
{
    k_layernorm_fwd<<<N, TBLOCK, TBLOCK * sizeof(float), ember_stream()>>>(x, gamma, beta, out,
                                                                           mean, rstd, N, D, eps);
    CUDA_POST_LAUNCH();
}

// dx (fused per row) + atomic accumulation of dgamma/dbeta across rows.
__global__ void k_layernorm_bwd(const float *dout, const float *x, const float *gamma,
                                const float *mean, const float *rstd, float *dx, float *dgamma,
                                float *dbeta, int N, int D)
{
    extern __shared__ float sh[];
    int row = blockIdx.x;
    if (row >= N) return;

    const float *dor = dout + (size_t)row * D;
    const float *xr = x + (size_t)row * D;
    float *dxr = dx + (size_t)row * D;
    float mu = mean[row], rs = rstd[row];

    float s1 = 0.0f, s2 = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float xhat = (xr[j] - mu) * rs;
        float dxh = dor[j] * gamma[j];
        s1 += dxh;
        s2 += dxh * xhat;
    }
    float c1 = block_reduce_sum(s1, sh) / D;
    float c2 = block_reduce_sum(s2, sh) / D;

    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float xhat = (xr[j] - mu) * rs;
        float dxh = dor[j] * gamma[j];
        dxr[j] = rs * (dxh - c1 - xhat * c2);
        atomicAdd(&dgamma[j], dor[j] * xhat);
        atomicAdd(&dbeta[j], dor[j]);
    }
}

extern "C" void layernorm_bwd(const float *dout, const float *x, const float *gamma,
                              const float *mean, const float *rstd, float *dx, float *dgamma,
                              float *dbeta, int N, int D)
{
    GPU_ERR_CHK(cudaMemsetAsync(dgamma, 0, (size_t)D * sizeof(float), ember_stream()));
    GPU_ERR_CHK(cudaMemsetAsync(dbeta, 0, (size_t)D * sizeof(float), ember_stream()));
    k_layernorm_bwd<<<N, TBLOCK, TBLOCK * sizeof(float), ember_stream()>>>(
        dout, x, gamma, mean, rstd, dx, dgamma, dbeta, N, D);
    CUDA_POST_LAUNCH();
}

/* ================= heads permute (0,1,2,3) -> (0,2,1,3) ================= */

__global__ void k_permute_0213(const float *x, float *out, int d0, int d1, int d2, int d3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2 * d3;
    if (idx >= total) return;

    int e = idx % d3;
    int r = idx / d3;
    int c = r % d2;
    r /= d2;
    int b = r % d1;
    int a = r / d1;

    int out_idx = ((a * d2 + c) * d1 + b) * d3 + e;  // shape (d0,d2,d1,d3)
    out[out_idx] = x[idx];
}

extern "C" void permute_0213(const float *x, float *out, int d0, int d1, int d2, int d3)
{
    int total = d0 * d1 * d2 * d3;
    k_permute_0213<<<grid1(total), TBLOCK, 0, ember_stream()>>>(x, out, d0, d1, d2, d3);
    CUDA_POST_LAUNCH();
}

/* ================= batched GEMM ================= */
// Row-major C(n,m) = alpha * opA(A)(n,k) * opB(B)(k,m), per batch. cuBLAS is
// column-major, so we compute C^T = opB(B)^T * opA(A)^T by swapping operands
// (this is exactly the transpose trick matmul() uses, generalized to arbitrary
// transA/transB and a strided batch).
extern "C" void bmm(const float *a, const float *b, float *out, int batch, int n, int m, int k,
                    int transA, int transB, float alpha)
{
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transA ? n : k;
    int ldb = transB ? k : m;
    float beta = 0.0f;

    ember_cublas_prepare(tf_cublas());
    CUBLAS_ERR_CHK(cublasSgemmStridedBatched(tf_cublas(), opB, opA, m, n, k, &alpha, b, ldb,
                                             (long long)k * m, a, lda, (long long)n * k, &beta, out,
                                             m, (long long)n * m, batch));
}

/* ================= embedding ================= */

__global__ void k_embedding_fwd(const float *weight, const int *idx, float *out, int n_idx, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_idx * dim;
    if (i >= total) return;
    int t = i / dim;
    int d = i % dim;
    out[i] = weight[(size_t)idx[t] * dim + d];
}

extern "C" void embedding_fwd(const float *weight, const int *idx, float *out, int n_idx, int dim)
{
    int total = n_idx * dim;
    k_embedding_fwd<<<grid1(total), TBLOCK, 0, ember_stream()>>>(weight, idx, out, n_idx, dim);
    CUDA_POST_LAUNCH();
}

__global__ void k_embedding_bwd(const float *dout, const int *idx, float *dweight, int n_idx,
                                int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_idx * dim;
    if (i >= total) return;
    int t = i / dim;
    int d = i % dim;
    atomicAdd(&dweight[(size_t)idx[t] * dim + d], dout[i]);
}

extern "C" void embedding_bwd(const float *dout, const int *idx, float *dweight, int n_idx, int dim,
                              int vocab)
{
    GPU_ERR_CHK(cudaMemsetAsync(dweight, 0, (size_t)vocab * dim * sizeof(float), ember_stream()));
    int total = n_idx * dim;
    k_embedding_bwd<<<grid1(total), TBLOCK, 0, ember_stream()>>>(dout, idx, dweight, n_idx, dim);
    CUDA_POST_LAUNCH();
}
