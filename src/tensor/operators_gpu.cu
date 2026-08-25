#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "../core/memory.h"
#include "../core/utils_gpu.cuh"
#include "operators.h"

#define BLOCK_SIZE 256

// Check for launch-configuration errors immediately after a kernel launch.
// Asynchronous execution errors surface at the next device synchronization
// (which happens on every host<->device transfer).
#define CUDA_POST_LAUNCH() GPU_ERR_CHK(cudaGetLastError())

static int grid(int n)
{
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// A single cuBLAS handle is created lazily and reused for the process lifetime,
// rather than being created and destroyed on every matmul.
static cublasHandle_t cublas_handle(void)
{
    static cublasHandle_t handle = NULL;
    if (!handle) {
        CUBLAS_ERR_CHK(cublasCreate(&handle));
    }
    return handle;
}

/* Element-wise CUDA kernels + host wrappers, generated from operators.def.
 * Kernels keep C++ linkage (called only within this file); the wrappers that
 * the rest of the library links against get C linkage. */
#define EMBER_BINARY_OP(name, expr)                                                         \
    __global__ void k_##name##_tensor(const float *a, const float *b, float *out, int size) \
    {                                                                                       \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                      \
        if (i < size) out[i] = (expr);                                                      \
    }                                                                                       \
    extern "C" void name##_tensor(const float *a, const float *b, float *out, int size)     \
    {                                                                                       \
        k_##name##_tensor<<<grid(size), BLOCK_SIZE, 0, ember_stream()>>>(a, b, out, size);  \
        CUDA_POST_LAUNCH();                                                                 \
    }

#define EMBER_SCALAR_OP(name, expr)                                                        \
    __global__ void k_##name##_scalar(const float *a, float b, float *out, int size)       \
    {                                                                                      \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                     \
        if (i < size) out[i] = (expr);                                                     \
    }                                                                                      \
    extern "C" void name##_scalar(const float *a, float b, float *out, int size)           \
    {                                                                                      \
        k_##name##_scalar<<<grid(size), BLOCK_SIZE, 0, ember_stream()>>>(a, b, out, size); \
        CUDA_POST_LAUNCH();                                                                \
    }

#define EMBER_UNARY_OP(name, expr)                                                      \
    __global__ void k_##name##_tensor(const float *a, float *out, int size)             \
    {                                                                                   \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                  \
        if (i < size) out[i] = (expr);                                                  \
    }                                                                                   \
    extern "C" void name##_tensor(const float *a, float *out, int size)                 \
    {                                                                                   \
        k_##name##_tensor<<<grid(size), BLOCK_SIZE, 0, ember_stream()>>>(a, out, size); \
        CUDA_POST_LAUNCH();                                                             \
    }

// Broadcast metadata is passed by value as a kernel parameter (no device
// allocation, host->device copy or synchronization per call). MAX_BCAST_DIMS
// bounds the rank; broadcasts are only used for low-rank tensors.
#define MAX_BCAST_DIMS 8
struct BcastMeta {
    int shape[MAX_BCAST_DIMS];
    int sa[MAX_BCAST_DIMS];
    int sb[MAX_BCAST_DIMS];
    int ndim;
    int total;
};

#define EMBER_BROADCAST_OP(name, expr)                                                           \
    __global__ void k_##name##_broadcasted(const float *a, const float *b, float *out,           \
                                           BcastMeta meta)                                       \
    {                                                                                            \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                           \
        if (i >= meta.total) return;                                                             \
        int rem = i, ia = 0, ib = 0;                                                             \
        for (int d = meta.ndim - 1; d >= 0; d--) {                                               \
            int coord = rem % meta.shape[d];                                                     \
            rem /= meta.shape[d];                                                                \
            ia += coord * meta.sa[d];                                                            \
            ib += coord * meta.sb[d];                                                            \
        }                                                                                        \
        out[i] = (expr);                                                                         \
    }                                                                                            \
    extern "C" void name##_broadcasted(const float *a, const float *b, float *out,               \
                                       const int *shape, const int *strides_a,                   \
                                       const int *strides_b, int ndim)                           \
    {                                                                                            \
        BcastMeta meta;                                                                          \
        meta.ndim = ndim;                                                                        \
        int total = 1;                                                                           \
        for (int d = 0; d < ndim; d++) {                                                         \
            meta.shape[d] = shape[d];                                                            \
            meta.sa[d] = strides_a[d];                                                           \
            meta.sb[d] = strides_b[d];                                                           \
            total *= shape[d];                                                                   \
        }                                                                                        \
        meta.total = total;                                                                      \
        k_##name##_broadcasted<<<grid(total), BLOCK_SIZE, 0, ember_stream()>>>(a, b, out, meta); \
        CUDA_POST_LAUNCH();                                                                      \
    }

#define EMBER_INPLACE_OP(name, expr)                                                   \
    __global__ void k_##name##_inplace(float *a, const float *b, int size)             \
    {                                                                                  \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
        if (i < size) a[i] = (expr);                                                   \
    }                                                                                  \
    extern "C" void name##_inplace(float *a, const float *b, int size)                 \
    {                                                                                  \
        k_##name##_inplace<<<grid(size), BLOCK_SIZE, 0, ember_stream()>>>(a, b, size); \
        CUDA_POST_LAUNCH();                                                            \
    }

#define EMBER_INPLACE_SCALAR_OP(name, expr)                                                   \
    __global__ void k_##name##_scalar_inplace(float *a, float b, int size)                    \
    {                                                                                         \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                        \
        if (i < size) a[i] = (expr);                                                          \
    }                                                                                         \
    extern "C" void name##_scalar_inplace(float *a, float b, int size)                        \
    {                                                                                         \
        k_##name##_scalar_inplace<<<grid(size), BLOCK_SIZE, 0, ember_stream()>>>(a, b, size); \
        CUDA_POST_LAUNCH();                                                                   \
    }

#include "operators.def"

/* ---- non-element-wise operators ---- */

// For very small matrices cuBLAS's fixed per-call launch/setup overhead
// dominates the actual FLOPs, so a trivial one-thread-per-output kernel is
// faster. Above this work threshold cuBLAS wins and is used instead.
#define MATMUL_SMALL_MAX_WORK (1 << 20)

__global__ void k_simple_matmul(const float *a, const float *b, float *out, int n, int m, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < m) {
        float acc = 0.0f;
        for (int l = 0; l < k; l++) acc += a[row * k + l] * b[l * m + col];
        out[row * m + col] = acc;
    }
}

extern "C" {

void matmul(const float *a, const float *b, float *out, int n, int m, int k)
{
    // C(n x m) = A(n x k) * B(k x m), row-major.
    if ((long long)n * m * k <= MATMUL_SMALL_MAX_WORK) {
        dim3 block(16, 16);
        dim3 g((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        k_simple_matmul<<<g, block, 0, ember_stream()>>>(a, b, out, n, m, k);
        CUDA_POST_LAUNCH();
        return;
    }

    // cuBLAS is column-major, so we compute C^T = B^T * A^T by swapping the
    // operands and using the fact that a row-major (n x m) buffer is a
    // column-major (m x n) buffer.
    const float alpha = 1.0f;
    const float beta = 0.0f;

    ember_cublas_prepare(cublas_handle());
    CUBLAS_ERR_CHK(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, b, m, a,
                               k, &beta, out, m));
}

void matmul_batched(const float *a, const float *b, float *out, int batch, int n, int m, int k)
{
    // Same column-major mapping as matmul(), applied to `batch` independent
    // matrices with fixed strides between them.
    const float alpha = 1.0f;
    const float beta = 0.0f;

    ember_cublas_prepare(cublas_handle());
    CUBLAS_ERR_CHK(cublasSgemmStridedBatched(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                             &alpha, b, m, (long long)k * m, a, k, (long long)n * k,
                                             &beta, out, m, (long long)n * m, batch));
}

__global__ void k_bias_add_rows(float *out, const float *bias, int n, int m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * m) out[idx] += bias[idx % m];
}

void matmul_bias(const float *a, const float *b, const float *bias, float *out, int n, int m, int k)
{
    matmul(a, b, out, n, m, k);
    k_bias_add_rows<<<grid(n * m), BLOCK_SIZE, 0, ember_stream()>>>(out, bias, n, m);
    CUDA_POST_LAUNCH();
}

__global__ void k_transpose(const float *a, float *out, int n, int m)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < m) out[j * n + i] = a[i * m + j];
}

void transpose(const float *a, float *out, int n, int m)
{
    dim3 block(16, 16);
    dim3 g((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    k_transpose<<<g, block, 0, ember_stream()>>>(a, out, n, m);
    CUDA_POST_LAUNCH();
}

__inline__ __device__ float warp_reduce_sum(float v)
{
    for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__global__ void k_sum_reduce(const float *a, float *out, int size)
{
    // Grid-stride load (multiple elements per thread), then a warp-shuffle
    // reduction within each warp, one shared slot per warp, then the first
    // warp reduces those and does a single atomicAdd per block.
    int tid = threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < size; i += stride) v += a[i];

    v = warp_reduce_sum(v);

    __shared__ float warp_sums[BLOCK_SIZE / 32];
    int lane = tid & (warpSize - 1);
    int wid = tid / warpSize;
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    if (wid == 0) {
        int nwarps = blockDim.x / warpSize;
        v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) atomicAdd(out, v);
    }
}

float sum(const float *a, int size)
{
    // On-device reduction; only a single 4-byte scalar is copied to the host
    // (instead of the whole array).
    float *d_out = (float *)alloc_memory(sizeof(float));
    GPU_ERR_CHK(cudaMemsetAsync(d_out, 0, sizeof(float), ember_stream()));

    int blocks = grid(size);
    if (blocks > 256) blocks = 256;
    if (blocks < 1) blocks = 1;
    k_sum_reduce<<<blocks, BLOCK_SIZE, 0, ember_stream()>>>(a, d_out, size);
    CUDA_POST_LAUNCH();

    float result = 0.0f;
    copy_from_device(&result, d_out, sizeof(float));
    free_memory(d_out);
    return result;
}

int sum_axis_product(const int *shape, int start, int end)
{
    int p = 1;
    for (int i = start; i < end; i++) p *= shape[i];
    return p;
}

__global__ void k_sum_axis(const float *a, float *out, int outer_stride, int inner_stride,
                           int axis_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_stride * inner_stride;
    if (idx >= total) return;

    int o = idx / inner_stride;
    int i = idx % inner_stride;
    int input_base = o * (axis_dim * inner_stride) + i;

    float s = 0.0f;
    for (int r = 0; r < axis_dim; r++) s += a[input_base + (r * inner_stride)];
    out[idx] = s;
}

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

// Contiguous last-axis reduction (inner_stride == 1): block-per-row with a
// block reduction instead of a per-thread serial row loop.
__global__ void k_sum_axis_rows(const float *a, float *out, int rows, int axis_dim)
{
    extern __shared__ float sh[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *ar = a + (size_t)row * axis_dim;
    float part = 0.0f;
    for (int j = threadIdx.x; j < axis_dim; j += blockDim.x) part += ar[j];
    float s = block_reduce_sum(part, sh);
    if (threadIdx.x == 0) out[row] = s;
}

// Few outputs but a long reduction axis (e.g. bias gradients: summing a
// (16384, 256) activation over axis 0): a single thread per output cannot
// fill the GPU, so split the axis over gridDim.y chunks and combine the
// partials with atomicAdd. `out` must be zeroed before launch.
__global__ void k_sum_axis_split(const float *a, float *out, int inner_stride, int axis_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inner_stride) return;
    int o = blockIdx.z;

    const float *base = a + ((size_t)o * axis_dim) * inner_stride + i;
    float s = 0.0f;
    for (int r = blockIdx.y; r < axis_dim; r += gridDim.y)
        s += base[(size_t)r * inner_stride];
    atomicAdd(&out[(size_t)o * inner_stride + i], s);
}

void sum_axis(const float *a, float *out, int outer_stride, int inner_stride, int axis_dim)
{
    int total = outer_stride * inner_stride;
    if (inner_stride == 1) {
        k_sum_axis_rows<<<outer_stride, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), ember_stream()>>>(
            a, out, outer_stride, axis_dim);
        CUDA_POST_LAUNCH();
        return;
    }
    if (total < (1 << 16) && axis_dim >= 256) {
        // Aim for >= 64k threads in flight; capped so chunks stay sizeable.
        int chunks = (1 << 16) / total;
        if (chunks > 64) chunks = 64;
        if (chunks > axis_dim) chunks = axis_dim;
        dim3 g(grid(inner_stride), chunks, outer_stride);
        GPU_ERR_CHK(cudaMemsetAsync(out, 0, (size_t)total * sizeof(float), ember_stream()));
        k_sum_axis_split<<<g, BLOCK_SIZE, 0, ember_stream()>>>(a, out, inner_stride, axis_dim);
        CUDA_POST_LAUNCH();
        return;
    }
    k_sum_axis<<<grid(total), BLOCK_SIZE, 0, ember_stream()>>>(a, out, outer_stride, inner_stride,
                                                               axis_dim);
    CUDA_POST_LAUNCH();
}

__global__ void k_max_axis(const float *a, float *out, int outer_stride, int inner_stride,
                           int axis_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_stride * inner_stride;
    if (idx >= total) return;

    int o = idx / inner_stride;
    int i = idx % inner_stride;
    int input_base = o * (axis_dim * inner_stride) + i;

    float m = a[input_base];
    for (int r = 1; r < axis_dim; r++) m = fmaxf(m, a[input_base + (r * inner_stride)]);
    out[idx] = m;
}

void max_axis(const float *a, float *out, int outer_stride, int inner_stride, int axis_dim)
{
    int total = outer_stride * inner_stride;
    k_max_axis<<<grid(total), BLOCK_SIZE, 0, ember_stream()>>>(a, out, outer_stride, inner_stride,
                                                               axis_dim);
    CUDA_POST_LAUNCH();
}

// y = 0.5*x*(1+t) with t = tanh(0.8x), so substituting it into
//     dy/dx = (1+t) * (0.5 + 0.8*(x - y))
// gives 0.5*(1+t)*(1 + 0.8*x*(1-t)) -- same value, but y no longer has to be
// read back from memory (this kernel is purely bandwidth-bound).
__global__ void k_gelu_bwd(const float *grad, const float *x, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = 0.8f;
    float xi = x[i];
    float t = tanhf(a * xi);
    out[i] = grad[i] * 0.5f * (1.0f + t) * (1.0f + a * xi * (1.0f - t));
}

void gelu_bwd(const float *grad, const float *x, float *out, int n)
{
    k_gelu_bwd<<<grid(n), BLOCK_SIZE, 0, ember_stream()>>>(grad, x, out, n);
    CUDA_POST_LAUNCH();
}

/* ---- fused optimizer steps ---- */
__global__ void k_adam_step(float *p, const float *g, float *m, float *v, int size, float lr,
                            float beta1, float mb1, float beta2, float mb2, float eps, float bc1,
                            float bc2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float gi = g[i];
    float mi = beta1 * m[i] + mb1 * gi;
    float vi = beta2 * v[i] + mb2 * gi * gi;
    m[i] = mi;
    v[i] = vi;
    p[i] -= lr * (mi * bc1) / (sqrtf(vi * bc2) + eps);
}

void adam_step(float *p, const float *g, float *m, float *v, int size, float lr, float beta1,
               float mb1, float beta2, float mb2, float eps, float bc1, float bc2)
{
    k_adam_step<<<grid(size), BLOCK_SIZE, 0, ember_stream()>>>(p, g, m, v, size, lr, beta1, mb1,
                                                               beta2, mb2, eps, bc1, bc2);
    CUDA_POST_LAUNCH();
}

__global__ void k_adamw_step(float *p, const float *g, float *m, float *v, int size, float lr,
                             float beta1, float mb1, float beta2, float mb2, float eps, float bc1,
                             float bc2, float weight_decay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float gi = g[i];
    float mi = beta1 * m[i] + mb1 * gi;
    float vi = beta2 * v[i] + mb2 * gi * gi;
    m[i] = mi;
    v[i] = vi;
    p[i] = p[i] * (1.0f - lr * weight_decay) - lr * (mi * bc1) / (sqrtf(vi * bc2) + eps);
}

void adamw_step(float *p, const float *g, float *m, float *v, int size, float lr, float beta1,
                float mb1, float beta2, float mb2, float eps, float bc1, float bc2,
                float weight_decay)
{
    k_adamw_step<<<grid(size), BLOCK_SIZE, 0, ember_stream()>>>(
        p, g, m, v, size, lr, beta1, mb1, beta2, mb2, eps, bc1, bc2, weight_decay);
    CUDA_POST_LAUNCH();
}

__global__ void k_sgd_step(float *p, const float *g, float *v, int size, float lr, float momentum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float vi = momentum * v[i] + g[i];
    v[i] = vi;
    p[i] -= lr * vi;
}

void sgd_step(float *p, const float *g, float *v, int size, float lr, float momentum)
{
    k_sgd_step<<<grid(size), BLOCK_SIZE, 0, ember_stream()>>>(p, g, v, size, lr, momentum);
    CUDA_POST_LAUNCH();
}

/* ---- grouped (multi-tensor) Adam: one launch for all parameters ---- */
__global__ void k_adam_step_group(float **ps, float **gs, float **ms, float **vs, const int *sizes,
                                  float lr, float beta1, float mb1, float beta2, float mb2,
                                  float eps, float bc1, float bc2)
{
    int pi = blockIdx.y;  // one grid row per parameter
    int n = sizes[pi];
    float *p = ps[pi];
    const float *g = gs[pi];
    float *m = ms[pi];
    float *v = vs[pi];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float gi = g[i];
        float mi = beta1 * m[i] + mb1 * gi;
        float vi = beta2 * v[i] + mb2 * gi * gi;
        m[i] = mi;
        v[i] = vi;
        p[i] -= lr * (mi * bc1) / (sqrtf(vi * bc2) + eps);
    }
}

void adam_step_group(float **ps, float **gs, float **ms, float **vs, const int *sizes, int nparams,
                     int max_size, float lr, float beta1, float mb1, float beta2, float mb2,
                     float eps, float bc1, float bc2)
{
    dim3 gdim(grid(max_size), nparams);
    k_adam_step_group<<<gdim, BLOCK_SIZE, 0, ember_stream()>>>(ps, gs, ms, vs, sizes, lr, beta1,
                                                               mb1, beta2, mb2, eps, bc1, bc2);
    CUDA_POST_LAUNCH();
}

__global__ void k_adamw_step_group(float **ps, float **gs, float **ms, float **vs, const int *sizes,
                                   float lr, float beta1, float mb1, float beta2, float mb2,
                                   float eps, float bc1, float bc2, float weight_decay)
{
    int pi = blockIdx.y;  // one grid row per parameter
    int n = sizes[pi];
    float *p = ps[pi];
    const float *g = gs[pi];
    float *m = ms[pi];
    float *v = vs[pi];
    float decay = 1.0f - lr * weight_decay;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float gi = g[i];
        float mi = beta1 * m[i] + mb1 * gi;
        float vi = beta2 * v[i] + mb2 * gi * gi;
        m[i] = mi;
        v[i] = vi;
        p[i] = p[i] * decay - lr * (mi * bc1) / (sqrtf(vi * bc2) + eps);
    }
}

void adamw_step_group(float **ps, float **gs, float **ms, float **vs, const int *sizes, int nparams,
                      int max_size, float lr, float beta1, float mb1, float beta2, float mb2,
                      float eps, float bc1, float bc2, float weight_decay)
{
    dim3 gdim(grid(max_size), nparams);
    k_adamw_step_group<<<gdim, BLOCK_SIZE, 0, ember_stream()>>>(
        ps, gs, ms, vs, sizes, lr, beta1, mb1, beta2, mb2, eps, bc1, bc2, weight_decay);
    CUDA_POST_LAUNCH();
}

/* ---- capturable Adam/AdamW (on-device step counter + bias correction) ---- */
__global__ void k_adam_bias(float *t, float *bc, float beta1, float beta2)
{
    float tt = t[0] + 1.0f;
    t[0] = tt;
    bc[0] = 1.0f / (1.0f - powf(beta1, tt));
    bc[1] = 1.0f / (1.0f - powf(beta2, tt));
}

void adam_bias_update(float *t, float *bc, float beta1, float beta2)
{
    k_adam_bias<<<1, 1, 0, ember_stream()>>>(t, bc, beta1, beta2);
    CUDA_POST_LAUNCH();
}

__global__ void k_adam_step_dev(float *p, const float *g, float *m, float *v, int size, float lr,
                                float beta1, float mb1, float beta2, float mb2, float eps,
                                const float *bc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float gi = g[i];
    float mi = beta1 * m[i] + mb1 * gi;
    float vi = beta2 * v[i] + mb2 * gi * gi;
    m[i] = mi;
    v[i] = vi;
    p[i] -= lr * (mi * bc[0]) / (sqrtf(vi * bc[1]) + eps);
}

void adam_step_dev(float *p, const float *g, float *m, float *v, int size, float lr, float beta1,
                   float mb1, float beta2, float mb2, float eps, const float *bc)
{
    k_adam_step_dev<<<grid(size), BLOCK_SIZE, 0, ember_stream()>>>(p, g, m, v, size, lr, beta1, mb1,
                                                                   beta2, mb2, eps, bc);
    CUDA_POST_LAUNCH();
}

__global__ void k_adamw_step_dev(float *p, const float *g, float *m, float *v, int size, float lr,
                                 float beta1, float mb1, float beta2, float mb2, float eps,
                                 const float *bc, float weight_decay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float gi = g[i];
    float mi = beta1 * m[i] + mb1 * gi;
    float vi = beta2 * v[i] + mb2 * gi * gi;
    m[i] = mi;
    v[i] = vi;
    p[i] = p[i] * (1.0f - lr * weight_decay) - lr * (mi * bc[0]) / (sqrtf(vi * bc[1]) + eps);
}

void adamw_step_dev(float *p, const float *g, float *m, float *v, int size, float lr, float beta1,
                    float mb1, float beta2, float mb2, float eps, const float *bc,
                    float weight_decay)
{
    k_adamw_step_dev<<<grid(size), BLOCK_SIZE, 0, ember_stream()>>>(
        p, g, m, v, size, lr, beta1, mb1, beta2, mb2, eps, bc, weight_decay);
    CUDA_POST_LAUNCH();
}
}
