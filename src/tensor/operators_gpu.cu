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
        k_##name##_tensor<<<grid(size), BLOCK_SIZE>>>(a, b, out, size);                     \
        CUDA_POST_LAUNCH();                                                                 \
    }

#define EMBER_SCALAR_OP(name, expr)                                                  \
    __global__ void k_##name##_scalar(const float *a, float b, float *out, int size) \
    {                                                                                \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                               \
        if (i < size) out[i] = (expr);                                               \
    }                                                                                \
    extern "C" void name##_scalar(const float *a, float b, float *out, int size)     \
    {                                                                                \
        k_##name##_scalar<<<grid(size), BLOCK_SIZE>>>(a, b, out, size);              \
        CUDA_POST_LAUNCH();                                                          \
    }

#define EMBER_UNARY_OP(name, expr)                                          \
    __global__ void k_##name##_tensor(const float *a, float *out, int size) \
    {                                                                       \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                      \
        if (i < size) out[i] = (expr);                                      \
    }                                                                       \
    extern "C" void name##_tensor(const float *a, float *out, int size)     \
    {                                                                       \
        k_##name##_tensor<<<grid(size), BLOCK_SIZE>>>(a, out, size);        \
        CUDA_POST_LAUNCH();                                                 \
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

#define EMBER_BROADCAST_OP(name, expr)                                                 \
    __global__ void k_##name##_broadcasted(const float *a, const float *b, float *out, \
                                           BcastMeta meta)                             \
    {                                                                                  \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
        if (i >= meta.total) return;                                                   \
        int rem = i, ia = 0, ib = 0;                                                   \
        for (int d = meta.ndim - 1; d >= 0; d--) {                                     \
            int coord = rem % meta.shape[d];                                           \
            rem /= meta.shape[d];                                                      \
            ia += coord * meta.sa[d];                                                  \
            ib += coord * meta.sb[d];                                                  \
        }                                                                              \
        out[i] = (expr);                                                               \
    }                                                                                  \
    extern "C" void name##_broadcasted(const float *a, const float *b, float *out,     \
                                       const int *shape, const int *strides_a,         \
                                       const int *strides_b, int ndim)                 \
    {                                                                                  \
        BcastMeta meta;                                                                \
        meta.ndim = ndim;                                                              \
        int total = 1;                                                                 \
        for (int d = 0; d < ndim; d++) {                                               \
            meta.shape[d] = shape[d];                                                  \
            meta.sa[d] = strides_a[d];                                                 \
            meta.sb[d] = strides_b[d];                                                 \
            total *= shape[d];                                                         \
        }                                                                              \
        meta.total = total;                                                            \
        k_##name##_broadcasted<<<grid(total), BLOCK_SIZE>>>(a, b, out, meta);          \
        CUDA_POST_LAUNCH();                                                            \
    }

#define EMBER_INPLACE_OP(name, expr)                                       \
    __global__ void k_##name##_inplace(float *a, const float *b, int size) \
    {                                                                      \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                     \
        if (i < size) a[i] = (expr);                                       \
    }                                                                      \
    extern "C" void name##_inplace(float *a, const float *b, int size)     \
    {                                                                      \
        k_##name##_inplace<<<grid(size), BLOCK_SIZE>>>(a, b, size);        \
        CUDA_POST_LAUNCH();                                                \
    }

#define EMBER_INPLACE_SCALAR_OP(name, expr)                                \
    __global__ void k_##name##_scalar_inplace(float *a, float b, int size) \
    {                                                                      \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                     \
        if (i < size) a[i] = (expr);                                       \
    }                                                                      \
    extern "C" void name##_scalar_inplace(float *a, float b, int size)     \
    {                                                                      \
        k_##name##_scalar_inplace<<<grid(size), BLOCK_SIZE>>>(a, b, size); \
        CUDA_POST_LAUNCH();                                                \
    }

#include "operators.def"

/* ---- non-element-wise operators ---- */
extern "C" {

void matmul(const float *a, const float *b, float *out, int n, int m, int k)
{
    // C(n x m) = A(n x k) * B(k x m), row-major. cuBLAS is column-major, so we
    // compute C^T = B^T * A^T by swapping the operands and using the fact that
    // a row-major (n x m) buffer is a column-major (m x n) buffer.
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_ERR_CHK(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, b, m, a,
                               k, &beta, out, m));
}

void matmul_batched(const float *a, const float *b, float *out, int batch, int n, int m, int k)
{
    // Same column-major mapping as matmul(), applied to `batch` independent
    // matrices with fixed strides between them.
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_ERR_CHK(cublasSgemmStridedBatched(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                             &alpha, b, m, (long long)k * m, a, k, (long long)n * k,
                                             &beta, out, m, (long long)n * m, batch));
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
    k_transpose<<<g, block>>>(a, out, n, m);
    CUDA_POST_LAUNCH();
}

__global__ void k_sum_reduce(const float *a, float *out, int size)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < size; i += stride) v += a[i];
    sdata[tid] = v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

float sum(const float *a, int size)
{
    // On-device reduction; only a single 4-byte scalar is copied to the host
    // (instead of the whole array).
    float *d_out = (float *)alloc_memory(sizeof(float));
    GPU_ERR_CHK(cudaMemset(d_out, 0, sizeof(float)));

    int blocks = grid(size);
    if (blocks > 256) blocks = 256;
    if (blocks < 1) blocks = 1;
    k_sum_reduce<<<blocks, BLOCK_SIZE>>>(a, d_out, size);
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

void sum_axis(const float *a, float *out, int outer_stride, int inner_stride, int axis_dim)
{
    int total = outer_stride * inner_stride;
    k_sum_axis<<<grid(total), BLOCK_SIZE>>>(a, out, outer_stride, inner_stride, axis_dim);
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
    k_max_axis<<<grid(total), BLOCK_SIZE>>>(a, out, outer_stride, inner_stride, axis_dim);
    CUDA_POST_LAUNCH();
}
}
